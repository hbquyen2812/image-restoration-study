"""Bounded paired training and CPU timing; never updates official checkpoints."""
import argparse
import json
import os
from pathlib import Path
import time

from project_paths import ROOT, DIVA_TEST, TRAIN400_DIR
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['MPLBACKEND']='Agg'
os.environ['MPLCONFIGDIR']=str(ROOT/'outputs/diva_research/mpl_cache')
import numpy as np
from PIL import Image
import tensorflow as tf
from skimage.metrics import peak_signal_noise_ratio,structural_similarity
from hamiltonian_repair import HamiltonianAction,original_model,repaired_model

tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.get_logger().setLevel('ERROR')
OUT=ROOT/'outputs/diva_research/experiments'
OUT.mkdir(parents=True,exist_ok=True)

def write(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2),encoding='utf-8')

def step_function(model):
    optimizer=tf.keras.optimizers.Adam(.001)
    @tf.function
    def step(noisy,clean):
        with tf.GradientTape() as tape:
            prediction=model(noisy,training=True)
            loss=tf.reduce_mean(tf.square(prediction-clean))
        gradients=tape.gradient(loss,model.trainable_variables)
        optimizer.apply_gradients([(g,v) for g,v in zip(gradients,model.trainable_variables) if g is not None])
        return loss
    return step

def checks():
    tf.keras.utils.set_random_seed(4)
    x=tf.Variable(tf.random.uniform([2,40,40,4]))
    v=tf.Variable(tf.random.uniform(x.shape))
    interaction=tf.Variable(tf.random.uniform(x.shape))
    layer=HamiltonianAction()
    with tf.GradientTape() as tape:
        y=layer([x,v,interaction])
        loss=tf.reduce_mean(y*y)
    vars=[x,v,interaction,layer.raw_alpha]
    grads=tape.gradient(loss,vars)
    norms=[float(tf.linalg.global_norm([g])) if g is not None else None for g in grads]
    assert all(n is not None and n>0 for n in norms)
    before=y.numpy()
    interaction.assign_add(tf.ones_like(interaction)*.1)
    change=float(np.max(np.abs(layer([x,v,interaction]).numpy()-before)))
    assert change>0
    model=repaired_model(depth=4,filters=16)
    with tf.GradientTape() as tape:
        pred=model(tf.random.uniform([2,40,40,1]),training=True)
        loss=tf.reduce_mean(pred**2)
    grads=tape.gradient(loss,model.trainable_variables)
    assert all(g is not None for g in grads)
    grad_rows=[{'name':w.name,'norm':float(tf.linalg.global_norm([g]))} for w,g in zip(model.trainable_variables,grads)]
    # Sizes that fail for the old pooled 3x3 assumption must remain valid.
    shapes=[list(model(tf.zeros([b,h,w,1])).shape) for b,h,w in [(1,40,40),(2,64,48),(1,128,128)]]
    probe=tf.random.uniform([1,40,40,1])
    before=model(probe).numpy()
    model.save(OUT/'repair_roundtrip.h5',include_optimizer=False)
    reloaded=tf.keras.models.load_model(OUT/'repair_roundtrip.h5',compile=False)
    reload_error=float(np.max(np.abs(reloaded(probe).numpy()-before)))
    assert reload_error<1e-6
    result={'status':'passed','operator_gradient_norms':dict(zip(['features','potential','interaction','raw_alpha'],norms)),
            'output_change_when_interaction_changes':change,'all_model_weights_have_gradient':True,
            'model_gradients':grad_rows,'valid_output_shapes':shapes,'serialization_max_error':reload_error}
    write('repair_checks.json',result)
    print('REPAIR CHECKS',json.dumps(result),flush=True)

def train(steps, data_dir=TRAIN400_DIR, test_dir=DIVA_TEST/'data/Set12'):
    if steps < 1:
        raise ValueError('--steps must be >= 1')
    files=sorted(data_dir.glob('*.png'))
    if len(files) != 64:
        raise ValueError(f'Expected 64 Train400 PNG files in {data_dir}; run scripts/download_train_subset.py or pass --train-dir')
    test_files=sorted(test_dir.glob('*.png'))
    if len(test_files) != 12:
        raise ValueError(f'Expected 12 Set12 PNG files in {test_dir}; clone upstream DIVA or pass --test-dir')
    images=[np.asarray(Image.open(p).convert('L'),np.float32)/255 for p in files]
    rng=np.random.RandomState(23)
    clean_batches=[]
    noisy_batches=[]
    # Fixed same batches for both models; last 16 images are held out by image.
    for _ in range(steps):
        batch=[]
        for _ in range(16):
            img=images[rng.randint(48)]
            y=rng.randint(img.shape[0]-39); x=rng.randint(img.shape[1]-39)
            patch=img[y:y+40,x:x+40]
            patch=np.rot90(patch,rng.randint(4))
            if rng.rand()<.5: patch=np.fliplr(patch)
            batch.append(patch)
        clean=np.array(batch)[...,None]
        noisy=(clean+rng.normal(0,25/255,clean.shape)).astype(np.float32)
        clean_batches.append(clean); noisy_batches.append(noisy)
    val_clean=np.stack([img[70:110,70:110] for img in images[48:]])[...,None]
    val_noisy=(val_clean+rng.normal(0,25/255,val_clean.shape)).astype(np.float32)
    tf.keras.utils.set_random_seed(7)
    original=original_model(depth=4,filters=16)
    shared={l.name:l.get_weights() for l in original.layers if l.get_weights() and l.name!='proj_coef'}
    results=[]
    for label in ['original_overwritten','experimental_H_action']:
        tf.keras.utils.set_random_seed(7)
        model=original if label=='original_overwritten' else repaired_model(depth=4,filters=16)
        if label!='original_overwritten':
            for layer in model.layers:
                if layer.name in shared: layer.set_weights(shared[layer.name])
        train_step=step_function(model)
        rows=[]
        start=time.perf_counter()
        initial_mse=float(np.mean((model(val_noisy,training=False).numpy()-val_clean)**2))
        for index,(noisy,clean) in enumerate(zip(noisy_batches,clean_batches),1):
            loss=float(train_step(noisy,clean).numpy())
            assert np.isfinite(loss)
            if index==1 or index%25==0 or index==steps:
                val_mse=float(np.mean((model(val_noisy,training=False).numpy()-val_clean)**2))
                rows.append({'step':index,'train_mse':loss,'val_mse':val_mse,'seconds':time.perf_counter()-start})
                print(label,index,'train MSE',round(loss,6),'val MSE',round(val_mse,6),flush=True)
        train_seconds=time.perf_counter()-start
        model.save(OUT/f'{label}_smoke.h5',include_optimizer=False)
        test_rng=np.random.RandomState(0)
        metrics=[]
        for path in test_files:
            clean=np.array(Image.open(path),np.float32)/255
            noisy=(clean+test_rng.normal(0,25/255,clean.shape)).astype(np.float32)
            denoised=model(noisy[None,...,None],training=False).numpy()[0,...,0]
            metrics.append({'image':path.name,'noisy_psnr':float(peak_signal_noise_ratio(clean,noisy,data_range=1)),
                            'psnr':float(peak_signal_noise_ratio(clean,denoised,data_range=1)),
                            'ssim':float(structural_similarity(clean,denoised,data_range=1))})
            if path.stem=='01':
                np.savez_compressed(OUT/f'{label}_cameraman.npz',clean=clean,noisy=noisy,denoised=denoised)
        result={'variant':label,'parameters':model.count_params(),'initial_val_mse':initial_mse,
                'final_val_mse':rows[-1]['val_mse'],'train_seconds':train_seconds,'history':rows,'set12_metrics':metrics,
                'set12_mean_psnr':float(np.mean([r['psnr'] for r in metrics])),
                'set12_mean_ssim':float(np.mean([r['ssim'] for r in metrics]))}
        results.append(result)
        write('paired_training.json',{'status':'completed' if len(results)==2 else 'running',
              'seed_model':7,'seed_data':23,'sigma':25,'steps':steps,'batch_size':16,'patch':40,'depth':4,'filters':16,
              'train_images':[p.name for p in files[:48]],'validation_images':[p.name for p in files[48:]],
              'shared_initial_weights_copied':True,'scope':'small one-seed diagnostic; not paper reproduction or accuracy ranking',
              'results':results})
        print('FINISHED',label,result['set12_mean_psnr'],result['train_seconds'],flush=True)

def profile():
    rows=[]
    for label,builder in [('original_overwritten',original_model),('experimental_H_action',repaired_model)]:
        for batch in [32,128]:
            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(19)
            model=builder(depth=10,filters=64)
            clean=tf.random.uniform([batch,40,40,1])
            noisy=clean+tf.random.normal(clean.shape,stddev=25/255)
            step=step_function(model)
            print('PROFILE START',label,'batch',batch,flush=True)
            warmup_start=time.perf_counter()
            for _ in range(2): float(step(noisy,clean).numpy())
            warmup=time.perf_counter()-warmup_start
            samples=[]
            for _ in range(5):
                start=time.perf_counter(); loss=float(step(noisy,clean).numpy()); samples.append(time.perf_counter()-start)
                assert np.isfinite(loss)
                print('timing',label,batch,samples[-1],flush=True)
            median=float(np.median(samples))
            row={'variant':label,'batch_size':batch,'patch_size':40,'depth':10,'filters':64,'params':model.count_params(),
                 'warmup_two_steps_seconds':warmup,'samples_seconds':samples,'median_step_seconds':median,
                 'min_step_seconds':min(samples),'max_step_seconds':max(samples),
                 'estimate_hours_60epochs_1000steps':median*60000/3600,
                 'estimate_hours_100epochs_1000steps':median*100000/3600}
            rows.append(row)
            write('training_profile.json',{'status':'completed' if len(rows)==4 else 'running','device':'CPU','threads':4,
                  'tensorflow':tf.__version__,'measurement':'forward + backward + Adam, eager caller / tf.function step, synchronization via .numpy()',
                  'excludes':'data generation and loading, validation, checkpoint writes; no full training performed','results':rows})

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['checks','train','profile'])
    parser.add_argument('--steps',type=int,default=300)
    parser.add_argument('--train-dir',type=Path,default=TRAIN400_DIR)
    parser.add_argument('--test-dir',type=Path,default=DIVA_TEST/'data/Set12')
    parser.add_argument('--out',type=Path,help='New output directory; default is outputs/diva_research/experiments')
    options=parser.parse_args()
    if options.out is not None:
        OUT=options.out.resolve()
        OUT.mkdir(parents=True,exist_ok=False)
    if options.mode=='checks': checks()
    elif options.mode=='train': train(options.steps,options.train_dir,options.test_dir)
    else: profile()
