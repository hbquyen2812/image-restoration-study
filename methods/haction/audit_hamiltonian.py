"""Executable audit of the unmodified upstream Hamiltonian definitions."""
import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Subtract
from tensorflow.keras.models import Model
from tensorflow.python.keras.utils import conv_utils
import tensorflow.keras.backend as K

from project_paths import ROOT, TRAINING_NOTEBOOK, DIVA_TEST
OUT = ROOT / 'outputs/diva_research/experiments/audit'
OUT.mkdir(parents=True, exist_ok=True)
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.keras.utils.set_random_seed(0)
nb = json.loads(TRAINING_NOTEBOOK.read_text(encoding='utf-8'))
source = ''.join(nb['cells'][0]['source'])
fragment = source[source.index('class Hamiltonian_Conv2D'):source.index('def findLastCheckpoint')]
(OUT / 'upstream_definitions.py').write_text(fragment, encoding='utf-8')
args = SimpleNamespace(kernel_size=5)
exec(compile(fragment, 'upstream_definitions.py', 'exec'), globals())
result = {'tensorflow':tf.__version__, 'source_changed': False}

# Build a single original layer, using actual variables for potential and interaction.
potential = tf.Variable(tf.ones([3,3,4]), name='potential_probe')
interaction = tf.Variable(tf.ones([3,3,4]), name='interaction_probe')
layer = Hamiltonian_Conv2D(4, 5, kernel_3=potential, kernel_4=interaction, padding='same')
features = tf.random.normal([1,40,40,4])
layer.build(features.shape)
result['kernel_after_build'] = {'name': layer.kernel.name, 'shape': layer.kernel.shape.as_list(), 'type':type(layer.kernel).__name__}
# Some Keras versions renamed this private field; do not change the source graph.
if not hasattr(layer, '_tf_data_format'):
    layer._tf_data_format = 'NHWC'
    result['runtime_compatibility_alias'] = '_tf_data_format=NHWC'
with tf.GradientTape() as tape:
    output = layer(features)
    loss = tf.reduce_mean(tf.square(output))
weights = layer.trainable_weights
grads = tape.gradient(loss, weights)
result['gradients'] = [{'name':v.name, 'shape':v.shape.as_list(), 'gradient_is_none':g is None,
                        'norm':None if g is None else float(tf.linalg.global_norm([g]))} for v,g in zip(weights,grads)]
before = output.numpy()
potential.assign_add(tf.ones_like(potential)*100)
interaction.assign_add(tf.ones_like(interaction)*100)
kinetic = next(v for v in weights if 'kernel_h' in v.name)
kinetic.assign_add(tf.ones_like(kinetic)*100)
result['max_output_change_after_changing_H_terms_by_100'] = float(np.max(np.abs(before-layer(features).numpy())))

# Check the complete original graph and which branch is pruned.
try:
    original = DIVA2D(depth=10,filters=64)
    original(tf.zeros([1,40,40,1]))
    result['full_model'] = {'status':'ran','params':original.count_params(),
                            'layers':[l.name for l in original.layers]}
    with tf.GradientTape() as tape:
        loss=tf.reduce_mean(tf.square(original(tf.random.normal([1,40,40,1]))))
    grads=tape.gradient(loss,original.trainable_weights)
    result['full_model']['no_gradient_weights']=[v.name for v,g in zip(original.trainable_weights,grads) if g is None]
except Exception as exc:
    result['full_model'] = {'status':'failed','error':f'{type(exc).__name__}: {exc}'}

# Try precisely the suggested one-line removal, with no shape redesign.
minimal = fragment.replace('super(Hamiltonian_Conv2D, self).build(input_shape)', '# removed base build for diagnostic only')
scope=dict(globals())
exec(compile(minimal,'one_line_removal.py','exec'),scope)
try:
    probe=scope['Hamiltonian_Conv2D'](4,5,kernel_3=tf.ones([1,3,3,4]),kernel_4=tf.ones([1,3,3,4]),padding='same')
    probe._tf_data_format='NHWC'
    probe(features)
    result['one_line_removal']={'status':'ran','kernel_shape':probe.kernel.shape.as_list()}
except Exception as exc:
    result['one_line_removal']={'status':'failed','error':f'{type(exc).__name__}: {exc}'}
checkpoints=DIVA_TEST/'models/DIVA_models_sigma_10_to_100'
import h5py
result['checkpoint_graphs']=[]
result['checkpoint_graph_audit_status'] = 'available' if checkpoints.is_dir() else 'skipped: external checkpoints not installed'
for path in sorted(checkpoints.glob('*.hdf5')):
    with h5py.File(path,'r') as f:
        config=json.loads(f.attrs['model_config'])
    classes=[l['class_name'] for l in config['config']['layers']]
    result['checkpoint_graphs'].append({'file':path.name,'has_custom_hamiltonian':any('Hamilton' in c for c in classes),'classes':sorted(set(classes))})
(OUT/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
