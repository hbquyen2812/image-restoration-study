"""Export and run the reviewed experimental HAction model on one image."""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from PIL import Image
import tensorflow as tf
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from hamiltonian_repair import HamiltonianAction

from project_paths import ROOT, DIVA_TEST


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'outputs/diva_research/experiments/experimental_H_action_smoke.h5')
    parser.add_argument('--image', type=Path, default=DIVA_TEST / 'data/Set12/01.png')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('--out must be a new directory so previous results are preserved')
    args.out.mkdir(parents=True)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    model = tf.keras.models.load_model(args.checkpoint, compile=False)
    layer = model.get_layer('proj_coef')
    assert isinstance(layer, HamiltonianAction)
    clean = np.asarray(Image.open(args.image).convert('L'), np.float32) / 255
    noisy = (clean + np.random.RandomState(0).normal(0, 25/255, clean.shape)).astype(np.float32)
    output = model(noisy[None,...,None], training=False).numpy()[0,...,0]
    assert np.isfinite(output).all()

    exported = args.out / 'DIVA_HAction_reviewed.h5'
    model.save(exported, include_optimizer=False)
    restored = tf.keras.models.load_model(exported, compile=False)
    reloaded_output = restored(noisy[None,...,None], training=False).numpy()[0,...,0]
    reload_error = float(np.max(np.abs(output-reloaded_output)))
    assert reload_error == 0
    np.savez_compressed(args.out / 'sample_arrays.npz', clean=clean, noisy=noisy, denoised=output)
    for name, array in [('clean',clean), ('noisy',noisy), ('denoised',output)]:
        Image.fromarray(np.round(np.clip(array,0,1)*255).astype(np.uint8)).save(args.out / f'{name}.png')
    result = dict(status='passed',model='experimental HAction; not an exact DIVA reproduction',
        source_checkpoint=str(args.checkpoint.resolve()),
        source_checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        exported_checkpoint=str(exported.resolve()),
        exported_checkpoint_sha256=hashlib.sha256(exported.read_bytes()).hexdigest(),
        layer_source_sha256=hashlib.sha256(Path(__file__).with_name('hamiltonian_repair.py').read_bytes()).hexdigest(),
        image=str(args.image.resolve()),sigma=25,seed=0,boundary=layer.boundary,
        params=model.count_params(),new_training_steps=0,
        prior_training_steps=None,training_history_note='Read the training run metadata; not inferred from an arbitrary input checkpoint',
        psnr_noisy=float(peak_signal_noise_ratio(clean,noisy,data_range=1)),
        psnr_denoised=float(peak_signal_noise_ratio(clean,output,data_range=1)),
        ssim_denoised=float(structural_similarity(clean,output,data_range=1)),
        serialization_max_error=reload_error)
    (args.out / 'run.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (args.out / 'README_VI.md').write_text(
        '# Bản mẫu Hamiltonian đã rà soát\n\n'
        'Model thử nghiệm giữ trọng số checkpoint đầu vào; không huấn luyện thêm. '
        'xuất lại với lớp HamiltonianAction đã sửa dtype, kiểm tra shape và serialize điều kiện biên. '
        'Đây là khối Hf thử nghiệm, chưa tái lập chính xác phép chiếu theo patch của DIVA.\n\n'
        'Kết quả ảnh mẫu và xác nhận lưu/nạp nằm trong run.json. PSNR/SSIM tính trên mảng chưa clip; '
        'PNG chỉ clip khi xuất ảnh. Mặc định Dirichlet giữ đúng hành vi của model đã train.\n\n'
        'Khi nạp HDF5 cần import HamiltonianAction từ methods/haction/hamiltonian_repair.py trước.\n',
        encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
