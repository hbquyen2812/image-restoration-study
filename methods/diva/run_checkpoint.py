"""Run the upstream DIVA pretrained model with reproducible noise and saved evidence.

Uses the original test notebook's tensor conversion and PSNR/SNR functions.
Compatibility changes: CLI paths, deterministic sorted inputs/seed, explicit SSIM
data_range=1, uint8 PNG export, and headless figures. Raw float arrays are retained.
"""

import argparse
import ast
import contextlib
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            if not stream.closed:
                stream.write(value)
                stream.flush()
        return len(value)

    def flush(self):
        for stream in self.streams:
            if not stream.closed:
                stream.flush()

    def close(self):
        self.flush()


def run(args, out):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["MPLCONFIGDIR"] = str(out / "matplotlib_cache")
    print("Loading TensorFlow and image libraries...", flush=True)
    import h5py
    import keras
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    import skimage
    from skimage.io import imread
    from skimage.metrics import structural_similarity
    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    root = args.repo.resolve()
    test_dir = root / "DIVA test code for image denoising"
    notebook = test_dir / "test_code.ipynb"
    checkpoint = test_dir / "models" / "DIVA_models_sigma_10_to_100" / f"model_sigma{args.sigma}.hdf5"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"No supplied checkpoint for sigma={args.sigma}: {checkpoint}")
    files = sorted((test_dir / "data" / args.dataset).glob("*.png"))
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise ValueError("No test images found")

    # Execute only these four original, self-contained utility definitions.
    nb = json.loads(notebook.read_text(encoding="utf-8"))
    source = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    names = {"to_tensor", "from_tensor", "psnr", "snr"}
    selected = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in selected} != names:
        raise RuntimeError("Upstream notebook utility functions changed")
    namespace = {"np": np}
    module = ast.Module(body=selected, type_ignores=[])
    exec(compile(module, str(notebook), "exec"), namespace)
    to_tensor, from_tensor = namespace["to_tensor"], namespace["from_tensor"]
    psnr, snr = namespace["psnr"], namespace["snr"]
    (out / "upstream_utilities.py").write_text(ast.unparse(module) + "\n", encoding="utf-8")
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    (out / "pip_freeze.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True), encoding="utf-8")
    (out / "git_diff.patch").write_text(subprocess.check_output(["git", "-C", str(root), "diff", "HEAD"], text=True), encoding="utf-8")
    (out / "runner_snapshot.py").write_bytes(Path(__file__).read_bytes())
    with h5py.File(checkpoint, "r") as h5:
        saved_keras = str(h5.attrs.get("keras_version", "unknown"))
        model_config = json.loads(h5.attrs["model_config"])
    save_json(out / "model_config.json", model_config)

    metadata = {
        "status": "running", "started_at": datetime.now(timezone(timedelta(hours=7))).isoformat(),
        "repo": str(root), "upstream": "https://github.com/SayantanDutta95/DIVA", "commit": commit,
        "notebook": str(notebook), "notebook_sha256": sha256(notebook),
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_keras_version": saved_keras,
        "python": platform.python_version(), "tensorflow": tf.__version__, "keras": keras.__version__,
        "numpy": np.__version__, "skimage": skimage.__version__, "matplotlib": matplotlib.__version__,
        "platform": platform.platform(), "device": "CPU", "threads": args.threads,
        "dataset": args.dataset, "sigma_0_255": args.sigma, "seed": args.seed,
        "noise": "numpy.random.seed(seed); N(0, sigma/255); no input clipping; float32",
        "metrics": "Full grayscale image; data_range=1; original float32 PSNR; skimage SSIM default window=7; no output clipping",
        "image_export": "Only PNG previews are clipped to [0,1] and rounded to uint8; NPZ retains raw float32 arrays",
        "input_images": [{"name": p.name, "sha256": sha256(p)} for p in files],
        "command": subprocess.list2cmdline([sys.executable, *sys.argv]),
    }
    save_json(out / "run.json", metadata)
    print(f"DIVA commit {commit}; TensorFlow {tf.__version__}; CPU")
    print(f"Checkpoint: {checkpoint.name}; saved Keras {saved_keras}")
    print(f"Dataset: {args.dataset}, images={len(files)}, sigma={args.sigma}, seed={args.seed}")
    started = time.perf_counter()
    model = tf.keras.models.load_model(checkpoint, compile=False)
    model.summary(print_fn=lambda line: print(line))
    metadata["model_parameters"] = model.count_params()
    metadata["model_load_seconds"] = time.perf_counter() - started
    save_json(out / "run.json", metadata)
    (out / "images").mkdir()
    (out / "arrays").mkdir()
    rows, gallery = [], []

    def save_png(path, pixels):
        Image.fromarray(np.rint(np.clip(pixels, 0, 1) * 255).astype(np.uint8)).save(path)

    for path in files:
        clean = np.array(imread(path), dtype=np.float32) / 255.0
        if clean.ndim != 2:
            raise ValueError(f"Expected grayscale image: {path}")
        noisy = (clean + np.random.normal(0, args.sigma / 255.0, clean.shape)).astype(np.float32)
        infer_start = time.perf_counter()
        denoised = from_tensor(model.predict(to_tensor(noisy), verbose=0))
        elapsed = time.perf_counter() - infer_start
        if denoised.shape != clean.shape or not np.isfinite(denoised).all():
            raise ValueError(f"Invalid model output for {path.name}")
        row = {
            "image": path.name, "height": clean.shape[0], "width": clean.shape[1],
            "noisy_psnr_db": float(psnr(clean, noisy)),
            "denoised_psnr_db": float(psnr(clean, denoised)),
            "noisy_ssim": float(structural_similarity(clean, noisy, data_range=1.0)),
            "denoised_ssim": float(structural_similarity(clean, denoised, data_range=1.0)),
            "noisy_snr_db": float(snr(clean, noisy)), "inference_seconds": elapsed,
            "output_min": float(denoised.min()), "output_max": float(denoised.max()),
        }
        row["psnr_gain_db"] = row["denoised_psnr_db"] - row["noisy_psnr_db"]
        rows.append(row)
        print(f"{path.name}: PSNR {row['noisy_psnr_db']:.4f} -> {row['denoised_psnr_db']:.4f} dB; "
              f"SSIM {row['noisy_ssim']:.4f} -> {row['denoised_ssim']:.4f}; {elapsed:.3f} s", flush=True)
        with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        np.savez_compressed(out / "arrays" / f"{path.stem}.npz", clean=clean, noisy=noisy, denoised=denoised)
        for label, pixels in [("clean", clean), ("noisy", noisy), ("denoised", denoised)]:
            save_png(out / "images" / f"{path.stem}_{label}.png", pixels)
        h, w = clean.shape
        y0, x0 = h // 2 - 48, w // 2 - 48
        fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))
        titles = ["Clean reference", f"Noisy | {row['noisy_psnr_db']:.2f} dB", f"DIVA | {row['denoised_psnr_db']:.2f} dB"]
        for col, (pixels, title) in enumerate(zip([clean, noisy, denoised], titles)):
            axes[0, col].imshow(pixels, cmap="gray", vmin=0, vmax=1)
            axes[0, col].set_title(title)
            axes[1, col].imshow(pixels[y0:y0+96, x0:x0+96], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            axes[1, col].set_title("Center crop 96 x 96")
            axes[0, col].axis("off")
            axes[1, col].axis("off")
        fig.suptitle(f"DIVA | {args.dataset}/{path.name} | Gaussian sigma={args.sigma} | seed={args.seed}")
        fig.tight_layout()
        fig.savefig(out / f"comparison_{path.stem}.png", dpi=140)
        plt.close(fig)
        gallery.append((row, noisy, denoised))

    summary = {"count": len(rows), "dataset": args.dataset, "sigma": args.sigma, "seed": args.seed}
    for key in ["noisy_psnr_db", "denoised_psnr_db", "noisy_ssim", "denoised_ssim", "psnr_gain_db", "inference_seconds"]:
        summary["mean_" + key] = float(np.mean([row[key] for row in rows]))
    summary["total_inference_seconds"] = sum(row["inference_seconds"] for row in rows)
    summary["wall_seconds_including_model_load_and_exports"] = time.perf_counter() - started
    summary["all_images_improved_psnr"] = all(row["psnr_gain_db"] > 0 for row in rows)
    save_json(out / "summary.json", summary)
    nrows = (len(gallery) + 2) // 3
    fig, axes = plt.subplots(nrows, 3, figsize=(15, nrows * 3.5), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (row, noisy, denoised) in zip(axes.flat, gallery):
        ax.imshow(np.hstack([noisy, denoised]), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"{row['image']} | Noisy {row['noisy_psnr_db']:.2f} -> DIVA {row['denoised_psnr_db']:.2f} dB", fontsize=10)
    fig.suptitle(f"DIVA / {args.dataset} / sigma={args.sigma} / seed={args.seed}\nLeft: noisy | Right: denoised", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(out / "overview.png", dpi=130)
    plt.close(fig)
    metadata["status"] = "completed"
    metadata["finished_at"] = datetime.now(timezone(timedelta(hours=7))).isoformat()
    save_json(out / "run.json", metadata)
    print("SUMMARY " + json.dumps(summary))
    print("Results: " + str(out))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2] / 'external/DIVA',
                        help="Upstream DIVA clone containing notebooks, test data and checkpoints")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dataset", default="Set12")
    parser.add_argument("--sigma", type=int, choices=[10, 15, 25, 75, 100], default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    with (out / "console.log").open("w", encoding="utf-8") as logfile:
        with contextlib.redirect_stdout(Tee(sys.stdout, logfile)), contextlib.redirect_stderr(Tee(sys.stderr, logfile)):
            try:
                run(args, out)
            except BaseException:
                traceback.print_exc()
                save_json(out / "failure.json", {"status": "failed", "traceback": traceback.format_exc()})
                raise


if __name__ == "__main__":
    main()
