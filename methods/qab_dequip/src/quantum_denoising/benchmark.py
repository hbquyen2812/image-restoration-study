"""Reproducible benchmarks; time the entire denoiser, excluding I/O/metrics."""

from dataclasses import asdict
from datetime import datetime
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
from time import perf_counter
from zoneinfo import ZoneInfo
import numpy as np
import scipy
from scipy.io import loadmat, savemat
from threadpoolctl import threadpool_info, threadpool_limits

from . import __version__
from .dequip import DeQuIPConfig, dequip_denoise
from .io import load_image, prepare_image, save_preview
from .metrics import quality, psnr, ssim
from .noise import add_noise
from .qab import QABImageConfig, QABSignalConfig, qab_image, qab_signal


def array_digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(array, dtype="<f8").tobytes(order="C")).hexdigest()


def _git(repo: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True,
                                       stderr=subprocess.DEVNULL, timeout=10).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def environment(repo_root: Path, threads: int) -> dict:
    cpu = platform.processor()
    if sys.platform == "win32":
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0]
        except OSError:
            pass
    return {"timestamp": datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).isoformat(),
            "python": sys.version, "platform": platform.platform(), "cpu": cpu,
            "logical_cpus": os.cpu_count(), "numpy": np.__version__, "scipy": scipy.__version__,
            "package": __version__, "threads": threads, "threadpools": threadpool_info(),
            "sources": {name: {"commit": _git(repo_root / name, "rev-parse", "HEAD"),
                                "changes": _git(repo_root / name, "status", "--short")}
                        for name in ("QAB-denoising", "De-QuIP-Denoising")}}


def measure(call, repeats: int, warmups: int) -> tuple[np.ndarray, dict]:
    warmup_times, times = [], []
    for _ in range(warmups):
        start = perf_counter()
        call()
        warmup_times.append(perf_counter() - start)
    result = None
    for iteration in range(repeats):
        start = perf_counter()
        result = call()
        times.append(perf_counter() - start)
        print(f"    repeat {iteration + 1}/{repeats}: {times[-1]:.3f} s", flush=True)
    return result, {"seconds": times, "median_seconds": statistics.median(times),
                    "min_seconds": min(times), "max_seconds": max(times), "warmup_seconds": warmup_times}


def _plot_case(path, clean, noisy, outputs, rows, peak):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    items = [("Clean", clean), ("Noisy", noisy)] + list(outputs.items())
    fig, axes = plt.subplots(1, len(items), figsize=(4 * len(items), 4.5))
    for ax, (name, data) in zip(axes, items):
        if clean.ndim == 2:
            ax.imshow(data, cmap="gray", vmin=min(0, clean.min()), vmax=peak)
            ax.axis("off")
        else:
            ax.plot(data, linewidth=1)
            ax.set_ylim(min(clean.min(), noisy.min()) - 1, max(clean.max(), noisy.max()) + 1)
        row = next((r for r in rows if r["algorithm"] == name), None)
        suffix = ""
        if row:
            suffix = f"\n{row['quality']['psnr_db']:.2f} dB | {row['timing']['median_seconds']:.3f} s"
        ax.set_title(name + suffix)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _write_report(out: Path, metadata: dict, rows: list[dict]) -> None:
    (out / "results.json").write_text(json.dumps({"environment": metadata, "results": rows},
                                                indent=2, ensure_ascii=False), encoding="utf-8")
    with (out / "results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["case", "algorithm", "median_seconds", "min_seconds", "max_seconds",
                         "input_psnr_db", "output_psnr_db", "input_ssim", "output_ssim"])
        for row in rows:
            writer.writerow([row["case"], row["algorithm"], row["timing"]["median_seconds"],
                             row["timing"]["min_seconds"], row["timing"]["max_seconds"],
                             row["input_quality"]["psnr_db"], row["quality"]["psnr_db"],
                             row["input_quality"].get("ssim"), row["quality"].get("ssim")])
    lines = ["# Kết quả chạy Python", "", f"CPU: {metadata['cpu']}",
             f"Python: {platform.python_version()}, NumPy: {np.__version__}, SciPy: {scipy.__version__}",
             f"BLAS threads: {metadata['threads']}. Thời gian là median, không gồm I/O và metric.", "",
             "| Dữ liệu | Thuật toán | Giây | PSNR nhiễu | PSNR ra | SSIM ra |",
             "|---|---|---:|---:|---:|---:|"]
    for row in rows:
        ss = row["quality"].get("ssim")
        ssim_text = f"{ss:.4f}" if ss is not None else "—"
        lines.append(f"| {row['case']} | {row['algorithm']} | {row['timing']['median_seconds']:.3f} | "
                     f"{row['input_quality']['psnr_db']:.3f} | {row['quality']['psnr_db']:.3f} | "
                     f"{ssim_text} |")
    lines += ["", "QAB-oracle dùng ảnh/tín hiệu sạch để chọn ngưỡng; không dùng bảng này để xếp hạng công bằng QAB với De-QuIP.",
              "Ảnh: PSNR dùng peak cố định ghi trong JSON (mặc định 255), SSIM Gaussian 3×3 sigma=1.5, valid/population; metric trên float chưa clip.",
              "Tín hiệu: PSNR peak=max(abs(clean)). Xem JSON cho noise, tham số và các lần đo.",
              "", "Chưa có số đo MATLAB trong báo cáo này. Không suy ra Python nhanh/chậm hơn MATLAB từ các số trên.",
              "Chạy matlab/benchmark_upstream.m với các file case_*.mat, rồi lệnh compare-matlab để đối chiếu."]
    (out / "REPORT_VI.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_benchmark(args) -> Path:
    if args.repeats < 1 or args.warmups < 0 or args.threads < 1:
        raise ValueError("Use repeats >= 1, warmups >= 0 and threads >= 1.")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = Path(args.image if args.command == "benchmark" else args.signal).resolve()
    rows = []
    with threadpool_limits(limits=args.threads):
        metadata = environment(Path(args.repo_root).resolve(), args.threads)
        metadata.update({"source_file": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                         "protocol": "full denoiser call, warmed, serial CPU, I/O and metrics excluded",
                         "preprocess": ("center-crop: integer centered region, no interpolation"
                                        if getattr(args, "preprocess", None) == "center-crop" else
                                        "Pillow float32 bilinear, then float64; exact arrays exported to MAT"
                                        if args.command == "benchmark" else "original signal unchanged"),
                         "matlab_status": "not_run"})
        if args.command == "benchmark":
            original = load_image(source, args.variable)
            cases = [(f"image_{size}", prepare_image(original, size, args.preprocess)) for size in args.sizes]
        else:
            original = np.asarray(loadmat(source)[args.variable], dtype=float).ravel()
            cases = [(f"signal_{original.size}", original)]
        for case, clean in cases:
            noisy = add_noise(clean, args.snr, args.seed, args.noise)
            peak = args.peak if clean.ndim == 2 else float(np.abs(clean).max())
            if not np.isfinite(peak) or peak <= 0:
                raise ValueError("Metric peak must be finite and positive.")
            input_quality = quality(clean, noisy, peak)
            payload = {"clean": clean, "noisy": noisy, "peak": peak, "repeats": args.repeats,
                       "warmups": args.warmups, "threads": args.threads, "case_name": case,
                       "input_sha256": array_digest(noisy), "mode": "image" if clean.ndim == 2 else "signal"}
            outputs, case_rows = {}, []
            if clean.ndim == 2:
                methods = []
                if "qab" in args.algorithms:
                    config = QABImageConfig(slopes=args.qab_slopes)
                    methods.append(("QAB-oracle", "qab", config, lambda c=config: qab_image(noisy, c, reference=clean)))
                if "dequip" in args.algorithms:
                    config = DeQuIPConfig()
                    methods.append(("De-QuIP", "dequip", config, lambda c=config: dequip_denoise(noisy, c)))
            else:
                config = QABSignalConfig(slopes=args.qab_slopes)
                methods = [("QAB-signal-oracle", "qab_signal", config, lambda: qab_signal(noisy, config, reference=clean))]
            for name, key, config, call in methods:
                print(f"{case}: {name}, warmups={args.warmups}, repeats={args.repeats}", flush=True)
                restored, timing = measure(call, args.repeats, args.warmups)
                metrics = quality(clean, restored, peak)
                if key.startswith("qab"):
                    metrics["upstream_psnr_db"] = psnr(clean, restored, float(max(clean.max(), restored.max())))
                if clean.ndim == 2:
                    metrics["upstream_ssim"] = ssim(clean, restored, float(clean.max()))
                row = {"case": case, "algorithm": name, "key": key, "shape": list(clean.shape),
                       "noise": args.noise, "snr_db": args.snr, "seed": args.seed, "peak": peak,
                       "oracle": key.startswith("qab"), "config": asdict(config), "timing": timing,
                       "input_quality": input_quality, "quality": metrics, "input_sha256": array_digest(noisy)}
                rows.append(row)
                case_rows.append(row)
                outputs[name] = restored
                payload.update({key + "_python": restored, key + "_python_seconds": timing["seconds"],
                                key + "_config": {k: v for k, v in asdict(config).items() if v is not None}})
                np.save(out / f"{case}_{key}.npy", restored)
                if clean.ndim == 2:
                    save_preview(out / f"{case}_{key}.png", restored, peak)
                print(f"    PSNR {input_quality['psnr_db']:.3f} -> {metrics['psnr_db']:.3f} dB", flush=True)
                _write_report(out, metadata, rows)
            savemat(out / f"case_{case}.mat", payload, do_compression=True)
            if clean.ndim == 2:
                save_preview(out / f"{case}_clean.png", clean, peak)
                save_preview(out / f"{case}_noisy.png", noisy, peak)
            _plot_case(out / f"{case}_comparison.png", clean, noisy, outputs, case_rows, peak)
        _write_report(out, metadata, rows)
    return out


def compare_matlab(folder: str, tolerance: float = 1e-6) -> list[dict]:
    """Read measured MATLAB results; never estimate missing MATLAB timings."""
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative.")
    folder = Path(folder)
    comparisons = []
    for path in sorted(folder.glob("case_*.mat")):
        baseline_path = path.with_name(path.stem + "_matlab.mat")
        if path.stem.endswith("_matlab") or not baseline_path.exists():
            continue
        case = loadmat(path, simplify_cells=True)
        baseline = loadmat(baseline_path, simplify_cells=True)
        if baseline["input_sha256"] != case["input_sha256"]:
            raise ValueError(f"Input digest mismatch: {path.name}")
        for field in ("clean", "noisy"):
            if field not in baseline or not np.array_equal(case[field], baseline[field]):
                raise ValueError(f"MATLAB {field} is missing or differs from the Python input.")
        if array_digest(np.asarray(case["noisy"])) != case["input_sha256"]:
            raise ValueError("Stored noisy array does not match its digest.")
        for key in ("qab", "dequip", "qab_signal"):
            if key + "_python" not in case:
                continue
            if key + "_matlab" not in baseline:
                raise ValueError(f"Missing {key} in {baseline_path.name}")
            if baseline.get(key + "_config") != case.get(key + "_config") or key + "_config" not in baseline:
                raise ValueError(f"MATLAB configuration mismatch for {key}.")
            python_output = np.asarray(case[key + "_python"])
            matlab_output = np.asarray(baseline[key + "_matlab"])
            if python_output.shape != matlab_output.shape or not np.isfinite(matlab_output).all():
                raise ValueError(f"Invalid MATLAB output for {key}.")
            error = float(np.max(np.abs(python_output - matlab_output)))
            py_seconds = float(np.median(case[key + "_python_seconds"]))
            mat_seconds = float(np.median(baseline[key + "_seconds"]))
            for timings in (case[key + "_python_seconds"], baseline[key + "_seconds"]):
                values = np.asarray(timings)
                if not values.size or not np.isfinite(values).all() or np.any(values <= 0):
                    raise ValueError("Every timing must be a finite positive measurement.")
            comparisons.append({"case": path.stem, "algorithm": key, "max_abs_error": error,
                                "within_tolerance": error <= tolerance, "tolerance": tolerance,
                                "python_seconds": py_seconds, "matlab_seconds": mat_seconds,
                                "python_over_matlab": py_seconds / mat_seconds,
                                "matlab_version": baseline.get("matlab_version"),
                                "python_quality": quality(case["clean"], python_output, float(case["peak"])),
                                "matlab_quality": quality(case["clean"], matlab_output, float(case["peak"]))})
    if not comparisons:
        raise ValueError("No matching MATLAB measurements. Run matlab/benchmark_upstream.m first.")
    (folder / "matlab_comparison.json").write_text(json.dumps(comparisons, indent=2), encoding="utf-8")
    return comparisons

