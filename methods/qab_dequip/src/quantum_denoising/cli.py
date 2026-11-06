"""Command-line entry point; run `python -m quantum_denoising --help`."""

import argparse
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from .benchmark import compare_matlab, run_benchmark
from .dequip import DeQuIPConfig, dequip_denoise
from .io import load_image, save_preview
from .qab import QABImageConfig, qab_image


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="QAB / De-QuIP Python CPU denoising and reproducible benchmarking")
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("benchmark", "benchmark-signal"):
        sub = commands.add_parser(name)
        if name == "benchmark":
            sub.add_argument("--image", required=True, help="PNG, MAT, or NPY; RGB uses first channel")
            sub.add_argument("--sizes", nargs="+", type=int, default=[64, 128])
            sub.add_argument("--preprocess", choices=["resize", "center-crop"], default="resize")
            sub.add_argument("--algorithms", nargs="+", choices=["qab", "dequip"], default=["qab", "dequip"])
            sub.add_argument("--peak", type=float, default=255.0, help="Fixed peak for shared metrics, default 255")
        else:
            sub.add_argument("--signal", required=True, help="MAT file containing S")
        sub.add_argument("--variable", default="I" if name == "benchmark" else "S")
        sub.add_argument("--out", required=True, help="New output directory (existing directory is rejected)")
        sub.add_argument("--repo-root", default="external", help="Reference clones under the project root")
        sub.add_argument("--repeats", type=int, default=3)
        sub.add_argument("--warmups", type=int, default=1)
        sub.add_argument("--threads", type=int, default=1)
        sub.add_argument("--seed", type=int, default=2)
        sub.add_argument("--snr", type=float, default=22.0 if name == "benchmark" else 15.0)
        sub.add_argument("--noise", choices=["gaussian", "heteroscedastic"],
                         default="gaussian" if name == "benchmark" else "heteroscedastic")
        sub.add_argument("--qab-slopes", type=int, default=20 if name == "benchmark" else 200)
    sub = commands.add_parser("denoise")
    sub.add_argument("--image", required=True)
    sub.add_argument("--algorithm", choices=["qab", "dequip"], default="dequip")
    sub.add_argument("--out", required=True, help="New NPY file; also writes a PNG preview")
    sub.add_argument("--threshold", type=float, help="De-QuIP: coefficient magnitude; QAB: 1-based basis index")
    sub.add_argument("--slope", type=float, default=128.0)
    sub.add_argument("--threads", type=int, default=1)
    sub = commands.add_parser("compare-matlab")
    sub.add_argument("folder")
    sub.add_argument("--tolerance", type=float, default=1e-6)
    return root


def main(argv=None) -> None:
    root = parser()
    args = root.parse_args(argv)
    try:
        if args.command.startswith("benchmark"):
            print(f"Saved: {run_benchmark(args)}")
        elif args.command == "compare-matlab":
            results = compare_matlab(args.folder, args.tolerance)
            print(json.dumps(results, indent=2))
            if not all(row["within_tolerance"] for row in results):
                raise SystemExit(1)
        else:
            out = Path(args.out).resolve()
            if out.suffix.lower() != ".npy":
                raise ValueError("--out must end in .npy (raw floating-point output).")
            if out.exists() or out.with_suffix(".png").exists():
                raise ValueError("Output already exists; choose a new output path.")
            if args.threads < 1:
                raise ValueError("threads must be >= 1.")
            noisy = load_image(args.image)
            with threadpool_limits(limits=args.threads):
                if args.algorithm == "dequip":
                    config = DeQuIPConfig(threshold=25.0 if args.threshold is None else args.threshold)
                    restored = dequip_denoise(noisy, config)
                else:
                    if args.threshold is None or not args.threshold.is_integer():
                        raise ValueError("QAB without a clean reference needs an integer --threshold and optional --slope.")
                    restored = qab_image(noisy, QABImageConfig(threshold=int(args.threshold), slope=args.slope))
            out.parent.mkdir(parents=True, exist_ok=True)
            np.save(out, restored)
            save_preview(out.with_suffix(".png"), restored)
            print(f"Saved: {out}")
    except (ValueError, FileExistsError, FileNotFoundError, KeyError) as error:
        root.error(str(error))

