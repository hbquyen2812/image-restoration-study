"""Small standalone API example. Run after installing the project.

python examples/denoise_image.py --input noisy.png --output restored.npy
"""

import argparse
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from quantum_denoising import DeQuIPConfig, dequip_denoise
from quantum_denoising.io import load_image, save_preview


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=25)
    args = parser.parse_args()
    destination = Path(args.output)
    if destination.suffix != ".npy":
        parser.error("Output must end in .npy")
    if destination.exists() or destination.with_suffix(".png").exists():
        parser.error("Output already exists")
    image = load_image(args.input)
    config = DeQuIPConfig(threshold=args.threshold)
    with threadpool_limits(limits=1):
        restored = dequip_denoise(image, config)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.save(destination, restored)
    save_preview(destination.with_suffix(".png"), restored)
    print(f"Saved raw output: {destination.resolve()}")


if __name__ == "__main__":
    main()

