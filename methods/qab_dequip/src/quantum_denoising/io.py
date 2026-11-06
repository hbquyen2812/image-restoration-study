"""Image loading, raw numerical output, and preview export."""

from pathlib import Path
import numpy as np
from PIL import Image
from scipy.io import loadmat
from .core import FloatArray, as_float


def load_image(path: str | Path, variable: str = "I") -> FloatArray:
    path = Path(path)
    if path.suffix.lower() == ".mat":
        data = loadmat(path)[variable]
    elif path.suffix.lower() == ".npy":
        data = np.load(path, allow_pickle=False)
    else:
        data = np.asarray(Image.open(path))
        if data.ndim == 3:
            data = data[:, :, 0]  # same channel selection as Demo_DeQuIP.m
    return as_float(data, 2)


def resize_square(image: FloatArray, side: int) -> FloatArray:
    if image.shape == (side, side):
        return image.copy()
    return np.asarray(Image.fromarray(image.astype(np.float32)).resize(
        (side, side), Image.Resampling.BILINEAR), dtype=np.float64)


def prepare_image(image: FloatArray, side: int, method: str) -> FloatArray:
    if side < 1:
        raise ValueError("Image size must be positive.")
    if method == "resize":
        return resize_square(image, side)
    if method != "center-crop" or min(image.shape) < side:
        raise ValueError("Center crop requires each image dimension >= requested size.")
    top, left = (image.shape[0] - side) // 2, (image.shape[1] - side) // 2
    return image[top:top + side, left:left + side].copy()


def save_preview(path: Path, image: FloatArray, peak: float = 255.0) -> None:
    # Metrics ALWAYS use raw floats, only display PNGs are clipped/quantized.
    Image.fromarray(np.rint(np.clip(image / peak, 0, 1) * 255).astype(np.uint8)).save(path)

