"""Explicit, shared metric conventions for both implementations."""

import numpy as np
from scipy.signal import convolve2d
from .core import FloatArray, as_float


def psnr(reference: FloatArray, estimate: FloatArray, peak: float = 255.0) -> float:
    mse = float(np.mean((reference - estimate) ** 2))
    return float(10 * np.log10(peak ** 2 / mse)) if mse else float("inf")


def ssim(reference: FloatArray, estimate: FloatArray, peak: float = 255.0,
         window_size: int = 3, sigma: float = 1.5) -> float:
    """Wang SSIM, Gaussian 3x3 valid window, population moments (upstream demos).

    Benchmark uses peak=255 for every image. Pass reference.max() for the
    original demos' SSIM convention. No clipping or quantization is applied.
    """
    if min(reference.shape) < window_size:
        raise ValueError("Image too small for the SSIM window.")
    x = np.arange(window_size) - (window_size - 1) / 2
    window = np.exp(-(x[:, None] ** 2 + x[None, :] ** 2) / (2 * sigma ** 2))
    window /= window.sum()
    def average(a):
        return convolve2d(a, window, mode="valid")
    mu1, mu2 = average(reference), average(estimate)
    var1 = average(reference ** 2) - mu1 ** 2
    var2 = average(estimate ** 2) - mu2 ** 2
    cov = average(reference * estimate) - mu1 * mu2
    c1, c2 = (0.01 * peak) ** 2, (0.03 * peak) ** 2
    return float(np.mean(((2 * mu1 * mu2 + c1) * (2 * cov + c2)) /
                         ((mu1 ** 2 + mu2 ** 2 + c1) * (var1 + var2 + c2))))


def quality(reference: FloatArray, estimate: FloatArray, peak: float = 255.0) -> dict:
    reference = as_float(reference, reference.ndim)
    estimate = as_float(estimate, reference.ndim)
    if reference.shape != estimate.shape:
        raise ValueError("Metric inputs must have the same shape.")
    mse = float(np.mean((reference - estimate) ** 2))
    power = float(np.mean(reference ** 2))
    result = {"mse": mse, "psnr_db": psnr(reference, estimate, peak),
              "snr_db": float(10 * np.log10(power / mse)) if mse else float("inf")}
    if reference.ndim == 2:
        result["ssim"] = ssim(reference, estimate, peak)
    return result

