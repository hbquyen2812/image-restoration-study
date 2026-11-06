"""QAB image and signal ports.

The upstream demos select thresholds using the clean reference (oracle tuning).
Supply reference=... to reproduce that experiment, or set fixed threshold/slope
to denoise without a reference. Reference data never constructs the basis.
"""

from dataclasses import dataclass
from typing import Callable
import numpy as np
from numpy.typing import ArrayLike

from .core import (FloatArray, as_float, image_basis, signal_basis, nonnegative,
                   patch_starts, positive_int, smooth_heaviside)


@dataclass(frozen=True)
class QABImageConfig:
    patch_size: int = 64
    stride: int = 32
    slopes: int = 20                    # upstream Ms
    weight: float = 3.0                 # upstream pds, directly hbar^2/(2m)
    gaussian_variance: float = 7.5      # upstream sg is variance, not stddev
    threshold_step: int = 12
    patience: int = 20
    threshold: int | None = None        # 1-based l in upstream
    slope: float = 128.0                # only used with a fixed threshold


@dataclass(frozen=True)
class QABSignalConfig:
    slopes: int = 200
    weight: float = 0.4
    gaussian_variance: float = 15.0
    threshold_step: int = 2
    threshold: int | None = None
    slope: float = 32.0


def _validate(config: QABImageConfig | QABSignalConfig, size: int) -> None:
    positive_int("slopes", config.slopes)
    positive_int("threshold_step", config.threshold_step)
    nonnegative("weight", config.weight, strict=True)
    nonnegative("gaussian_variance", config.gaussian_variance, strict=True)
    nonnegative("slope", config.slope, strict=True)
    if config.threshold is not None:
        positive_int("threshold", config.threshold)
        if config.threshold > size:
            raise ValueError("threshold must not exceed the number of basis vectors.")


def _blur(data: FloatArray, variance: float) -> FloatArray:
    """Finite sampled Gaussian + circular FFT convolution, without renormalizing."""
    n = data.shape[0]
    coordinates = np.arange(-n / 2, n / 2)
    squared = coordinates ** 2
    if data.ndim == 2:
        squared = squared[:, None] + squared[None, :]
    kernel = np.exp(-squared / (2 * variance)) / (2 * np.pi * variance) ** (data.ndim / 2)
    return np.fft.ifftn(np.fft.fftn(np.fft.ifftshift(kernel)) * np.fft.fftn(data)).real


def _reconstruct(basis: FloatArray, noisy: FloatArray, reference: FloatArray | None,
                 config: QABImageConfig | QABSignalConfig,
                 slope_exponents: tuple[int, int], patience: int | None) -> FloatArray:
    coefficients = basis.T @ noisy
    indices = np.arange(1, noisy.size + 1)
    if config.threshold is not None:
        taper = smooth_heaviside(indices - config.threshold + 2, config.slope)
        return basis @ (coefficients * taper)
    if reference is None:
        raise ValueError("QAB oracle mode needs reference; otherwise set config.threshold and config.slope.")

    # Orthonormality gives ||Psi*a - reference||^2 = ||a - Psi.T*reference||^2.
    # This preserves the upstream SNR search and early stopping but avoids a
    # dense reconstruction for EVERY candidate. Only reconstruct the winner.
    target = basis.T @ reference
    reference_energy = float(reference @ reference)
    best_error = reference_energy        # upstream best SNR starts at 0 dB
    best_coefficients = np.zeros_like(coefficients)
    # MATLAB linspace(a,b,1) returns b, whereas NumPy returns a.
    exponents = ([slope_exponents[1]] if config.slopes == 1 else
                 np.linspace(*slope_exponents, config.slopes))
    for width in np.exp2(exponents):
        local_error = reference_energy
        failures = 0
        for threshold in range(1, noisy.size + 1, config.threshold_step):
            candidate = coefficients * smooth_heaviside(indices - threshold + 2, width)
            residual = candidate - target
            error = float(residual @ residual)
            failures = failures + 1 if error > local_error else 0
            local_error = min(local_error, error)
            if error < best_error:
                best_error, best_coefficients = error, candidate
            if patience is not None and failures >= patience:
                break
    return basis @ best_coefficients


def qab_image(noisy: ArrayLike, config: QABImageConfig = QABImageConfig(), *,
              reference: ArrayLike | None = None,
              progress: Callable[[int, int], None] | None = None) -> FloatArray:
    """Denoise a 2D grayscale image, preserving its units and without clipping.

    Defaults match image_denoising_QAB.m. Rectangles and non-aligned edges are
    additionally supported by covering the final edge; use square multiples
    of 32 (>=64) when comparing with the unmodified MATLAB source.
    """
    noisy = as_float(noisy, 2)
    n = config.patch_size
    positive_int("patch_size", n, 2)
    positive_int("stride", config.stride)
    positive_int("patience", config.patience)
    if n % 2 or config.stride > n or min(noisy.shape) < n:
        raise ValueError("Use an even patch size <= image dimensions and stride <= patch size.")
    _validate(config, n * n)
    clean = None if reference is None else as_float(reference, 2)
    if clean is not None and clean.shape != noisy.shape:
        raise ValueError("reference and noisy must have the same shape.")
    if config.threshold is None and clean is None:
        raise ValueError("QAB oracle mode needs reference; otherwise set config.threshold and config.slope.")
    positions = [(i, j) for i in patch_starts(noisy.shape[0], n, config.stride)
                 for j in patch_starts(noisy.shape[1], n, config.stride)]
    output, counts = np.zeros_like(noisy), np.zeros_like(noisy)
    for done, (i, j) in enumerate(positions, 1):
        region = np.s_[i:i + n, j:j + n]
        patch = noisy[region]
        scale = float(patch.max())       # intentionally max, NOT max(abs): upstream
        if scale == 0:
            if np.any(patch):
                raise ValueError("QAB cannot normalize a nonzero patch whose maximum is zero.")
            restored = np.zeros_like(patch)
        else:
            normalized = patch / scale
            _, basis = image_basis(_blur(normalized, config.gaussian_variance), config.weight)
            target = None if clean is None else (clean[region] / scale).ravel(order="F")
            restored = _reconstruct(basis, normalized.ravel(order="F"), target, config,
                                    (7, 11), config.patience).reshape((n, n), order="F") * scale
        output[region] += restored
        counts[region] += 1
        if progress is not None:
            progress(done, len(positions))
    return output / counts


def qab_signal(noisy: ArrayLike, config: QABSignalConfig = QABSignalConfig(), *,
               reference: ArrayLike | None = None) -> FloatArray:
    """Denoise an even-length 1D signal with the periodic upstream Hamiltonian."""
    noisy = as_float(noisy, 1)
    if noisy.size < 4 or noisy.size % 2:
        raise ValueError("The upstream signal smoothing requires an even length >= 4.")
    _validate(config, noisy.size)
    clean = None if reference is None else as_float(reference, 1)
    if clean is not None and clean.shape != noisy.shape:
        raise ValueError("reference and noisy must have the same shape.")
    if config.threshold is None and clean is None:
        raise ValueError("QAB oracle mode needs reference; otherwise set config.threshold and config.slope.")
    scale = float(np.abs(noisy).max())
    if scale == 0:
        return np.zeros_like(noisy)
    normalized = noisy / scale
    _, basis = signal_basis(_blur(normalized, config.gaussian_variance), config.weight)
    target = None if clean is None else clean / scale
    return _reconstruct(basis, normalized, target, config, (1, 10), None) * scale

