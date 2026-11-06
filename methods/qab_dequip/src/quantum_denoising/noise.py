"""Seeded, exactly SNR-scaled noise; arrays are exported for MATLAB reuse."""

import numpy as np
from .core import FloatArray


def add_noise(clean: FloatArray, snr_db: float, seed: int, model: str = "gaussian") -> FloatArray:
    if not np.isfinite(snr_db):
        raise ValueError("snr_db must be finite.")
    noise = np.random.default_rng(seed).standard_normal(clean.shape)
    if model == "heteroscedastic":
        # QAB demo labels this Poisson; it is Gaussian with sqrt(abs(I)) scaling.
        noise *= np.sqrt(np.abs(clean))
    elif model != "gaussian":
        raise ValueError("noise must be gaussian or heteroscedastic.")
    power = float(np.mean(clean ** 2))
    if power == 0:
        return clean.copy()
    noise *= np.sqrt(power * 10 ** (-snr_db / 10) / np.mean(noise ** 2))
    return clean + noise

