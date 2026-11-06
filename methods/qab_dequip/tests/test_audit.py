"""Additional independent checks of patch overlap and oracle early stopping."""
import numpy as np
import pytest
from quantum_denoising import QABImageConfig, qab_image
from quantum_denoising.qab import _reconstruct
from test_numerics import literal_qab, literal_heavi


def test_qab_overlapping_patch_oracle():
    rng = np.random.default_rng(44)
    clean = rng.uniform(5, 20, (12, 12))
    noisy = clean + rng.normal(size=clean.shape)
    cfg = QABImageConfig(patch_size=8, stride=4, slopes=3)
    expected, count = np.zeros_like(clean), np.zeros_like(clean)
    for i in (0, 4):
        for j in (0, 4):
            region = np.s_[i:i+8, j:j+8]
            expected[region] += literal_qab(clean[region], noisy[region], cfg)
            count[region] += 1
    np.testing.assert_allclose(qab_image(noisy, cfg, reference=clean), expected/count, atol=2e-11)


@pytest.mark.parametrize("seed", [1, 7, 42])
def test_coefficient_search_matches_snr_search_with_actual_early_stop(seed):
    rng = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(rng.normal(size=(128, 128)))
    target = np.zeros(128)
    target[:12] = rng.normal(size=12) * 10
    noisy_coeff = target + rng.normal(size=128) * 2
    clean, noisy = basis @ target, basis @ noisy_coeff
    cfg = QABImageConfig(slopes=3, threshold_step=2, patience=3)
    coeff = np.linalg.solve(basis, noisy)
    best_snr, expected, stops = 0, np.zeros(128), 0
    for width in 2.0 ** np.linspace(1, 5, 3):
        local_snr, failures = 0, 0
        for threshold in range(1, 129, 2):
            candidate = basis @ (coeff * literal_heavi(np.arange(1,129)-threshold+2, width))
            snr = 10*np.log10(np.sum(clean**2)/np.sum((candidate-clean)**2))
            failures = failures+1 if local_snr > snr else 0
            local_snr = max(local_snr, snr)
            if snr > best_snr:
                best_snr, expected = snr, candidate
            if failures >= 3:
                stops += 1
                break
    assert stops > 0
    actual = _reconstruct(basis, noisy, clean, cfg, (1,5), 3)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
