"""Independent small literal references, NOT measurements from a MATLAB runtime."""

from dataclasses import replace
import numpy as np
import pytest
from threadpoolctl import threadpool_limits
from quantum_denoising import QABImageConfig, QABSignalConfig, DeQuIPConfig, qab_image, qab_signal, dequip_denoise
from quantum_denoising.core import image_laplacian, image_basis, signal_basis, smooth_heaviside
from quantum_denoising.dequip import extract_patches, aggregate_patches, total_interactions, window_centers
from quantum_denoising.metrics import quality, ssim
from quantum_denoising.noise import add_noise


@pytest.fixture(autouse=True)
def single_thread():
    with threadpool_limits(limits=1):
        yield


def literal_hamiltonian(potential, weight):
    """Literal diagonal + boundary edits from f_Hamil2D.m/f_ondes2D.m."""
    vector = np.asarray(potential).ravel(order="F")
    size = vector.size
    side = int(np.sqrt(size))
    h = np.diag(vector + 4 * weight)
    h -= np.diag(np.full(size - 1, weight), 1) + np.diag(np.full(size - 1, weight), -1)
    h -= np.diag(np.full(size - side, weight), side) + np.diag(np.full(size - side, weight), -side)
    for block in range(side):
        h[side * block, side * block] -= weight
        h[side * (block + 1) - 1, side * (block + 1) - 1] -= weight
    for line in range(side):
        h[line, line] -= weight
        h[size - 1 - line, size - 1 - line] -= weight
    for inter in range(1, side):
        h[side * inter, side * inter - 1] = 0
        h[side * inter - 1, side * inter] = 0
    return h


def literal_heavi(x, width):
    result = np.zeros(len(x))
    for i, value in enumerate(x):
        if value <= -width:
            result[i] = 1
        elif value < width:
            result[i] = 0.5 * (1 - value / width - np.sin(np.pi * value / width) / np.pi)
    return result


def literal_qab(clean, noisy, config, signal=False):
    n = noisy.shape[0]
    scale = max(abs(noisy)) if signal else noisy.max()
    data, target = noisy / scale, clean / scale
    coords = np.arange(-n / 2, n / 2)
    variance = config.gaussian_variance
    if signal:
        kernel = np.exp(-coords ** 2 / (2 * variance)) / np.sqrt(2 * np.pi * variance)
    else:
        kernel = np.exp(-(coords[:, None] ** 2 + coords[None, :] ** 2) / (2 * variance)) / (2 * np.pi * variance)
    potential = np.fft.ifftn(np.fft.fftn(np.fft.ifftshift(kernel)) * np.fft.fftn(data)).real
    if signal:
        h = np.diag(potential + 2 * config.weight)
        for i in range(n - 1):
            h[i, i + 1] = h[i + 1, i] = -config.weight
        h[0, -1] = h[-1, 0] = -config.weight
    else:
        h = literal_hamiltonian(potential, config.weight)
    _, basis = np.linalg.eigh(h)
    coefficients = np.linalg.solve(basis, data.ravel(order="F"))
    output = np.zeros(data.size)
    best_snr = 0.0
    exponents = (1, 10) if signal else (7, 11)
    widths = [2.0 ** exponents[1]] if config.slopes == 1 else 2.0 ** np.linspace(*exponents, config.slopes)
    for width in widths:
        local_snr, failures = 0.0, 0
        for threshold in range(1, data.size + 1, config.threshold_step):
            taper = literal_heavi(np.arange(1, data.size + 1) - threshold + 2, width)
            candidate = np.zeros(data.size)
            for t in range(data.size):
                candidate += basis[:, t] * taper[t] * coefficients[t]
            snr = 10 * np.log10(np.sum(target ** 2) /
                               np.sum((candidate - target.ravel(order="F")) ** 2))
            failures = (failures + 1) if local_snr > snr else 0
            local_snr = max(local_snr, snr)
            if snr > best_snr:
                best_snr, output = snr, candidate
            if not signal and failures >= config.patience:
                break
    return output.reshape(noisy.shape, order="F") * scale


def literal_dequip(image, config):
    n = config.patch_size
    rows, columns = image.shape[0] - n + 1, image.shape[1] - n + 1
    patches = np.zeros((rows, columns, n * n))
    for i in range(rows):
        for j in range(columns):
            patches[i, j] = image[i:i + n, j:j + n].ravel(order="F")
    filtered, counts = np.zeros_like(patches), np.zeros((rows, columns))
    radius = config.window_radius
    width = 2 * radius + 1
    irange = list(range(radius, rows - radius, config.stride)) + [rows - radius - 1]
    jrange = list(range(radius, columns - radius, config.stride)) + [columns - radius - 1]
    for i in irange:
        for j in jrange:
            row_indices = [(i + offset) % rows for offset in range(-radius, radius + 1)]
            col_indices = [(j + offset) % columns for offset in range(-radius, radius + 1)]
            block = np.empty((width, width, n * n))
            for col, source_col in enumerate(col_indices):
                for row, source_row in enumerate(row_indices):
                    block[row, col] = patches[source_row, source_col]
            window = block.reshape(width * width, n * n, order="F")
            means = window.mean(axis=1)
            x = window - means[:, None]
            restored = np.empty_like(x)
            for k in range(len(x)):
                interaction = np.zeros(n * n)
                for other in range(len(x)):
                    if other != k:
                        distance = (k % width - other % width) ** 2 + (k // width - other // width) ** 2
                        interaction += config.interaction * np.abs(x[k] - x[other]) / distance
                potential = x[k] + interaction
                _, basis = np.linalg.eigh(literal_hamiltonian(potential, config.weight_factor * max(potential)))
                coeff = x[k] @ basis
                coeff *= np.abs(coeff) > config.threshold
                coeff[config.retained_vectors:] = 0
                restored[k] = coeff @ basis.T + means[k]
            restored = restored.reshape(width, width, n * n, order="F")
            old_filtered, old_counts = filtered.copy(), counts.copy()
            for col, source_col in enumerate(col_indices):
                for row, source_row in enumerate(row_indices):
                    filtered[source_row, source_col] = old_filtered[source_row, source_col] + restored[row, col]
                    counts[source_row, source_col] = old_counts[source_row, source_col] + 1
    filtered /= counts[:, :, None]
    output, norm = np.zeros_like(image), np.zeros_like(image)
    for i in range(rows):
        for j in range(columns):
            output[i:i + n, j:j + n] += filtered[i, j].reshape(n, n, order="F")
            norm[i:i + n, j:j + n] += 1
    return output / norm


@pytest.mark.parametrize("side", [2, 3, 7])
def test_hamiltonian_matches_literal_boundaries(side):
    potential = np.random.default_rng(side).normal(size=(side, side))
    expected = literal_hamiltonian(potential, 2.3)
    actual = 2.3 * image_laplacian(side) + np.diag(potential.ravel(order="F"))
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    values, basis = image_basis(potential, 2.3)
    np.testing.assert_allclose(actual @ basis, basis * values, atol=1e-12)
    np.testing.assert_allclose(basis.T @ basis, np.eye(side * side), atol=1e-12)


def test_signal_periodic_boundary():
    potential = np.arange(6.0)
    values, basis = signal_basis(potential, 0.4)
    h = (basis * values) @ basis.T
    assert h[0, -1] == pytest.approx(-0.4)
    assert h[0, 0] == pytest.approx(0.8)


def test_threshold_boundary_and_patch_order():
    np.testing.assert_allclose(smooth_heaviside([-2, -1, 0, 1, 2], 1), [1, 1, 0.5, 0, 0])
    image = np.arange(30.0).reshape(5, 6)
    patches = extract_patches(image, 3)
    np.testing.assert_array_equal(patches[0, 0], [0, 6, 12, 1, 7, 13, 2, 8, 14])
    np.testing.assert_array_equal(aggregate_patches(patches), image)
    assert window_centers(7, 1, 2) == [1, 3, 5, 5]


@pytest.mark.parametrize("slopes", [1, 3])
def test_qab_image_oracle_matches_direct_reconstructions(slopes):
    rng = np.random.default_rng(7)
    clean = 20 + rng.random((8, 8)) * 10
    noisy = clean + rng.normal(size=clean.shape) * 5
    cfg = QABImageConfig(patch_size=8, stride=4, slopes=slopes, threshold_step=2, patience=3)
    np.testing.assert_allclose(qab_image(noisy, cfg, reference=clean), literal_qab(clean, noisy, cfg), atol=2e-11)


def test_qab_signal_matches_direct_reconstructions():
    clean = 5 + np.sin(np.arange(32) / 3)
    noisy = clean + np.random.default_rng(2).normal(size=32)
    cfg = QABSignalConfig(slopes=5)
    np.testing.assert_allclose(qab_signal(noisy, cfg, reference=clean),
                               literal_qab(clean, noisy, cfg, signal=True), atol=2e-11)


@pytest.mark.parametrize("stride", [1, 2, 3])
def test_dequip_matches_literal_full_pipeline(stride):
    image = np.random.default_rng(3).uniform(0, 255, (8, 9))
    cfg = DeQuIPConfig(patch_size=3, window_radius=1, stride=stride, retained_vectors=7, threshold=20)
    np.testing.assert_allclose(dequip_denoise(image, cfg), literal_dequip(image, cfg), atol=2e-10)


def test_interactions_batch_invariance():
    data = np.random.default_rng(9).normal(size=(9, 16))
    np.testing.assert_allclose(total_interactions(data, 0.1, 1), total_interactions(data, 0.1, 8))
    np.testing.assert_array_equal(total_interactions(data, 0), np.zeros_like(data))


@pytest.mark.parametrize("shape", [(5, 6), (5, 12)])
def test_dequip_wrapping_and_repeated_assignment_match_matlab_semantics(shape):
    image = np.random.default_rng(17).uniform(10, 200, shape)
    cfg = DeQuIPConfig(patch_size=3, window_radius=3, stride=3, retained_vectors=7)
    np.testing.assert_allclose(dequip_denoise(image, cfg), literal_dequip(image, cfg), atol=2e-10)


def test_constant_and_zero_inputs():
    cfg = DeQuIPConfig(patch_size=3, window_radius=1, stride=2, retained_vectors=7)
    np.testing.assert_allclose(dequip_denoise(np.full((9, 9), 80.0), cfg), 80)
    np.testing.assert_array_equal(qab_image(np.zeros((8, 8)), QABImageConfig(patch_size=8, stride=4, threshold=3)), np.zeros((8, 8)))
    np.testing.assert_array_equal(qab_signal(np.zeros(8), QABSignalConfig(threshold=3)), np.zeros(8))


def test_fixed_qab_never_uses_reference_and_covers_rectangle():
    noisy = np.random.default_rng(8).uniform(20, 100, (13, 17))
    cfg = QABImageConfig(patch_size=8, stride=4, threshold=32)
    result = qab_image(noisy, cfg)
    np.testing.assert_array_equal(result, qab_image(noisy, cfg, reference=noisy * 10))
    assert result.shape == noisy.shape and np.isfinite(result).all()


@pytest.mark.parametrize("model", ["gaussian", "heteroscedastic"])
def test_seeded_noise_exact_snr(model):
    clean = np.arange(256.0).reshape(16, 16)
    noisy = add_noise(clean, 22, 2, model)
    assert quality(clean, noisy)["snr_db"] == pytest.approx(22)
    np.testing.assert_array_equal(noisy, add_noise(clean, 22, 2, model))
    assert ssim(clean, clean) == pytest.approx(1)


@pytest.mark.parametrize("cfg", [DeQuIPConfig(stride=22), DeQuIPConfig(retained_vectors=50),
                                 DeQuIPConfig(interaction=-1), DeQuIPConfig(threshold=float("nan"))])
def test_invalid_dequip_parameters(cfg):
    with pytest.raises(ValueError):
        dequip_denoise(np.ones((32, 32)), cfg)


def test_invalid_qab_and_reference_contract():
    with pytest.raises(ValueError, match="reference"):
        qab_image(np.ones((64, 64)))
    with pytest.raises(ValueError, match="even"):
        qab_signal(np.ones(7), reference=np.ones(7))
    with pytest.raises(ValueError):
        qab_image(np.ones((8, 8)), QABImageConfig(patch_size=7, stride=2), reference=np.ones((8, 8)))
    with pytest.raises(ValueError, match="finite"):
        dequip_denoise(np.full((32, 32), np.nan))

