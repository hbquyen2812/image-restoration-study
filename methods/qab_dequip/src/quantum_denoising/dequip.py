"""De-QuIP: interacting centered patches, adaptive basis, hard thresholding.

Equations and defaults follow Sayantan Dutta's De-QuIP-Denoising MATLAB source.
The final duplicated window center in the original is deliberately preserved.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import ArrayLike

from .core import FloatArray, as_float, image_basis, nonnegative, positive_int


@dataclass(frozen=True)
class DeQuIPConfig:
    patch_size: int = 7
    window_radius: int = 10
    stride: int = 10
    interaction: float = 0.1
    retained_vectors: int = 41
    weight_factor: float = 2.5
    threshold: float = 25.0
    interaction_batch: int = 32

    def validate(self, shape: tuple[int, int]) -> None:
        positive_int("patch_size", self.patch_size, 2)
        positive_int("window_radius", self.window_radius, 0)
        positive_int("stride", self.stride)
        positive_int("retained_vectors", self.retained_vectors)
        positive_int("interaction_batch", self.interaction_batch)
        for key in ("interaction", "weight_factor", "threshold"):
            nonnegative(key, getattr(self, key))
        if self.retained_vectors > self.patch_size ** 2:
            raise ValueError("retained_vectors must be <= patch_size squared.")
        if self.stride > 2 * self.window_radius + 1:
            raise ValueError("stride leaves uncovered patch locations.")
        if min(shape) < self.patch_size:
            raise ValueError("Image is too small for the configured patch.")


def extract_patches(image: FloatArray, side: int) -> FloatArray:
    """(patch row, patch column, pixels in MATLAB/F order)."""
    windows = sliding_window_view(image, (side, side))
    return windows.swapaxes(-1, -2).reshape(*windows.shape[:2], side * side).copy()


def aggregate_patches(patches: FloatArray) -> FloatArray:
    rows, columns, pixels = patches.shape
    side = int(np.sqrt(pixels))
    output = np.zeros((rows + side - 1, columns + side - 1))
    counts = np.zeros_like(output)
    for column in range(side):
        for row in range(side):
            region = np.s_[row:row + rows, column:column + columns]
            output[region] += patches[:, :, row + column * side]
            counts[region] += 1
    return output / counts


def window_centers(length: int, radius: int, stride: int) -> list[int]:
    # MATLAB [(1+hW):delta:(M-hW) M-hW], including a duplicate endpoint.
    return list(range(radius, length - radius, stride)) + [length - radius - 1]


@lru_cache(maxsize=4)
def _inverse_distances(side: int) -> FloatArray:
    indices = np.arange(side * side)
    rows, columns = indices % side, indices // side
    distances = (rows[:, None] - rows) ** 2 + (columns[:, None] - columns) ** 2
    inverse = np.zeros_like(distances, dtype=np.float64)
    np.divide(1, distances, out=inverse, where=distances != 0)
    inverse.flags.writeable = False
    return inverse


def total_interactions(centered: FloatArray, strength: float, batch: int = 32) -> FloatArray:
    """Sum p*abs(x[k]-x[n])/distance(k,n)^2, excluding self-interaction.

    Batching bounds the temporary to (batch, window_patches, patch_pixels).
    """
    count = centered.shape[0]
    weights = _inverse_distances(int(np.sqrt(count)))
    result = np.zeros_like(centered)
    if strength == 0:
        return result
    for start in range(0, count, batch):
        stop = min(start + batch, count)
        differences = np.abs(centered[start:stop, None, :] - centered[None, :, :])
        result[start:stop] = strength * np.einsum("knp,kn->kp", differences, weights[start:stop])
    return result


def denoise_window(patches: FloatArray, config: DeQuIPConfig) -> FloatArray:
    means = patches.mean(axis=1, keepdims=True)
    centered = patches - means
    potential = centered + total_interactions(centered, config.interaction, config.interaction_batch)
    result = np.empty_like(patches)
    for k in range(len(patches)):
        weight = config.weight_factor * float(potential[k].max())
        _, basis = image_basis(potential[k], weight)
        coefficients = centered[k] @ basis
        # Strict inequality matches hardthresholding.m; del is unused there.
        coefficients[np.abs(coefficients) <= config.threshold] = 0
        coefficients[config.retained_vectors:] = 0
        result[k] = basis @ coefficients + means[k, 0]
    return result


def dequip_denoise(noisy: ArrayLike, config: DeQuIPConfig = DeQuIPConfig(), *,
                   progress: Callable[[int, int], None] | None = None) -> FloatArray:
    """Denoise grayscale data in its original units; defaults expect 0..255 units.

    No clean reference, training, clipping, or automatic rescaling is used.
    """
    image = as_float(noisy, 2)
    config.validate(image.shape)
    patches = extract_patches(image, config.patch_size)
    rows, columns, pixels = patches.shape
    filtered, counts = np.zeros_like(patches), np.zeros((rows, columns))
    radius = config.window_radius
    width = 2 * radius + 1
    positions = [(i, j) for i in window_centers(rows, radius, config.stride)
                 for j in window_centers(columns, radius, config.stride)]
    for done, (i, j) in enumerate(positions, 1):
        if rows >= width and columns >= width:
            region = np.s_[i - radius:i + radius + 1, j - radius:j + radius + 1]
        else:
            # Upstream modulo indexing also permits windows larger than the
            # patch grid (e.g. a 24x24 image with WP=7, hW=10).
            row_indices = (i + np.arange(-radius, radius + 1)) % rows
            column_indices = (j + np.arange(-radius, radius + 1)) % columns
            region = np.ix_(row_indices, column_indices)
        window = patches[region].reshape(width * width, pixels, order="F")
        restored = denoise_window(window, config).reshape(width, width, pixels, order="F")
        # Advanced assignment intentionally keeps the last occurrence when
        # modulo indices repeat, matching MATLAB assignment, NOT np.add.at.
        filtered[region] += restored
        counts[region] += 1
        if progress is not None:
            progress(done, len(positions))
    if not np.all(counts):
        raise RuntimeError("Internal error: uncovered patch locations.")
    return aggregate_patches(filtered / counts[:, :, None])

