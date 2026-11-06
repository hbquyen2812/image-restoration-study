"""Shared operators, using MATLAB column-major pixel order throughout."""

from functools import lru_cache
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import eigh

FloatArray = NDArray[np.float64]


def as_float(data: ArrayLike, ndim: int) -> FloatArray:
    result = np.asarray(data, dtype=np.float64)
    if result.ndim != ndim or result.size == 0 or not np.isfinite(result).all():
        raise ValueError(f"Expected a nonempty, finite {ndim}D array.")
    return result


def positive_int(name: str, value: int, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")


def nonnegative(name: str, value: float, *, strict: bool = False) -> None:
    if not np.isfinite(value) or value < 0 or (strict and value == 0):
        raise ValueError(f"{name} must be finite and {'positive' if strict else 'nonnegative'}.")


@lru_cache(maxsize=8)
def image_laplacian(side: int) -> FloatArray:
    """Graph Laplacian: 4 interior, 3 edge, 2 corner; no wraparound edges.

    Matches f_ondes2D.m and f_Hamil2D.m, including their boundary correction.
    """
    positive_int("side", side, 2)
    one = np.diag(np.r_[1.0, np.full(side - 2, 2.0), 1.0])
    one -= np.diag(np.ones(side - 1), 1) + np.diag(np.ones(side - 1), -1)
    laplacian = np.kron(np.eye(side), one) + np.kron(one, np.eye(side))
    laplacian.flags.writeable = False
    return laplacian


def image_basis(potential: FloatArray, weight: float) -> tuple[FloatArray, FloatArray]:
    """Return increasing eigenvalues and orthonormal eigenvector columns."""
    size = potential.size
    side = int(np.sqrt(size))
    if side * side != size:
        raise ValueError("Potential must contain a square number of pixels.")
    hamiltonian = np.array(weight * image_laplacian(side), order="F")
    hamiltonian.flat[::size + 1] += potential.ravel(order="F")
    return eigh(hamiltonian, driver="evd", overwrite_a=True, check_finite=False)


def signal_basis(potential: FloatArray, weight: float) -> tuple[FloatArray, FloatArray]:
    """Periodic 1D Hamiltonian from f_ondes1D.m."""
    n = potential.size
    hamiltonian = np.diag(potential + 2 * weight)
    hamiltonian += np.diag(np.full(n - 1, -weight), 1)
    hamiltonian += np.diag(np.full(n - 1, -weight), -1)
    hamiltonian[0, -1] = hamiltonian[-1, 0] = -weight
    return eigh(hamiltonian, driver="evd", overwrite_a=True, check_finite=False)


def smooth_heaviside(x: ArrayLike, width: float) -> FloatArray:
    """The descending, compact, smooth threshold in upstream heavi.m."""
    x = np.asarray(x, dtype=np.float64)
    ratio = np.clip(x / width, -1, 1)
    middle = 0.5 * (1 - ratio - np.sin(np.pi * ratio) / np.pi)
    return np.where(x <= -width, 1.0, np.where(x >= width, 0.0, middle))


def patch_starts(length: int, side: int, stride: int) -> list[int]:
    """Cover the last edge too; identical to upstream on its supported sizes."""
    starts = list(range(0, length - side + 1, stride))
    if starts[-1] != length - side:
        starts.append(length - side)
    return starts

