"""Diagonalization and middle-spectrum eigenstate selection."""

from __future__ import annotations

import numpy as np


def diagonalize_hamiltonian(H: np.ndarray, check_hermitian: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Diagonalize a dense Hermitian Hamiltonian."""
    if check_hermitian and not np.allclose(H, H.conj().T, atol=1e-10):
        raise ValueError("Hamiltonian is not Hermitian within tolerance.")

    evals, evecs = np.linalg.eigh(H)
    if evals.ndim != 1 or evecs.ndim != 2 or evecs.shape[0] != evecs.shape[1]:
        raise ValueError("Unexpected shapes returned by diagonalization.")
    if evecs.shape[1] != evals.shape[0]:
        raise ValueError("Eigenvalue/eigenvector count mismatch.")

    # eigh should return normalized vectors, but keep an explicit sanity check.
    norms = np.sum(np.abs(evecs) ** 2, axis=0)
    if not np.allclose(norms, 1.0, atol=1e-10):
        raise ValueError("Diagonalization returned non-normalized eigenvectors.")
    return evals, evecs


def middle_spectrum_index(dim: int) -> int:
    """Return the selected middle-spectrum index using a fixed upper-middle rule.

    The spectrum is assumed to be sorted in ascending order. For even dimension
    `dim`, this returns `dim // 2`, i.e. the upper of the two central states.
    """
    if dim < 1:
        raise ValueError("dim must be positive.")
    return dim // 2


def select_middle_eigenstate(
    evals: np.ndarray,
    evecs: np.ndarray,
) -> tuple[int, float, np.ndarray]:
    """Select the middle eigenstate of the full sorted spectrum."""
    if evals.ndim != 1:
        raise ValueError("evals must be a 1D array.")
    if evecs.ndim != 2:
        raise ValueError("evecs must be a 2D array.")
    if evecs.shape[1] != evals.shape[0]:
        raise ValueError("evecs must have one column per eigenvalue.")

    index = middle_spectrum_index(evals.shape[0])
    state = np.asarray(evecs[:, index], dtype=np.complex128)
    norm = np.linalg.norm(state)
    if not np.isclose(norm, 1.0, atol=1e-10):
        raise ValueError(f"Selected middle eigenstate is not normalized: ||psi||={norm}.")
    return index, float(evals[index]), state

