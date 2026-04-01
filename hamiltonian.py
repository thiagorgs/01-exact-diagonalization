"""Hamiltonian construction for the full-space disordered J1-J2 isotropic XY chain.

Convention:
    H = sum_i J1 * (sigma_x(i) sigma_x(i+1) + sigma_y(i) sigma_y(i+1))
      + sum_i J2 * (sigma_x(i) sigma_x(i+2) + sigma_y(i) sigma_y(i+2))
      + sum_i h_i * sigma_z(i)

The code uses Pauli matrices sigma_x, sigma_y, sigma_z directly, not spin-1/2
operators S^a = sigma^a / 2. Open boundary conditions are used.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np


SIGMA_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
SIGMA_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
SIGMA_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
IDENTITY_2 = np.eye(2, dtype=np.complex128)


def hilbert_dimension(L: int) -> int:
    """Return the full Hilbert-space dimension 2^L."""
    if L < 1:
        raise ValueError("L must be at least 1.")
    return 1 << L


def _kron_all(operators: list[np.ndarray]) -> np.ndarray:
    result = operators[0]
    for op in operators[1:]:
        result = np.kron(result, op)
    return result


@lru_cache(maxsize=None)
def single_site_operator(L: int, site: int, axis: str) -> np.ndarray:
    """Return the full-space operator sigma_axis(site) for a chain of length L."""
    if not 0 <= site < L:
        raise IndexError(f"site={site} is outside the valid range [0, {L - 1}].")

    pauli_map = {"x": SIGMA_X, "y": SIGMA_Y, "z": SIGMA_Z}
    if axis not in pauli_map:
        raise ValueError(f"Unsupported axis {axis!r}. Choose from 'x', 'y', 'z'.")

    factors = [IDENTITY_2] * L
    factors[site] = pauli_map[axis]
    return _kron_all(factors)


def build_hamiltonian(L: int, J1: float, J2: float, h: np.ndarray) -> np.ndarray:
    """Build the dense full Hamiltonian matrix for the disordered XY chain."""
    h = np.asarray(h, dtype=float)
    if h.shape != (L,):
        raise ValueError(f"Field array must have shape ({L},), got {h.shape}.")

    dim = hilbert_dimension(L)
    H = np.zeros((dim, dim), dtype=np.complex128)

    sx_ops = [single_site_operator(L, site, "x") for site in range(L)]
    sy_ops = [single_site_operator(L, site, "y") for site in range(L)]
    sz_ops = [single_site_operator(L, site, "z") for site in range(L)]

    for i in range(L - 1):
        H += J1 * (sx_ops[i] @ sx_ops[i + 1] + sy_ops[i] @ sy_ops[i + 1])

    for i in range(L - 2):
        H += J2 * (sx_ops[i] @ sx_ops[i + 2] + sy_ops[i] @ sy_ops[i + 2])

    for i in range(L):
        H += h[i] * sz_ops[i]

    # Enforce exact Hermiticity up to machine precision.
    H = 0.5 * (H + H.conj().T)
    return H


def validate_hamiltonian(H: np.ndarray, atol: float = 1e-10) -> None:
    """Raise if the Hamiltonian is not Hermitian."""
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError(f"Hamiltonian must be square, got shape {H.shape}.")
    if not np.allclose(H, H.conj().T, atol=atol):
        raise ValueError("Hamiltonian failed the Hermiticity check.")

