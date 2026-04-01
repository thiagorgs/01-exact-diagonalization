"""Half-chain entanglement utilities for pure states."""

from __future__ import annotations

import numpy as np


def _validate_state_length(state: np.ndarray, L: int) -> np.ndarray:
    state = np.asarray(state, dtype=np.complex128).reshape(-1)
    if state.size != (1 << L):
        raise ValueError(f"State size {state.size} does not match Hilbert dimension 2^{L}.")
    norm = np.linalg.norm(state)
    if not np.isclose(norm, 1.0, atol=1e-10):
        raise ValueError(f"State is not normalized: ||psi||={norm}.")
    return state


def reduced_density_matrix_half_chain(state: np.ndarray, L: int) -> np.ndarray:
    """Return rho_A for A = first L/2 spins and B = remaining L/2 spins."""
    if L % 2 != 0:
        raise ValueError("Half-chain bipartition requires even L.")

    state = _validate_state_length(state, L)
    LA = L // 2
    LB = L - LA
    psi_matrix = state.reshape((1 << LA), (1 << LB))
    rho_a = psi_matrix @ psi_matrix.conj().T
    rho_a = 0.5 * (rho_a + rho_a.conj().T)
    return rho_a


def density_matrix_spectrum(rho: np.ndarray, clip_tol: float = 1e-12) -> np.ndarray:
    """Return a safely clipped eigenvalue spectrum of a density matrix."""
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError("rho must be a square matrix.")
    evals = np.linalg.eigvalsh(rho)
    evals = np.where(np.abs(evals) < clip_tol, 0.0, evals)
    if np.min(evals) < -clip_tol:
        raise ValueError(f"rho has eigenvalues below tolerance: min={np.min(evals)}.")
    evals = np.clip(evals.real, 0.0, None)
    trace = np.sum(evals)
    if not np.isclose(trace, 1.0, atol=1e-10):
        raise ValueError(f"Reduced density matrix trace is not 1: Tr(rho)={trace}.")
    return evals


def von_neumann_entropy_from_rho(rho: np.ndarray, clip_tol: float = 1e-12) -> float:
    """Compute S_vN = -Tr(rho log rho) with natural logarithms."""
    evals = density_matrix_spectrum(rho, clip_tol=clip_tol)
    positive = evals[evals > 0.0]
    if positive.size == 0:
        return 0.0
    entropy = -np.sum(positive * np.log(positive))
    return float(max(entropy, 0.0))


def renyi2_entropy_from_rho(rho: np.ndarray, clip_tol: float = 1e-12) -> float:
    """Compute S2 = -log(Tr(rho^2)) with natural logarithms."""
    evals = density_matrix_spectrum(rho, clip_tol=clip_tol)
    purity = float(np.sum(evals**2))
    if purity <= 0.0 or purity > 1.0 + 1e-10:
        raise ValueError(f"Unphysical purity encountered: Tr(rho^2)={purity}.")
    entropy = -np.log(purity)
    return float(max(entropy, 0.0))


def half_chain_entropies(state: np.ndarray, L: int, clip_tol: float = 1e-12) -> tuple[np.ndarray, float, float]:
    """Compute rho_A, von Neumann entropy, and Renyi-2 entropy from the same rho_A."""
    rho_a = reduced_density_matrix_half_chain(state, L)
    s_vn = von_neumann_entropy_from_rho(rho_a, clip_tol=clip_tol)
    s2 = renyi2_entropy_from_rho(rho_a, clip_tol=clip_tol)
    if s_vn < -1e-12 or s2 < -1e-12:
        raise ValueError("Entropies must be nonnegative within tolerance.")
    return rho_a, s_vn, s2

