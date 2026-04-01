"""Shared helpers for RNG, observables, statistics, and validation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np


def make_rng(seed: int | None) -> np.random.Generator:
    """Create a reproducible NumPy random number generator."""
    return np.random.default_rng(seed)


def generate_random_fields(L: int, W: float, rng: np.random.Generator) -> np.ndarray:
    """Draw h_i ~ Uniform[-W, W]."""
    if W < 0.0:
        raise ValueError("W must be nonnegative.")
    return rng.uniform(-W, W, size=L)


def ensure_directory(path: str | Path) -> Path:
    """Create and return a directory path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def mean_std_sem(values: np.ndarray) -> tuple[float, float, float]:
    """Return mean, sample standard deviation, and standard error."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("values must be a non-empty 1D array.")
    mean = float(np.mean(values))
    if values.size == 1:
        return mean, 0.0, 0.0
    std = float(np.std(values, ddof=1))
    sem = float(std / np.sqrt(values.size))
    return mean, std, sem


def write_csv(path: str | Path, fieldnames: list[str], rows: Iterable[dict]) -> None:
    """Write a CSV file from an iterable of dictionaries."""
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: str | Path, payload: dict) -> None:
    """Write JSON with stable formatting."""
    path = Path(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def basis_state(index: int, L: int) -> np.ndarray:
    """Return a computational-basis product state |index> in the full Hilbert space."""
    dim = 1 << L
    if not 0 <= index < dim:
        raise ValueError(f"index must be in [0, {dim - 1}].")
    state = np.zeros(dim, dtype=np.complex128)
    state[index] = 1.0
    return state


def total_magnetization_operator_diagonal(L: int) -> np.ndarray:
    """Return the diagonal of sum_i sigma_z(i) in the computational basis.

    Basis convention:
        bit 0 at a site corresponds to sigma_z eigenvalue +1
        bit 1 at a site corresponds to sigma_z eigenvalue -1
    """
    dim = 1 << L
    diag = np.empty(dim, dtype=float)
    for basis_index in range(dim):
        total = 0
        for site in range(L):
            bit = (basis_index >> (L - 1 - site)) & 1
            total += 1 if bit == 0 else -1
        diag[basis_index] = total
    return diag


def total_magnetization_expectation(state: np.ndarray, L: int) -> float:
    """Compute <psi| sum_i sigma_z(i) |psi>."""
    state = np.asarray(state, dtype=np.complex128).reshape(-1)
    if state.size != (1 << L):
        raise ValueError(f"State size {state.size} does not match Hilbert dimension 2^{L}.")
    diag = total_magnetization_operator_diagonal(L)
    probabilities = np.abs(state) ** 2
    return float(np.dot(probabilities, diag))

