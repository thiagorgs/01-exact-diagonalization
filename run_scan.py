"""Run disorder-averaged scans for the disordered J1-J2 isotropic XY chain."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from diagonalization import diagonalize_hamiltonian, select_middle_eigenstate
from entanglement import half_chain_entropies
from hamiltonian import build_hamiltonian, validate_hamiltonian
from utils import (
    basis_state,
    ensure_directory,
    generate_random_fields,
    make_rng,
    mean_std_sem,
    total_magnetization_expectation,
    write_csv,
    write_json,
)


DEFAULT_J1 = 1.0
DEFAULT_J2 = 0.1
DEFAULT_LENGTHS = [8, 10, 12]
DEFAULT_DISORDER = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
DEFAULT_N_REAL = 50
DEFAULT_OUTPUT_DIR = Path("outputs") / "xy_j1j2_ed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Exact diagonalization scan for a 1D disordered J1-J2 isotropic XY chain "
            "with open boundaries in the full Hilbert space."
        )
    )
    parser.add_argument("--j1", type=float, default=DEFAULT_J1, help="Nearest-neighbor XY coupling.")
    parser.add_argument("--j2", type=float, default=DEFAULT_J2, help="Next-nearest-neighbor XY coupling.")
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=DEFAULT_LENGTHS,
        help="Even system sizes L to scan.",
    )
    parser.add_argument(
        "--disorder",
        type=float,
        nargs="+",
        default=DEFAULT_DISORDER,
        help="Disorder strengths W for h_i ~ Uniform[-W, W].",
    )
    parser.add_argument("--n-real", type=int, default=DEFAULT_N_REAL, help="Number of disorder realizations.")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where raw data, summaries, and plots will be written.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip the lightweight validation checks before the scan.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run validation checks and exit without performing the scan.",
    )
    return parser.parse_args()


def validate_inputs(lengths: list[int], disorder: list[float], n_real: int) -> None:
    if any(L <= 0 or L % 2 != 0 for L in lengths):
        raise ValueError("All system sizes must be positive even integers.")
    if any(W < 0.0 for W in disorder):
        raise ValueError("All disorder strengths W must be nonnegative.")
    if n_real <= 0:
        raise ValueError("n_real must be positive.")


def run_validation_checks() -> None:
    """Run simple internal validation checks requested for the first version."""
    # Product state entanglement should be zero.
    product_state = basis_state(0, 4)
    rho_a, s_vn, s2 = half_chain_entropies(product_state, 4)
    if not np.isclose(np.trace(rho_a), 1.0, atol=1e-10):
        raise ValueError("Validation failed: reduced density matrix trace is not 1.")
    if not np.isclose(s_vn, 0.0, atol=1e-12):
        raise ValueError(f"Validation failed: product-state S_vN should be 0, got {s_vn}.")
    if not np.isclose(s2, 0.0, atol=1e-12):
        raise ValueError(f"Validation failed: product-state S2 should be 0, got {s2}.")

    # Total magnetization on basis states.
    if not np.isclose(total_magnetization_expectation(basis_state(0, 4), 4), 4.0):
        raise ValueError("Validation failed: |0000> should have total sigma_z = +4.")
    if not np.isclose(total_magnetization_expectation(basis_state((1 << 4) - 1, 4), 4), -4.0):
        raise ValueError("Validation failed: |1111> should have total sigma_z = -4.")

    # Hamiltonian Hermiticity and middle-state normalization.
    test_fields = np.array([0.2, -0.1, 0.3, -0.4], dtype=float)
    H = build_hamiltonian(L=4, J1=1.0, J2=0.1, h=test_fields)
    validate_hamiltonian(H)
    evals, evecs = diagonalize_hamiltonian(H)
    _, _, state = select_middle_eigenstate(evals, evecs)
    if not np.isclose(np.linalg.norm(state), 1.0, atol=1e-10):
        raise ValueError("Validation failed: selected middle state is not normalized.")


def summarize_group(raw_rows: list[dict]) -> list[dict]:
    """Aggregate raw realization-level results by (L, W)."""
    grouped: dict[tuple[int, float], list[dict]] = {}
    for row in raw_rows:
        key = (int(row["L"]), float(row["W"]))
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict] = []
    for (L, W) in sorted(grouped):
        rows = grouped[(L, W)]
        s_vn_values = np.array([row["S_vN"] for row in rows], dtype=float)
        s2_values = np.array([row["S2"] for row in rows], dtype=float)
        mz_values = np.array([row["M_total"] for row in rows], dtype=float)

        mean_s_vn, std_s_vn, sem_s_vn = mean_std_sem(s_vn_values)
        mean_s2, std_s2, sem_s2 = mean_std_sem(s2_values)
        mean_mz, std_mz, sem_mz = mean_std_sem(mz_values)

        summary_rows.append(
            {
                "L": L,
                "W": W,
                "n_real": len(rows),
                "mean_S_vN": mean_s_vn,
                "std_S_vN": std_s_vn,
                "sem_S_vN": sem_s_vn,
                "mean_S2": mean_s2,
                "std_S2": std_s2,
                "sem_S2": sem_s2,
                "mean_M_total": mean_mz,
                "std_M_total": std_mz,
                "sem_M_total": sem_mz,
            }
        )
    return summary_rows


def save_raw_results(raw_rows: list[dict], output_dir: Path) -> None:
    fieldnames = ["L", "W", "realization", "middle_index", "energy", "S_vN", "S2", "M_total"]
    write_csv(output_dir / "raw_results.csv", fieldnames, raw_rows)

    np.savez_compressed(
        output_dir / "raw_results.npz",
        L=np.array([row["L"] for row in raw_rows], dtype=int),
        W=np.array([row["W"] for row in raw_rows], dtype=float),
        realization=np.array([row["realization"] for row in raw_rows], dtype=int),
        middle_index=np.array([row["middle_index"] for row in raw_rows], dtype=int),
        energy=np.array([row["energy"] for row in raw_rows], dtype=float),
        S_vN=np.array([row["S_vN"] for row in raw_rows], dtype=float),
        S2=np.array([row["S2"] for row in raw_rows], dtype=float),
        M_total=np.array([row["M_total"] for row in raw_rows], dtype=float),
    )


def save_summary_results(summary_rows: list[dict], output_dir: Path) -> None:
    fieldnames = [
        "L",
        "W",
        "n_real",
        "mean_S_vN",
        "std_S_vN",
        "sem_S_vN",
        "mean_S2",
        "std_S2",
        "sem_S2",
        "mean_M_total",
        "std_M_total",
        "sem_M_total",
    ]
    write_csv(output_dir / "summary_results.csv", fieldnames, summary_rows)

    np.savez_compressed(
        output_dir / "summary_results.npz",
        L=np.array([row["L"] for row in summary_rows], dtype=int),
        W=np.array([row["W"] for row in summary_rows], dtype=float),
        n_real=np.array([row["n_real"] for row in summary_rows], dtype=int),
        mean_S_vN=np.array([row["mean_S_vN"] for row in summary_rows], dtype=float),
        std_S_vN=np.array([row["std_S_vN"] for row in summary_rows], dtype=float),
        sem_S_vN=np.array([row["sem_S_vN"] for row in summary_rows], dtype=float),
        mean_S2=np.array([row["mean_S2"] for row in summary_rows], dtype=float),
        std_S2=np.array([row["std_S2"] for row in summary_rows], dtype=float),
        sem_S2=np.array([row["sem_S2"] for row in summary_rows], dtype=float),
        mean_M_total=np.array([row["mean_M_total"] for row in summary_rows], dtype=float),
        std_M_total=np.array([row["std_M_total"] for row in summary_rows], dtype=float),
        sem_M_total=np.array([row["sem_M_total"] for row in summary_rows], dtype=float),
    )


def _group_summary_by_length(summary_rows: list[dict]) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = {}
    for row in summary_rows:
        grouped.setdefault(int(row["L"]), []).append(row)
    for L in grouped:
        grouped[L] = sorted(grouped[L], key=lambda row: float(row["W"]))
    return grouped


def plot_summary(summary_rows: list[dict], output_dir: Path) -> None:
    grouped = _group_summary_by_length(summary_rows)

    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    for L, rows in grouped.items():
        w = np.array([row["W"] for row in rows], dtype=float)
        mean = np.array([row["mean_S_vN"] for row in rows], dtype=float)
        sem = np.array([row["sem_S_vN"] for row in rows], dtype=float)
        ax.errorbar(w, mean, yerr=sem, marker="o", linewidth=1.8, capsize=3, label=f"L={L}")
    ax.set_xlabel("Disorder strength W")
    ax.set_ylabel("Average half-chain von Neumann entropy")
    ax.set_title("Middle-spectrum entanglement entropy vs disorder")
    ax.legend()
    fig.savefig(output_dir / "svon_vs_W.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    for L, rows in grouped.items():
        w = np.array([row["W"] for row in rows], dtype=float)
        mean = np.array([row["mean_S2"] for row in rows], dtype=float)
        sem = np.array([row["sem_S2"] for row in rows], dtype=float)
        ax.errorbar(w, mean, yerr=sem, marker="s", linewidth=1.8, capsize=3, label=f"L={L}")
    ax.set_xlabel("Disorder strength W")
    ax.set_ylabel("Average half-chain Renyi-2 entropy")
    ax.set_title("Middle-spectrum Renyi-2 entropy vs disorder")
    ax.legend()
    fig.savefig(output_dir / "s2_vs_W.png", dpi=200)
    plt.close(fig)

    for L, rows in grouped.items():
        w = np.array([row["W"] for row in rows], dtype=float)
        mean_s_vn = np.array([row["mean_S_vN"] for row in rows], dtype=float)
        sem_s_vn = np.array([row["sem_S_vN"] for row in rows], dtype=float)
        mean_s2 = np.array([row["mean_S2"] for row in rows], dtype=float)
        sem_s2 = np.array([row["sem_S2"] for row in rows], dtype=float)

        fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
        ax.errorbar(w, mean_s_vn, yerr=sem_s_vn, marker="o", linewidth=1.8, capsize=3, label="S_vN")
        ax.errorbar(w, mean_s2, yerr=sem_s2, marker="s", linewidth=1.8, capsize=3, label="S2")
        ax.set_xlabel("Disorder strength W")
        ax.set_ylabel("Entropy")
        ax.set_title(f"Entropy comparison for L={L}")
        ax.legend()
        fig.savefig(output_dir / f"entropy_compare_L{L}.png", dpi=200)
        plt.close(fig)


def run_scan(
    j1: float,
    j2: float,
    lengths: list[int],
    disorder: list[float],
    n_real: int,
    seed: int | None,
) -> list[dict]:
    rng = make_rng(seed)
    raw_rows: list[dict] = []

    for L in lengths:
        for W in disorder:
            for realization in range(n_real):
                h = generate_random_fields(L, W, rng)
                H = build_hamiltonian(L=L, J1=j1, J2=j2, h=h)
                validate_hamiltonian(H)

                evals, evecs = diagonalize_hamiltonian(H)
                middle_index, energy, state = select_middle_eigenstate(evals, evecs)
                _, s_vn, s2 = half_chain_entropies(state, L)
                m_total = total_magnetization_expectation(state, L)

                raw_rows.append(
                    {
                        "L": int(L),
                        "W": float(W),
                        "realization": int(realization),
                        "middle_index": int(middle_index),
                        "energy": float(energy),
                        "S_vN": float(s_vn),
                        "S2": float(s2),
                        "M_total": float(m_total),
                    }
                )
    return raw_rows


def main() -> None:
    args = parse_args()
    lengths = list(args.lengths)
    disorder = list(args.disorder)
    validate_inputs(lengths, disorder, args.n_real)

    if not args.skip_validation or args.validate_only:
        run_validation_checks()
    if args.validate_only:
        print("Validation checks passed.")
        return

    output_dir = ensure_directory(args.output_dir)
    raw_rows = run_scan(
        j1=args.j1,
        j2=args.j2,
        lengths=lengths,
        disorder=disorder,
        n_real=args.n_real,
        seed=args.seed,
    )
    summary_rows = summarize_group(raw_rows)

    save_raw_results(raw_rows, output_dir)
    save_summary_results(summary_rows, output_dir)
    plot_summary(summary_rows, output_dir)

    write_json(
        output_dir / "run_metadata.json",
        {
            "model": "1D disordered J1-J2 isotropic XY chain",
            "boundary_conditions": "open",
            "hilbert_space": "full",
            "middle_eigenstate_rule": "ascending full spectrum, zero-based index dim // 2",
            "entropy_bipartition": "first L/2 spins versus remaining L/2 spins",
            "pauli_convention": "sigma_x, sigma_y, sigma_z used directly",
            "parameters": {
                "J1": args.j1,
                "J2": args.j2,
                "lengths": lengths,
                "disorder": disorder,
                "n_real": args.n_real,
                "seed": args.seed,
            },
        },
    )

    print(f"Finished scan. Results written to: {output_dir}")


if __name__ == "__main__":
    main()
