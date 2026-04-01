# Exact Diagonalization for a Disordered 1D J1-J2 Isotropic XY Spin Chain

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-brightgreen)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.10%2B-green)](https://scipy.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

This is a precise exact diagonalization (ED) implementation for studying entanglement and localization properties in a 1D disordered isotropic XY spin chain with nearest-neighbor and next-nearest-neighbor couplings. The code explores the many-body localization (MBL) transition through half-chain entanglement entropy analysis.

## Physics Model

The Hamiltonian is:

```text
H = sum_i J1 * (sigma_x(i) sigma_x(i+1) + sigma_y(i) sigma_y(i+1))
  + sum_i J2 * (sigma_x(i) sigma_x(i+2) + sigma_y(i) sigma_y(i+2))
  + sum_i h_i * sigma_z(i)
```

### Parameters

- **J1**: nearest-neighbor isotropic XY coupling (default: 1.0)
- **J2**: next-nearest-neighbor isotropic XY coupling (default: 0.1)
- **h_i**: i.i.d. random fields drawn uniformly from [-W, W], where W is the disorder strength
- **L**: system size (number of spins)

### Conventions

- Open boundary conditions
- Full Hilbert space of dimension 2^L
- No restriction to fixed total magnetization sectors
- Standard Pauli matrices σ_x, σ_y, σ_z without additional factors

## Numerical Method

For each disorder realization:

1. Draw random fields h_i ~ Uniform[-W, W]
2. Build the full Hamiltonian matrix
3. Diagonalize using dense Hermitian eigendecomposition
4. Select the middle eigenstate from the full spectrum
5. Compute half-chain entanglement entropies
6. Collect total magnetization expectation value as a diagnostic

### Middle Eigenstate Convention

The selected eigenstate corresponds to index `dim // 2` in the ascending eigenvalue spectrum, where `dim = 2^L`. This consistently picks the upper of the two central eigenstates across all disorder realizations and system sizes.

## Entanglement Analysis

For even L, the chain is bipartitioned into:
- Subsystem A: first L/2 spins
- Subsystem B: remaining L/2 spins

The code computes two measures of bipartite entanglement:

- **von Neumann entropy**: S_vN = -Tr(ρ_A log ρ_A)
- **Rényi-2 entropy**: S2 = -log(Tr(ρ_A²))

Both are evaluated from the reduced density matrix ρ_A = Tr_B(|ψ⟩⟨ψ|) using natural logarithms. Numerical stability is ensured by clipping negative eigenvalues to zero.

## Project Structure

```
01-exact-diagonalization/
├── hamiltonian.py        # Full Hamiltonian construction
├── diagonalization.py    # Dense diagonalization and eigenstate selection
├── entanglement.py       # Reduced density matrix and entropy computation
├── utils.py              # RNG, helpers, magnetization, validation
├── run_scan.py           # Command-line scan driver and analysis
├── requirements.txt      # Python dependencies
├── .gitignore           # Git ignore patterns
└── README.md            # This file
```

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/thiago-girao/quantum-computing-portfolio.git
cd 01-exact-diagonalization
pip install -r requirements.txt
```

## Usage

### Run default scan

```bash
python run_scan.py
```

### Custom parameters

```bash
python run_scan.py \
  --j1 1.0 \
  --j2 0.1 \
  --lengths 8 10 12 \
  --disorder 0.0 0.5 1.0 1.5 2.0 2.5 3.0 4.0 \
  --n-real 50 \
  --seed 12345 \
  --output-dir outputs/xy_j1j2_scan
```

### Quick validation

```bash
python run_scan.py --validate-only
```

### Fast test run

```bash
python run_scan.py --lengths 8 --disorder 0.0 1.0 2.0 --n-real 4 --seed 7
```

### Command-line options

- `--j1`: Nearest-neighbor coupling (default: 1.0)
- `--j2`: Next-nearest-neighbor coupling (default: 0.1)
- `--lengths`: System sizes to scan (default: 8 10 12)
- `--disorder`: Disorder strengths W to scan (default: 0.0 0.5 1.0 1.5 2.0 2.5 3.0 4.0)
- `--n-real`: Disorder realizations per parameter set (default: 50)
- `--seed`: Random seed for reproducibility
- `--output-dir`: Output directory (default: outputs/)
- `--validate-only`: Run only validation checks

## Output

For each run, the output directory contains:

- `raw_results.csv`: One row per disorder realization with all computed quantities
- `raw_results.npz`: Raw arrays for post-processing
- `summary_results.csv`: Disorder-averaged entropies and magnetization diagnostics
- `summary_results.npz`: Summary arrays
- `svon_vs_W.png`: Average von Neumann entropy vs disorder strength W
- `s2_vs_W.png`: Average Rényi-2 entropy vs disorder strength W
- `entropy_compare_L*.png`: Per-system-size comparison plots
- `run_metadata.json`: Complete metadata and parameter snapshot

## Results

The code produces high-quality plots visualizing:

- **Entanglement entropy evolution**: How entanglement changes with increasing disorder
- **System size scaling**: Entanglement behavior across different chain lengths
- **MBL signature**: Entropy saturation at strong disorder indicating localization

These plots are saved as PNG files in the output directory and provide key insights into the many-body localization transition.

## Validation

The code includes comprehensive validation checks:

- Hamiltonian Hermiticity
- Zero entanglement for product states
- Non-negative entanglement entropies
- Consistency between entropy calculation methods
- Eigenstate normalization
- Correct magnetization expectation values on basis states

All checks are run automatically and reported to the user.

## Limitations and Design Choices

- **Dense diagonalization**: Full Hilbert space computation suitable for L ≤ 12
- **No symmetry reduction**: Code prioritizes clarity over performance
- **Single eigenstate**: Focuses on the middle eigenstate; extension to energy windows is straightforward
- **No time evolution**: Current version for static properties; dynamics not included

Default sizes (L = 8, 10, 12) balance computational cost with sufficient physics range to observe disorder effects and potential MBL signatures.

## References

- Alet, F., & Laflorencie, N. (2018). Many-body localization: Strong disorder physics meets strong correlations. *C. R. Physique*, 19(6), 498-525.
- Basko, D. M., Aleiner, I. L., & Altshuler, B. L. (2006). Metal-insulator transition in a weakly interacting many-electron system with random potential. *Annals of Physics*, 321(5), 1126-1205.
- Oganesyan, V., & Huse, D. A. (2007). Localization of interacting fermions at high temperature. *Physical Review B*, 75(15), 155111.
- Serbyn, M., Papić, Z., & Abanin, D. A. (2013). Local conservation laws and the structure of the many-body localized states. *Physical Review Letters*, 111(12), 127201.

## Author

**Thiago Girao** - PhD candidate in Physics, researching quantum information and quantum computing.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

This project is part of PhD research in quantum computing and many-body physics.
