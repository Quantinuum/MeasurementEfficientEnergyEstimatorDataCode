# MeasurementEfficientEnergyEstimatorDataCode

Scripts and data for the paper [*Measurement-efficient energy estimator for quantum chemistry of photodynamic materials*](https://arxiv.org/abs/2610.26689).

## Scripts

- Top-level `make_*.py` scripts generate plots and figures for the paper.
- `fig*/` contains additional figure-specific plotting scripts.
- `ai_smcs/` contains scripts for the AI-assisted SMCS data analysis; see its [README](ai_smcs/README.md) for details.
- `compute_uncompute.py` and `classical_shadows.py` analyze hardware-shot data for the compute-uncompute and classical-shadow methods, respectively.
- `guppy_overlaps_protocol.py` builds overlap circuits for different protocols.
- `qsys_result_sampling.py` supports processing Qsys results, including leakage filtering, bootstrap resampling, and JSON input/output.
- `selene_backend.py` provides a local emulator function that takes a program and can run (noisy) shots.

## Shared utilities

- `utils.py` contains shared utility functions for processing circuits and energies.
- `inquanto_utils.py` converts InQuanto objects into forms that can be used by the post-processing workflows.

## Data

- `data/circuit_resources/` contains compiled, optimized and transpiled resource estimates for state-preparation circuits at different truncations.
- `data/energies/` contains CASCI, CS (classical shadows), CU (compute-uncompute), and SV (state-vector) energies used in the paper figures.
- `data/hamiltonians/` contains the Hamiltonians used in the paper.
- `data/<geo><cas>/` contains circuits and shot tables for the hardware runs, grouped by geometry and active-space size (for example, `rx66` and `t11010`).
- `data/pauli_averaging.json` records the number of shots estimated to reach chemical accuracy with Pauli averaging (PA).
- `data/tups_params.json` contains the optimized tUPS parameters from VQE.

Some files make reference to the state "chi". This has been renamed to "lambda" in the manuscript.

## Load a Hamiltonian into Zixy

The six `.npz` files in `data/hamiltonians/` contain NumPy integral arrays and
Pauli expansions. `load_hamiltonians.py` reads only `constant`, `one_body`, and
`two_body` from an `.npz` file and constructs a normal-ordered Zixy fermion
`RealTermSum` (Zixy >= 0.4.0), without InQuanto. Spatial integrals use chemist's
notation, with the two-body factor of 1/2 and interleaved up/down spin orbitals:

```python
from load_hamiltonians import load_hamiltonian

hamiltonian = load_hamiltonian("data/hamiltonians/ham_Rx_geom_triplet_spin_shciscf_basis_66_6e_6o.npz")
```
