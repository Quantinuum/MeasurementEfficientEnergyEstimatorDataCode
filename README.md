# MeasurementEfficientEnergyEstimatorDataCode

Scripts and data for the paper [*Measurement-efficient energy estimator for quantum chemistry of photodynamic materials*](https://arxiv.org/abs/2610.26689).

## Scripts

- Top-level `make_*.py` scripts generate plots and figures for the paper.
- `fig*/` contains additional figure-specific plotting scripts.
- `ai_smcs/` contains scripts for the AI-assisted SMCS data analysis; see its [README](ai_smcs/README.md) for details.
- `compute_uncompute.py` and `classical_shadows.py` analyze hardware-shot data for the compute-uncompute and classical-shadow methods, respectively.

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
