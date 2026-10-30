# Measurement Efficient Energy Estimator (10,10) hardware circuits

Circuit-generation scripts for the AI-selected (10,10) sMCS states used in
Table 1 and Supplementary Table S8 of the Measurement Efficient Energy Estimator paper.

The returned hardware shots and energy-analysis script are in
[hardware_analysis/](hardware_analysis/).

| Script | Method |
| --- | --- |
| `no_parity.py` | Leakage-only, without a parity check |
| `cpc.py` | Coherent parity check (CPC) |
| `mbpc.py` | Measurement-based parity check (MBPC) |

Each script builds the two compute--uncompute circuits, `s` and `h`, at both
geometries. The `s` circuit measures the squared overlap with the reference
state; the `h` circuit measures the squared overlap with the residual state.
Both use the same target-state preparation at a given geometry. CPC and MBPC
insert the parity checks between preparation and uncomputation. All methods
use ternary leakage measurements on the data qubits.

## Inputs

| Geometry | `(m,k,l)` | Target support |
| --- | --- | --- |
| Rx | `(1,21,24)` | Largest-amplitude determinants |
| R0 | `(1,21,22)` | Residual-support guarding, without determinant addback |

Here `m`, `k` and `l` are the reference-, residual- and target-state determinant
counts. `data/` contains three preparation circuits for each geometry in pytket
JSON format: `psil` (target), `phi` (reference) and `chik` (residual). These same
inputs are shared by all three methods. The selected spin-orbital mappings are
already included; do not apply another permutation.

`metadata/states.json` gives the mappings, reference quantities and ideal
overlap probabilities. The three method JSONs in `metadata/` give the circuit
settings, hardware shot allocations and qubit counts. The hardware allocation
was 2,000 shots for `s` and 8,000 for `h`, per geometry and method.

MBPC allocates 50 qubits across the program. The two spin-sector checks are
called sequentially, and the hardware routing configuration used 35 qubits.
Both counts are recorded in `metadata/mbpc.json`.

## Build the circuits

Use Python 3.12:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python no_parity.py
python cpc.py
python mbpc.py
```

The scripts write HUGR program files and a compilation report under
`compiled/<method>/`. They compile locally; they do not simulate the circuits
or submit jobs. No InQuanto license or QNexus login is needed for compilation
from the supplied inputs.

To build one circuit:

```sh
python cpc.py --geometry R0 --overlap h --out compiled/r0_cpc_h
```

Existing outputs are not overwritten. Use a new `--out` directory to compile
again. `late_init.py` implements late-reset preparation; `common.py` handles
input loading and output files.

The input checksums in `metadata/input_checksums.json` are checked before
compilation to detect changes to the supplied files. They identify the input
files, not the scientific accuracy of a result.

This release reproduces the final circuit construction from the supplied
state-preparation inputs, not the AI-assisted search. The circuit graphs were
checked against the constructors used for the reported hardware runs.
Byte-identical reproduction of the archived HUGR packages is not claimed.
