# Noor (10,10) hardware analysis

Raw Helios-1 measurements and postprocessing for the AI-selected (10,10) sMCS
states: leakage-only, coherent parity check (CPC), and full measurement-based
parity check (MBPC). The twelve files contain 60,000 returned shots, including
rejected records.

## Run

Tested with Python 3.12. No third-party packages are required.

```sh
python3.12 analyse_hardware.py
```

The script reads `manifest.json`, `references.json` and the files under `raw/`.
It runs locally and does not contact QNexus, submit jobs or change the inputs.
To write the results elsewhere, use `--out <directory>`.

## Calculation

Each raw shot is a list of `[tag, value]` entries. The `res` tag holds the data
measurements: 0 and 1 are computational outcomes; 2 indicates leakage. A shot
without `res` is counted as aborted. A shot containing a 2 is rejected for
leakage. The remaining shots are accepted; only accepted all-zero outcomes
contribute to the overlap numerator. These categories are mutually exclusive.

For each overlap, the probability is the all-zero count divided by the accepted
count. Its binomial standard error is `sqrt(p * (1 - p) / N_accepted)`.
The script forms `ratio = sqrt(p_h / p_s)` and propagates the independent
overlap errors to the ratio. The energy is

```text
E = reference_energy + overlap_ratio_sign * ratio * residual_norm
```

The reference quantities, overlap signs and CASCI energies are read from
`references.json`. The energy standard error is the ratio standard error times
the residual norm. Energy-difference errors are combined in quadrature, assuming
independent geometry measurements. These are the first-order statistical
uncertainties used in the paper, not estimates of device bias.

All differences use **E_Rx - E_R0**. Errors relative to CASCI are estimate minus
CASCI; the script reports both their signed values and magnitudes in mHa.
There is no bootstrap analysis, emulator rescaling or shot extrapolation.

## Outputs

- `results/results.json`: full-precision counts, overlaps, energies, differences,
  errors and standard errors. These values can also be used for plotting.
- `results/table1_hardware.csv`: the six hardware overlap rows of Table 1.
- `results/tableS8_hardware.csv`: the six (10,10) hardware energy rows of Table S8.
- `results/energy_differences.csv`: the three hardware energy differences.

Only the table CSVs are rounded to the paper's displayed precision, using decimal
round-half-up. The JSON and energy-difference CSV retain full precision.
Running the script again overwrites these generated outputs, not the raw data.

The manifest records the source jobs, file checksums, submitted shots and method
settings. `saved_received_counts` are reference values for checking the data;
they are not used to calculate the results. The supplied checksums identify the
input files. A different or malformed input raises an error rather than being
silently dropped.

This release covers the hardware results only. It does not rerun the emulator,
repeat state selection or generate new quantum circuits.
