from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy
from pytket.circuit import Circuit

from guppylang import guppy

from guppy_overlap_protocols import (
    controlled_state,
    meta_cu,
    meta_dst,
    meta_fcu,
    meta_rht,
    meta_rist,
    meta_sst,
)
from qsys_result_sampling import (
    bootstrap_qsys_results,
    filter_leaked_shots,
    load_qsys_result_json,
)

CODE_DIR = Path(__file__).resolve().parent
RX_CASE_DIR = CODE_DIR / "data" / "rx66" / "protocol"
RX_OUTPUT_DIR = CODE_DIR / "data" / "rx66" / "stderr"
SHOTS = 40_000
SHOTS_CHI = 32_000
SHOTS_PHI = 8_000
BOOTSTRAP_SHOTS = 10_000
BOOTSTRAP_SHOTS_CHI = 8_000
BOOTSTRAP_SHOTS_PHI = 2_000
SEED = 0
N_BOOTSTRAP = 2_000


def run_rx66_m1_k10_l13_protocol_shots_v2_step3_bootstrap(
    shots_chi: int = SHOTS_CHI,
    shots_phi: int = SHOTS_PHI,
    shots: int = SHOTS,
    seed: int = SEED,
    output_dir: Path = RX_OUTPUT_DIR,
    n_bootstrap: int = N_BOOTSTRAP,
    bootstrap_shots_chi: int = BOOTSTRAP_SHOTS_CHI,
    bootstrap_shots_phi: int = BOOTSTRAP_SHOTS_PHI,
    bootstrap_shots: int = BOOTSTRAP_SHOTS,
) -> dict[str, Any]:
    state_circuits = json.loads((RX_CASE_DIR / "state_circuits.json").read_text())
    psi_circuit = Circuit.from_dict(state_circuits["circuits"]["psil"])
    phi_circuit = Circuit.from_dict(state_circuits["circuits"]["phi"])
    chi_circuit = Circuit.from_dict(state_circuits["circuits"]["chik"])
    if (
        psi_circuit.n_qubits != phi_circuit.n_qubits
        or psi_circuit.n_qubits != chi_circuit.n_qubits
    ):
        raise ValueError(
            "psil, phi, and chik circuits must use the same number of qubits"
        )
    states = json.loads((RX_CASE_DIR / "states.json").read_text())
    n_qubits = psi_circuit.n_qubits
    ephi = float(states["ephi"])
    rphi = float(states["rphi"])
    ratio_sign = -1.0 if float(states["hkl"]) / float(states["sl"]) < 0.0 else 1.0
    output_dir.mkdir(parents=True, exist_ok=True)

    psi = guppy.load_pytket(
        "rx66_m1_k10_l13_v2_step3_psi", psi_circuit, use_arrays=True
    )
    phi = guppy.load_pytket(
        "rx66_m1_k10_l13_v2_step3_phi", phi_circuit, use_arrays=True
    )
    chi = guppy.load_pytket(
        "rx66_m1_k10_l13_v2_step3_chi", chi_circuit, use_arrays=True
    )
    phi_dagger = guppy.load_pytket(
        "rx66_m1_k10_l13_v2_step3_phi_dagger", phi_circuit.dagger(), use_arrays=True
    )
    chi_dagger = guppy.load_pytket(
        "rx66_m1_k10_l13_v2_step3_chi_dagger", chi_circuit.dagger(), use_arrays=True
    )
    cpsi = guppy.load_pytket(
        "rx66_m1_k10_l13_v2_step3_cpsi", controlled_state(psi_circuit), use_arrays=True
    )
    cphi = guppy.load_pytket(
        "rx66_m1_k10_l13_v2_step3_cphi", controlled_state(phi_circuit), use_arrays=True
    )
    cchi = guppy.load_pytket(
        "rx66_m1_k10_l13_v2_step3_cchi", controlled_state(chi_circuit), use_arrays=True
    )
    cphi_dagger = guppy.load_pytket(
        "rx66_m1_k10_l13_v2_step3_cphi_dagger",
        controlled_state(phi_circuit).dagger(),
        use_arrays=True,
    )
    cchi_dagger = guppy.load_pytket(
        "rx66_m1_k10_l13_v2_step3_cchi_dagger",
        controlled_state(chi_circuit).dagger(),
        use_arrays=True,
    )

    _fcu_program, fcu_outcome = meta_fcu(
        psi, phi, phi_dagger, chi_dagger, n_qubits, shots=shots
    )
    _cu_chi, _cu_phi, cu_outcome = meta_cu(
        psi, phi_dagger, chi_dagger, n_qubits, shots_chi=shots_chi, shots_phi=shots_phi
    )
    _dst_chi, _dst_phi, dst_outcome = meta_dst(
        psi, phi, chi, n_qubits, shots_chi=shots_chi, shots_phi=shots_phi
    )
    _sst_chi, _sst_phi, sst_outcome = meta_sst(
        psi, phi, chi, n_qubits, shots_chi=shots_chi, shots_phi=shots_phi
    )
    _rht_chi, _rht_phi, rht_outcome = meta_rht(
        cpsi,
        cphi_dagger,
        cchi_dagger,
        n_qubits,
        shots_chi=shots_chi,
        shots_phi=shots_phi,
    )
    _rist_chi, _rist_phi, rist_outcome = meta_rist(
        cpsi, cphi, cchi, n_qubits, shots_chi=shots_chi, shots_phi=shots_phi
    )

    fcu, fcu_leakage = filter_leaked_shots(
        load_qsys_result_json(output_dir / "fcu_raw.qsys.json")
    )
    cu_chi_result, cu_chi_leakage = filter_leaked_shots(
        load_qsys_result_json(output_dir / "cu_chi_raw.qsys.json")
    )
    cu_phi_result, cu_phi_leakage = filter_leaked_shots(
        load_qsys_result_json(output_dir / "cu_phi_raw.qsys.json")
    )
    dst_chi_result, dst_chi_leakage = filter_leaked_shots(
        load_qsys_result_json(output_dir / "dst_chi_raw.qsys.json")
    )
    dst_phi_result, dst_phi_leakage = filter_leaked_shots(
        load_qsys_result_json(output_dir / "dst_phi_raw.qsys.json")
    )
    sst_chi_result, sst_chi_leakage = filter_leaked_shots(
        load_qsys_result_json(output_dir / "sst_chi_raw.qsys.json")
    )
    sst_phi_result, sst_phi_leakage = filter_leaked_shots(
        load_qsys_result_json(output_dir / "sst_phi_raw.qsys.json")
    )
    rht_chi_result, rht_chi_leakage = filter_leaked_shots(
        load_qsys_result_json(output_dir / "rht_chi_raw.qsys.json")
    )
    rht_phi_result, rht_phi_leakage = filter_leaked_shots(
        load_qsys_result_json(output_dir / "rht_phi_raw.qsys.json")
    )
    rist_chi_result, rist_chi_leakage = filter_leaked_shots(
        load_qsys_result_json(output_dir / "rist_chi_raw.qsys.json")
    )
    rist_phi_result, rist_phi_leakage = filter_leaked_shots(
        load_qsys_result_json(output_dir / "rist_phi_raw.qsys.json")
    )

    fcu_values = fcu_outcome(fcu, ratio_sign=ratio_sign)
    fcu_h_over_s = fcu_values[-1]
    observed_rows = [
        {
            "protocol": "FCU",
            "sample": -1,
            "energy": ephi + rphi * float(fcu_h_over_s.nominal_value),
            "h_over_s": float(fcu_h_over_s.nominal_value),
            "h2": float(fcu_values[-3].nominal_value),
            "s2": float(fcu_values[-2].nominal_value),
        }
    ]
    fcu_bootstrap = bootstrap_qsys_results(
        fcu, n_bootstrap, seed + 1001, n_shots=bootstrap_shots
    )

    cu_values = cu_outcome(cu_chi_result, cu_phi_result, ratio_sign=ratio_sign)
    cu_h_over_s = cu_values[-1]
    observed_rows.append(
        {
            "protocol": "CU",
            "sample": -1,
            "energy": ephi + rphi * float(cu_h_over_s.nominal_value),
            "h_over_s": float(cu_h_over_s.nominal_value),
            "h2": float(cu_values[-3].nominal_value),
            "s2": float(cu_values[-2].nominal_value),
        }
    )
    cu_chi_bootstrap = bootstrap_qsys_results(
        cu_chi_result, n_bootstrap, seed + 1002, n_shots=bootstrap_shots_chi
    )
    cu_phi_bootstrap = bootstrap_qsys_results(
        cu_phi_result, n_bootstrap, seed + 1003, n_shots=bootstrap_shots_phi
    )

    dst_values = dst_outcome(dst_chi_result, dst_phi_result, ratio_sign=ratio_sign)
    dst_h_over_s = dst_values[-1]
    observed_rows.append(
        {
            "protocol": "DST",
            "sample": -1,
            "energy": ephi + rphi * float(dst_h_over_s.nominal_value),
            "h_over_s": float(dst_h_over_s.nominal_value),
            "h2": float(dst_values[-3].nominal_value),
            "s2": float(dst_values[-2].nominal_value),
        }
    )
    dst_chi_bootstrap = bootstrap_qsys_results(
        dst_chi_result, n_bootstrap, seed + 1004, n_shots=bootstrap_shots_chi
    )
    dst_phi_bootstrap = bootstrap_qsys_results(
        dst_phi_result, n_bootstrap, seed + 1005, n_shots=bootstrap_shots_phi
    )

    sst_values = sst_outcome(sst_chi_result, sst_phi_result, ratio_sign=ratio_sign)
    sst_h_over_s = sst_values[-1]
    observed_rows.append(
        {
            "protocol": "SST",
            "sample": -1,
            "energy": ephi + rphi * float(sst_h_over_s.nominal_value),
            "h_over_s": float(sst_h_over_s.nominal_value),
            "h2": float(sst_values[-3].nominal_value),
            "s2": float(sst_values[-2].nominal_value),
        }
    )
    sst_chi_bootstrap = bootstrap_qsys_results(
        sst_chi_result, n_bootstrap, seed + 1006, n_shots=bootstrap_shots_chi
    )
    sst_phi_bootstrap = bootstrap_qsys_results(
        sst_phi_result, n_bootstrap, seed + 1007, n_shots=bootstrap_shots_phi
    )

    rht_values = rht_outcome(rht_chi_result, rht_phi_result)
    rht_h_over_s = rht_values[-1]
    observed_rows.append(
        {
            "protocol": "RHT",
            "sample": -1,
            "energy": ephi + rphi * float(rht_h_over_s.nominal_value),
            "h_over_s": float(rht_h_over_s.nominal_value),
            "reh": float(rht_values[-3].nominal_value),
            "res": float(rht_values[-2].nominal_value),
        }
    )
    rht_chi_bootstrap = bootstrap_qsys_results(
        rht_chi_result, n_bootstrap, seed + 1008, n_shots=bootstrap_shots_chi
    )
    rht_phi_bootstrap = bootstrap_qsys_results(
        rht_phi_result, n_bootstrap, seed + 1009, n_shots=bootstrap_shots_phi
    )

    rist_values = rist_outcome(rist_chi_result, rist_phi_result)
    rist_h_over_s = rist_values[-1]
    observed_rows.append(
        {
            "protocol": "RIST",
            "sample": -1,
            "energy": ephi + rphi * float(rist_h_over_s.nominal_value),
            "h_over_s": float(rist_h_over_s.nominal_value),
            "reh": float(rist_values[-3].nominal_value),
            "res": float(rist_values[-2].nominal_value),
        }
    )
    rist_chi_bootstrap = bootstrap_qsys_results(
        rist_chi_result, n_bootstrap, seed + 1010, n_shots=bootstrap_shots_chi
    )
    rist_phi_bootstrap = bootstrap_qsys_results(
        rist_phi_result, n_bootstrap, seed + 1011, n_shots=bootstrap_shots_phi
    )

    sample_rows: list[dict[str, Any]] = []
    for sample in range(n_bootstrap):
        try:
            fcu_sample = fcu_outcome(next(fcu_bootstrap), ratio_sign=ratio_sign)
            fcu_sample_h_over_s = fcu_sample[-1]
            sample_rows.append(
                {
                    "protocol": "FCU",
                    "sample": sample,
                    "energy": ephi + rphi * float(fcu_sample_h_over_s.nominal_value),
                    "h_over_s": float(fcu_sample_h_over_s.nominal_value),
                    "h2": float(fcu_sample[-3].nominal_value),
                    "s2": float(fcu_sample[-2].nominal_value),
                }
            )
        except ValueError:
            pass
        try:
            cu_sample = cu_outcome(
                next(cu_chi_bootstrap), next(cu_phi_bootstrap), ratio_sign=ratio_sign
            )
            cu_sample_h_over_s = cu_sample[-1]
            sample_rows.append(
                {
                    "protocol": "CU",
                    "sample": sample,
                    "energy": ephi + rphi * float(cu_sample_h_over_s.nominal_value),
                    "h_over_s": float(cu_sample_h_over_s.nominal_value),
                    "h2": float(cu_sample[-3].nominal_value),
                    "s2": float(cu_sample[-2].nominal_value),
                }
            )
        except ValueError:
            pass
        try:
            dst_sample = dst_outcome(
                next(dst_chi_bootstrap), next(dst_phi_bootstrap), ratio_sign=ratio_sign
            )
            dst_sample_h_over_s = dst_sample[-1]
            sample_rows.append(
                {
                    "protocol": "DST",
                    "sample": sample,
                    "energy": ephi + rphi * float(dst_sample_h_over_s.nominal_value),
                    "h_over_s": float(dst_sample_h_over_s.nominal_value),
                    "h2": float(dst_sample[-3].nominal_value),
                    "s2": float(dst_sample[-2].nominal_value),
                }
            )
        except ValueError:
            pass
        try:
            sst_sample = sst_outcome(
                next(sst_chi_bootstrap), next(sst_phi_bootstrap), ratio_sign=ratio_sign
            )
            sst_sample_h_over_s = sst_sample[-1]
            sample_rows.append(
                {
                    "protocol": "SST",
                    "sample": sample,
                    "energy": ephi + rphi * float(sst_sample_h_over_s.nominal_value),
                    "h_over_s": float(sst_sample_h_over_s.nominal_value),
                    "h2": float(sst_sample[-3].nominal_value),
                    "s2": float(sst_sample[-2].nominal_value),
                }
            )
        except ValueError:
            pass
        try:
            rht_sample = rht_outcome(next(rht_chi_bootstrap), next(rht_phi_bootstrap))
            rht_sample_h_over_s = rht_sample[-1]
            sample_rows.append(
                {
                    "protocol": "RHT",
                    "sample": sample,
                    "energy": ephi + rphi * float(rht_sample_h_over_s.nominal_value),
                    "h_over_s": float(rht_sample_h_over_s.nominal_value),
                    "reh": float(rht_sample[-3].nominal_value),
                    "res": float(rht_sample[-2].nominal_value),
                }
            )
        except ValueError:
            pass
        try:
            rist_sample = rist_outcome(
                next(rist_chi_bootstrap), next(rist_phi_bootstrap)
            )
            rist_sample_h_over_s = rist_sample[-1]
            sample_rows.append(
                {
                    "protocol": "RIST",
                    "sample": sample,
                    "energy": ephi + rphi * float(rist_sample_h_over_s.nominal_value),
                    "h_over_s": float(rist_sample_h_over_s.nominal_value),
                    "reh": float(rist_sample[-3].nominal_value),
                    "res": float(rist_sample[-2].nominal_value),
                }
            )
        except ValueError:
            pass

    samples_path = output_dir / "protocol_samples.jsonl"
    with samples_path.open("w") as fobj:
        for row in sample_rows:
            fobj.write(json.dumps(row, sort_keys=True) + "\n")

    summary: dict[str, Any] = {
        "case": "Rx66_m1_k10_l13",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "reference": {
            "efci": float(states["efci"]),
            "ephi": ephi,
            "rphi": rphi,
            "h": float(states["hkl"]),
            "s": float(states["sl"]),
            "h_over_s": float(states["hkl"]) / float(states["sl"]),
            "energy": float(states["emekl"]),
        },
        "bootstrap_seed": seed + 1000,
        "n_bootstrap": n_bootstrap,
        "raw_shots": {"joint": shots, "chi": shots_chi, "phi": shots_phi},
        "bootstrap_sample_shots": {
            "joint": bootstrap_shots,
            "chi": bootstrap_shots_chi,
            "phi": bootstrap_shots_phi,
        },
        "leakage_filter": {
            "fcu": fcu_leakage,
            "cu_chi": cu_chi_leakage,
            "cu_phi": cu_phi_leakage,
            "dst_chi": dst_chi_leakage,
            "dst_phi": dst_phi_leakage,
            "sst_chi": sst_chi_leakage,
            "sst_phi": sst_phi_leakage,
            "rht_chi": rht_chi_leakage,
            "rht_phi": rht_phi_leakage,
            "rist_chi": rist_chi_leakage,
            "rist_phi": rist_phi_leakage,
        },
        "observed": {row["protocol"]: row for row in observed_rows},
        "protocols": {},
    }
    histograms: dict[str, Any] = {}
    for protocol in ("FCU", "CU", "DST", "SST", "RHT", "RIST"):
        energies = numpy.asarray(
            [row["energy"] for row in sample_rows if row["protocol"] == protocol]
        )
        energies = energies[numpy.isfinite(energies)]
        if len(energies) == 0:
            continue
        counts, edges = numpy.histogram(energies, bins=50)
        summary["protocols"][protocol] = {
            "mean": float(numpy.mean(energies)),
            "std": float(numpy.std(energies, ddof=1)) if len(energies) > 1 else 0.0,
            "q025": float(numpy.quantile(energies, 0.025)),
            "q500": float(numpy.quantile(energies, 0.500)),
            "q975": float(numpy.quantile(energies, 0.975)),
        }
        histograms[protocol] = {
            "bin_edges": [float(edge) for edge in edges],
            "counts": [int(count) for count in counts],
        }

    (output_dir / "energy_bootstrap_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "energy_bootstrap_histograms.json").write_text(
        json.dumps(histograms, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "reference.json").write_text(
        json.dumps(
            {
                "efci": float(states["efci"]),
                "energy": float(states["emekl"]),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"Saved bootstrap samples to {samples_path}")
    print(f"Saved bootstrap summary to {output_dir / 'energy_bootstrap_summary.json'}")
    print(
        f"Saved bootstrap histograms to {output_dir / 'energy_bootstrap_histograms.json'}"
    )
    return summary


if __name__ == "__main__":
    run_rx66_m1_k10_l13_protocol_shots_v2_step3_bootstrap()
