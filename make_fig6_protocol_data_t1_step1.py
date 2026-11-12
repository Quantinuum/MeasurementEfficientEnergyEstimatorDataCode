from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

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
from qsys_result_sampling import save_qsys_result_json
from selene_backend import emulate_sv

CODE_DIR = Path(__file__).resolve().parent
CASE_DIR = CODE_DIR / "data" / "t166" / "protocol"
OUTPUT_DIR = CODE_DIR / "data" / "t166" / "stderr"
SHOTS = 40_000
SHOTS_CHI = 32_000
SHOTS_PHI = 8_000
SEED = 1_000_000
SCALE = 0.0


def run_t166_m1_k10_l13_protocol_shots_v2_step1_run_raw(
    shots_chi: int = SHOTS_CHI,
    shots_phi: int = SHOTS_PHI,
    shots: int = SHOTS,
    seed: int = SEED,
    scale: float = SCALE,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    state_circuits = json.loads((CASE_DIR / "state_circuits.json").read_text())
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
    n_qubits = psi_circuit.n_qubits
    output_dir.mkdir(parents=True, exist_ok=True)

    psi = guppy.load_pytket(
        "t166_m1_k10_l13_v2_step1_psi", psi_circuit, use_arrays=True
    )
    phi = guppy.load_pytket(
        "t166_m1_k10_l13_v2_step1_phi", phi_circuit, use_arrays=True
    )
    chi = guppy.load_pytket(
        "t166_m1_k10_l13_v2_step1_chi", chi_circuit, use_arrays=True
    )
    phi_dagger = guppy.load_pytket(
        "t166_m1_k10_l13_v2_step1_phi_dagger", phi_circuit.dagger(), use_arrays=True
    )
    chi_dagger = guppy.load_pytket(
        "t166_m1_k10_l13_v2_step1_chi_dagger", chi_circuit.dagger(), use_arrays=True
    )
    cpsi = guppy.load_pytket(
        "t166_m1_k10_l13_v2_step1_cpsi", controlled_state(psi_circuit), use_arrays=True
    )
    cphi = guppy.load_pytket(
        "t166_m1_k10_l13_v2_step1_cphi", controlled_state(phi_circuit), use_arrays=True
    )
    cchi = guppy.load_pytket(
        "t166_m1_k10_l13_v2_step1_cchi", controlled_state(chi_circuit), use_arrays=True
    )
    cphi_dagger = guppy.load_pytket(
        "t166_m1_k10_l13_v2_step1_cphi_dagger",
        controlled_state(phi_circuit).dagger(),
        use_arrays=True,
    )
    cchi_dagger = guppy.load_pytket(
        "t166_m1_k10_l13_v2_step1_cchi_dagger",
        controlled_state(chi_circuit).dagger(),
        use_arrays=True,
    )

    outputs: list[dict[str, object]] = []

    # FCU
    fcu_program, _fcu_outcome = meta_fcu(
        psi, phi, phi_dagger, chi_dagger, n_qubits, shots=shots
    )
    print("Running FCU raw")
    fcu_raw = emulate_sv(
        fcu_program[0], fcu_program[1], fcu_program[2], seed=seed, scale=scale
    )
    fcu_file = "fcu_raw.qsys.json"
    save_qsys_result_json(fcu_raw, output_dir / fcu_file)
    outputs.append(
        {
            "protocol": "FCU",
            "leg": "joint",
            "qsys_result_file": fcu_file,
            "shots_requested": fcu_program[2],
            "shots_saved": len(fcu_raw.results),
            "n_qubits": fcu_program[1],
            "seed": seed,
        }
    )

    # CU
    cu_chi, cu_phi, _cu_outcome = meta_cu(
        psi, phi_dagger, chi_dagger, n_qubits, shots_chi=shots_chi, shots_phi=shots_phi
    )
    print("Running CU chi raw")
    cu_chi_raw = emulate_sv(
        cu_chi[0], cu_chi[1], cu_chi[2], seed=seed + 1 * shots, scale=scale
    )
    cu_chi_file = "cu_chi_raw.qsys.json"
    save_qsys_result_json(cu_chi_raw, output_dir / cu_chi_file)
    outputs.append(
        {
            "protocol": "CU",
            "leg": "chi",
            "qsys_result_file": cu_chi_file,
            "shots_requested": cu_chi[2],
            "shots_saved": len(cu_chi_raw.results),
            "n_qubits": cu_chi[1],
            "seed": seed + 1 * shots,
        }
    )
    print("Running CU phi raw")
    cu_phi_raw = emulate_sv(
        cu_phi[0], cu_phi[1], cu_phi[2], seed=seed + 2 * shots, scale=scale
    )
    cu_phi_file = "cu_phi_raw.qsys.json"
    save_qsys_result_json(cu_phi_raw, output_dir / cu_phi_file)
    outputs.append(
        {
            "protocol": "CU",
            "leg": "phi",
            "qsys_result_file": cu_phi_file,
            "shots_requested": cu_phi[2],
            "shots_saved": len(cu_phi_raw.results),
            "n_qubits": cu_phi[1],
            "seed": seed + 2 * shots,
        }
    )

    # DST
    dst_chi, dst_phi, _dst_outcome = meta_dst(
        psi, phi, chi, n_qubits, shots_chi=shots_chi, shots_phi=shots_phi
    )
    print("Running DST chi raw")
    dst_chi_raw = emulate_sv(
        dst_chi[0], dst_chi[1], dst_chi[2], seed=seed + 3 * shots, scale=scale
    )
    dst_chi_file = "dst_chi_raw.qsys.json"
    save_qsys_result_json(dst_chi_raw, output_dir / dst_chi_file)
    outputs.append(
        {
            "protocol": "DST",
            "leg": "chi",
            "qsys_result_file": dst_chi_file,
            "shots_requested": dst_chi[2],
            "shots_saved": len(dst_chi_raw.results),
            "n_qubits": dst_chi[1],
            "seed": seed + 3 * shots,
        }
    )
    print("Running DST phi raw")
    dst_phi_raw = emulate_sv(
        dst_phi[0], dst_phi[1], dst_phi[2], seed=seed + 4 * shots, scale=scale
    )
    dst_phi_file = "dst_phi_raw.qsys.json"
    save_qsys_result_json(dst_phi_raw, output_dir / dst_phi_file)
    outputs.append(
        {
            "protocol": "DST",
            "leg": "phi",
            "qsys_result_file": dst_phi_file,
            "shots_requested": dst_phi[2],
            "shots_saved": len(dst_phi_raw.results),
            "n_qubits": dst_phi[1],
            "seed": seed + 4 * shots,
        }
    )

    # SST
    sst_chi, sst_phi, _sst_outcome = meta_sst(
        psi, phi, chi, n_qubits, shots_chi=shots_chi, shots_phi=shots_phi
    )
    print("Running SST chi raw")
    sst_chi_raw = emulate_sv(
        sst_chi[0], sst_chi[1], sst_chi[2], seed=seed + 5 * shots, scale=scale
    )
    sst_chi_file = "sst_chi_raw.qsys.json"
    save_qsys_result_json(sst_chi_raw, output_dir / sst_chi_file)
    outputs.append(
        {
            "protocol": "SST",
            "leg": "chi",
            "qsys_result_file": sst_chi_file,
            "shots_requested": sst_chi[2],
            "shots_saved": len(sst_chi_raw.results),
            "n_qubits": sst_chi[1],
            "seed": seed + 5 * shots,
        }
    )
    print("Running SST phi raw")
    sst_phi_raw = emulate_sv(
        sst_phi[0], sst_phi[1], sst_phi[2], seed=seed + 6 * shots, scale=scale
    )
    sst_phi_file = "sst_phi_raw.qsys.json"
    save_qsys_result_json(sst_phi_raw, output_dir / sst_phi_file)
    outputs.append(
        {
            "protocol": "SST",
            "leg": "phi",
            "qsys_result_file": sst_phi_file,
            "shots_requested": sst_phi[2],
            "shots_saved": len(sst_phi_raw.results),
            "n_qubits": sst_phi[1],
            "seed": seed + 6 * shots,
        }
    )

    # RHT
    rht_chi, rht_phi, _rht_outcome = meta_rht(
        cpsi,
        cphi_dagger,
        cchi_dagger,
        n_qubits,
        shots_chi=shots_chi,
        shots_phi=shots_phi,
    )
    print("Running RHT chi raw")
    rht_chi_raw = emulate_sv(
        rht_chi[0], rht_chi[1], rht_chi[2], seed=seed + 7 * shots, scale=scale
    )
    rht_chi_file = "rht_chi_raw.qsys.json"
    save_qsys_result_json(rht_chi_raw, output_dir / rht_chi_file)
    outputs.append(
        {
            "protocol": "RHT",
            "leg": "chi",
            "qsys_result_file": rht_chi_file,
            "shots_requested": rht_chi[2],
            "shots_saved": len(rht_chi_raw.results),
            "n_qubits": rht_chi[1],
            "seed": seed + 7 * shots,
        }
    )
    print("Running RHT phi raw")
    rht_phi_raw = emulate_sv(
        rht_phi[0], rht_phi[1], rht_phi[2], seed=seed + 8 * shots, scale=scale
    )
    rht_phi_file = "rht_phi_raw.qsys.json"
    save_qsys_result_json(rht_phi_raw, output_dir / rht_phi_file)
    outputs.append(
        {
            "protocol": "RHT",
            "leg": "phi",
            "qsys_result_file": rht_phi_file,
            "shots_requested": rht_phi[2],
            "shots_saved": len(rht_phi_raw.results),
            "n_qubits": rht_phi[1],
            "seed": seed + 8 * shots,
        }
    )

    # RIST
    rist_chi, rist_phi, _rist_outcome = meta_rist(
        cpsi, cphi, cchi, n_qubits, shots_chi=shots_chi, shots_phi=shots_phi
    )
    print("Running RIST chi raw")
    rist_chi_raw = emulate_sv(
        rist_chi[0], rist_chi[1], rist_chi[2], seed=seed + 9 * shots, scale=scale
    )
    rist_chi_file = "rist_chi_raw.qsys.json"
    save_qsys_result_json(rist_chi_raw, output_dir / rist_chi_file)
    outputs.append(
        {
            "protocol": "RIST",
            "leg": "chi",
            "qsys_result_file": rist_chi_file,
            "shots_requested": rist_chi[2],
            "shots_saved": len(rist_chi_raw.results),
            "n_qubits": rist_chi[1],
            "seed": seed + 9 * shots,
        }
    )
    print("Running RIST phi raw")
    rist_phi_raw = emulate_sv(
        rist_phi[0], rist_phi[1], rist_phi[2], seed=seed + 10 * shots, scale=scale
    )
    rist_phi_file = "rist_phi_raw.qsys.json"
    save_qsys_result_json(rist_phi_raw, output_dir / rist_phi_file)
    outputs.append(
        {
            "protocol": "RIST",
            "leg": "phi",
            "qsys_result_file": rist_phi_file,
            "shots_requested": rist_phi[2],
            "shots_saved": len(rist_phi_raw.results),
            "n_qubits": rist_phi[1],
            "seed": seed + 10 * shots,
        }
    )

    (output_dir / "raw_qsys_manifest.json").write_text(
        json.dumps(
            {
                "case": "T166_m1_k10_l13",
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "state_circuits": "data/t166/protocol/state_circuits.json",
                "states": "data/t166/protocol/states.json",
                "n_qubits": n_qubits,
                "scale": scale,
                "shots": {"joint": shots, "chi": shots_chi, "phi": shots_phi},
                "outputs": outputs,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"Saved raw QsysResult JSON files to {output_dir}")


if __name__ == "__main__":
    run_t166_m1_k10_l13_protocol_shots_v2_step1_run_raw()
