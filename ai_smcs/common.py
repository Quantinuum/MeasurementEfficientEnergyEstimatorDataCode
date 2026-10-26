"""Inputs and local compilation for the three Noor hardware configurations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pytket import Circuit


STATE_ROOT = Path(__file__).resolve().parent / "data"
ARMS = ("Rx", "R0")
KINDS = ("s", "h")
GEOMETRIES = {"Rx": "Rx", "R0": "R0"}

# Alpha-spin qubits after the selected spin-orbital mapping.
ASETS = {
    "Rx": [0, 2, 4, 6, 8, 10, 12, 14, 16, 18],
    "R0": [4, 6, 7, 8, 10, 11, 13, 15, 16, 19],
}

CHECKSUMS_FILE = STATE_ROOT.parent / "metadata" / "input_checksums.json"


def load_circuit(state_root: Path, arm: str, name: str) -> Circuit:
    path = state_root / f"{arm}_state_{name}.json"
    return Circuit.from_dict(json.loads(path.read_text()))


def validate_state_files(state_root: Path) -> None:
    checksums = json.loads(CHECKSUMS_FILE.read_text())
    for filename, expected in checksums.items():
        path = state_root / filename
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f"Input file has changed: {path}")
        circuit = Circuit.from_dict(json.loads(path.read_text()))
        if circuit.n_qubits != 20:
            raise ValueError(f"Expected 20 data qubits: {path}")


def run_cli(configuration: str, settings: dict, compile_program) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--geometry", choices=GEOMETRIES, help="Default: both geometries"
    )
    parser.add_argument("--overlap", choices=KINDS, help="Default: both overlaps")
    parser.add_argument("--state-root", type=Path, default=STATE_ROOT)
    parser.add_argument("--out", type=Path, default=Path("compiled") / configuration)
    args = parser.parse_args()

    validate_state_files(args.state_root)
    arms = (GEOMETRIES[args.geometry],) if args.geometry else ARMS
    kinds = (args.overlap,) if args.overlap else KINDS
    paths = [args.out / f"{arm}_{kind}.hugr" for arm in arms for kind in kinds]
    report_path = args.out / "compile_report.json"
    for path in [*paths, report_path]:
        if path.exists():
            parser.error(
                f"Output already exists; choose a different --out directory: {path}"
            )

    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    for arm in arms:
        for kind in kinds:
            row = {**settings, "arm": arm, "kind": kind}
            package, routing_qubits, metadata = compile_program(args.state_root, row)
            payload = package.to_bytes()
            path = args.out / f"{arm}_{kind}.hugr"
            path.write_bytes(payload)
            records.append(
                {
                    "geometry": arm,
                    "overlap": kind,
                    "routing_qubits": routing_qubits,
                    "package_file": path.name,
                    "package_bytes": len(payload),
                    "package_sha256": hashlib.sha256(payload).hexdigest(),
                    "metadata": metadata,
                }
            )
            print(f"{configuration}: {arm} {kind} -> {path}")
    report_path.write_text(
        json.dumps(
            {
                "configuration": configuration,
                "compiled": records,
            },
            indent=2,
        )
        + "\n"
    )
