from __future__ import annotations

import json
import math
from pathlib import Path

import numpy
from pytket.partition import GraphColourMethod, PauliPartitionStrat

from inquanto_utils import to_complex_state
from utils import from_operator_dict, from_state_dict, partition_operator

CODE_DIR = Path(__file__).resolve().parent
DATA_DIR = CODE_DIR / "data"
OUTPUT = DATA_DIR / "pauli_averaging.json"
TARGET_STDERR_MHA = 1.6
SYSTEMS = ("rx66", "t166", "rx88", "t188", "rx1010", "t11010")


def pauli_averaging_shots(path: Path) -> int:
    data = json.loads(path.read_text())
    operator = from_operator_dict(data["hamiltonian"])
    fragments = partition_operator(
        operator,
        PauliPartitionStrat.NonConflictingSets,
        GraphColourMethod.Lazy,
    )
    state = from_state_dict(data["state"])
    complex_state = to_complex_state(state)
    variances: list[float] = []
    for fragment in fragments:
        fragment_state = fragment.apply(state)
        mean = complex_state.vdot(fragment_state)
        second = fragment_state.vdot(fragment_state)
        if abs(second.imag) > 1e-10:
            raise ValueError("Expected a real second moment")
        variances.append(max(0.0, second.real - abs(mean) ** 2))
    variance_weight = float(numpy.sum(numpy.sqrt(variances)))
    theoretical_shots = (variance_weight / (TARGET_STDERR_MHA * 1e-3)) ** 2
    return math.ceil(theoretical_shots)


def main() -> None:
    results = {
        "sigma_mha": TARGET_STDERR_MHA,
        "method": {
            "partition": "QWC",
            "graph_colouring": "Lazy",
            "term_order": "descending_absolute_coefficient",
            "identity_removed": True,
        },
        "shots": {
            key: pauli_averaging_shots(DATA_DIR / key / "jordan_wigner.json")
            for key in SYSTEMS
        },
    }
    OUTPUT.write_text(json.dumps(results, indent=2) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
