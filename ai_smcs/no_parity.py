"""Leakage-only compute--uncompute circuits for the selected (10,10) states.

State-preparation routines adapted from Luca's implementation.
This script compiles circuits locally; it does not submit jobs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from guppylang import guppy
from guppylang.std.array import array_swap
from guppylang.std.builtins import array, comptime
from guppylang.std.platform import result
from guppylang.std.qsystem import _measure_leaked
from guppylang.std.quantum import qubit

import common
from late_init import prepare_state_late_init


CONFIGS = (
    {
        "id": "no_parity",
        "description": "Leakage-only control; no parity gadget or syndrome ancillas",
        "kp": 0,
        "constant_fan_out": False,
        "data_tree_concentration": False,
        "n_data_ancilla_connections": 1,
        "backend": "Helios-1",
        "simulator": "hardware",
        "n_qubits": 20,
    },
)


def implicit_swap_sequence(circuit) -> list[tuple[int, int]]:
    if any(command.op.type.name == "SWAP" for command in circuit.get_commands()):
        raise ValueError("late-init input unexpectedly contains explicit SWAP gates")
    explicit = circuit.copy()
    explicit.replace_implicit_wire_swaps()
    return [
        (int(command.args[0].index[0]), int(command.args[1].index[0]))
        for command in explicit.get_commands()
        if command.op.type.name == "SWAP"
    ]


def compile_program(
    state_root: Path, row: dict[str, Any]
) -> tuple[Any, int, dict[str, Any]]:
    arm = row["arm"]
    kind = row["kind"]
    psi = common.load_circuit(state_root, arm, "psil")
    uncompute_name = "phi" if kind == "s" else "chik"
    uncompute_circuit = common.load_circuit(state_root, arm, uncompute_name).dagger()
    n_data = psi.n_qubits
    alpha_set = common.ASETS[arm]
    beta_set = [index for index in range(n_data) if index not in alpha_set]
    prepare = prepare_state_late_init(psi)
    prep_swaps = implicit_swap_sequence(psi)
    uncompute = guppy.load_pytket("cu", uncompute_circuit, use_arrays=True)

    @guppy
    def restore_pytket_permutation(data: array[qubit, comptime(n_data)]) -> None:
        for pair in comptime(prep_swaps):
            left, right = pair
            array_swap(data, left, right)

    kp = int(row["kp"])
    alpha_representatives = alpha_set
    beta_representatives = beta_set

    @guppy
    def main() -> None:
        data = array(qubit() for _ in range(comptime(n_data)))
        prepare(data)
        restore_pytket_permutation(data)
        uncompute(data)
        measurements = array(_measure_leaked(q).read() for q in data)
        result("res", measurements)

    metadata = {
        "prep_swaps": prep_swaps,
        "alpha_set": alpha_set,
        "beta_set": beta_set,
        "alpha_representatives": alpha_representatives,
        "beta_representatives": beta_representatives,
        "constant_forward_fanout": bool(row["constant_fan_out"]),
        "ordinary_unfanout": kp > 0,
        "data_tree_concentration": bool(row["data_tree_concentration"]),
    }
    return main.compile(), int(row["n_qubits"]), metadata


if __name__ == "__main__":
    common.run_cli("no_parity", CONFIGS[0], compile_program)
