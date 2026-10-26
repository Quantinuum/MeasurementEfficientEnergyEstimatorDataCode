"""Compute--uncompute circuits with the full measurement-based parity check (MBPC).

Parity and state-preparation routines adapted from Luca's implementation.
This script compiles circuits locally; it does not submit jobs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, no_type_check

from guppylang import guppy
from guppylang.std.array import array_swap
from guppylang.std.builtins import array, comptime, nat
from guppylang.std.platform import barrier, exit, result
from guppylang.std.qsystem import _measure_leaked
from guppylang.std.quantum import cz, h, measure, measure_array, qubit, reset, z

import common
from late_init import prepare_state_late_init


CONFIGS = (
    {
        "id": "MBPC",
        "description": "Full measurement-based parity check; ten data qubits per spin sector",
        "parity_check": True,
        "data_tree_concentration": False,
        "n_data_ancilla_connections": 1,
        "backend": "Helios-1",
        "simulator": "hardware",
        "n_qubits": 35,
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


def void_concentration(n: int):
    @guppy
    def void(qs: array[qubit, comptime(n)]) -> None:
        return None

    return void


@guppy
@no_type_check
def qarray(n: nat @ comptime) -> "array[qubit, n]":  # noqa: F821
    return array(qubit() for _ in range(n))


def get_parity_laqcc(
    n_qs_qubits: int,
    qubit_index: list[int],
    parity: bool,
    data_tree_concentration: bool,
    n_data_ancilla_connections: int,
):
    tree_forward = void_concentration(n_qs_qubits)
    tree_backward = void_concentration(n_qs_qubits)
    n_protected = len(qubit_index)

    @guppy
    @no_type_check
    def parity_laqcc(qs_inputs: array[qubit, comptime(n_qs_qubits)]) -> None:
        if comptime(data_tree_concentration):
            tree_forward(qs_inputs)
        q_target = qubit()
        equal_checks = qarray(comptime(n_protected - 3))
        target_copies = qarray(comptime(n_protected - 3))

        # Keep the target in the barrier with the data and other ancillas.
        barrier(qs_inputs, q_target, equal_checks, target_copies)
        for index in range(comptime(n_protected - 3)):
            reset(equal_checks[index])
        for index in range(comptime(n_protected - 3)):
            reset(target_copies[index])
        reset(q_target)

        h(q_target)
        for index in range(comptime(n_protected) - 3):
            h(equal_checks[index])
            h(target_copies[index])
        cz(q_target, qs_inputs[comptime(qubit_index)[0]])
        for index in range(comptime(n_protected) - 3):
            cz(equal_checks[index], target_copies[index])
        cz(q_target, equal_checks[0])
        for index in range(comptime(n_protected) - 4):
            cz(target_copies[index], equal_checks[index + 1])
        cz(
            target_copies[comptime(n_protected) - 4],
            qs_inputs[comptime(qubit_index[n_protected - 1])],
        )
        cz(q_target, qs_inputs[comptime(qubit_index)[1]])
        for index in range(comptime(n_protected) - 3):
            cz(target_copies[index], qs_inputs[comptime(qubit_index)[index + 2]])
        for index in range(comptime(n_protected) - 3):
            h(equal_checks[index])
            h(target_copies[index])

        bits_ab = measure_array(equal_checks)
        bits_c = measure_array(target_copies)
        for index in range(1, comptime(n_protected) - 3):
            bits_c[0] ^= bits_c[index]
            bits_ab[index] ^= bits_ab[index - 1]
        if bits_c[0]:
            z(q_target)
        h(q_target)
        for index in range(comptime(n_protected) - 3):
            if bits_ab[index]:
                z(qs_inputs[comptime(qubit_index)[index + 2]])
        if bits_ab[comptime(n_protected) - 4]:
            z(qs_inputs[comptime(qubit_index[n_protected - 1])])
        if comptime(data_tree_concentration):
            tree_backward(qs_inputs)
        syndrome = measure(q_target)
        if syndrome != comptime(parity):
            exit("Syndrome parity error", 1)

    parity_laqcc.check()
    return parity_laqcc, qubit_index


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

    alpha_representatives = alpha_set
    beta_representatives = beta_set
    alpha_gadget, alpha_representatives = get_parity_laqcc(
        n_data,
        alpha_set,
        False,
        bool(row["data_tree_concentration"]),
        int(row["n_data_ancilla_connections"]),
    )
    beta_gadget, beta_representatives = get_parity_laqcc(
        n_data,
        beta_set,
        False,
        bool(row["data_tree_concentration"]),
        int(row["n_data_ancilla_connections"]),
    )

    @guppy
    def main() -> None:
        data = array(qubit() for _ in range(comptime(n_data)))
        prepare(data)
        restore_pytket_permutation(data)
        alpha_gadget(data)
        beta_gadget(data)
        uncompute(data)
        measurements = array(_measure_leaked(q).read() for q in data)
        result("res", measurements)

    metadata = {
        "prep_swaps": prep_swaps,
        "alpha_set": alpha_set,
        "beta_set": beta_set,
        "alpha_representatives": alpha_representatives,
        "beta_representatives": beta_representatives,
        "data_tree_concentration": bool(row["data_tree_concentration"]),
        "q_target_in_large_barrier": True,
    }
    return main.compile(), int(row["n_qubits"]), metadata


if __name__ == "__main__":
    common.run_cli("mbpc", CONFIGS[0], compile_program)
