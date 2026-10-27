"""Compute--uncompute circuits with the coherent parity check (CPC).

Parity and state-preparation routines adapted from Luca's implementation.
This script compiles circuits locally; it does not submit jobs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from guppylang import guppy
from guppylang.std.array import array_swap
from guppylang.std.builtins import array, comptime
from guppylang.std.platform import barrier, exit, result
from guppylang.std.qsystem import _measure_leaked
from guppylang.std.quantum import cx, cz, h, measure_array, qubit, reset

import common
from late_init import prepare_state_late_init


CONFIGS = (
    {
        "id": "CPC",
        "description": "Coherent parity check; one syndrome ancilla per spin sector",
        "kp": 1,
        "constant_fan_out": False,
        "data_tree_concentration": False,
        "n_data_ancilla_connections": 1,
        "backend": "Helios-1",
        "simulator": "hardware",
        "n_qubits": 22,
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


def meta_fanout(n: int, unfanout: bool = False):
    if n <= 1:

        @guppy
        def empty_fan(qs: array[qubit, comptime(n)]) -> None:
            return None

        return empty_fan
    indices = range(n)
    controls = indices[1:]
    layers = []
    sources = [indices[0]]
    index = 0
    while index < len(controls):
        layer = []
        new_sources = []
        for source in sources:
            if index >= len(controls):
                break
            target = controls[index]
            layer.append((source, target))
            new_sources.append(target)
            index += 1
        layers.append(layer)
        sources += new_sources
    sequence = layers[::-1] if unfanout else layers
    sequence = [pair for layer in sequence for pair in layer]

    @guppy
    def fan(qs: array[qubit, comptime(n)]) -> None:
        for pair in comptime(sequence):
            source, target = pair
            cx(qs[source], qs[target])

    return fan


def _void_concentration(n: int):
    @guppy
    def void(qs: array[qubit, comptime(n)]) -> None:
        return None

    return void


def gadget(
    n_qubits: int,
    qubit_index: list[int],
    nf: int,
    parity: bool,
    *,
    constant_fan_out: bool,
    data_tree_concentration: bool,
    n_data_ancilla_connections: int,
):
    forward_fanout = meta_fanout(nf)
    ordinary_unfanout = meta_fanout(nf, unfanout=True)
    forward_tree = _void_concentration(n_qubits)
    backward_tree = _void_concentration(n_qubits)
    representatives = qubit_index.copy()

    @guppy
    def g(protected: array[qubit, comptime(n_qubits)]) -> None:
        syndrome = array(qubit() for _ in range(comptime(nf)))
        barrier(protected, syndrome)
        for index in range(comptime(nf)):
            reset(syndrome[index])

        if comptime(constant_fan_out):
            forward_fanout(syndrome)
        else:
            h(syndrome[0])
            forward_fanout(syndrome)

        if comptime(data_tree_concentration):
            forward_tree(protected)
        for index in range(comptime(len(representatives))):
            cz(
                syndrome[index % comptime(nf)],
                protected[comptime(representatives)[index]],
            )
        if comptime(data_tree_concentration):
            backward_tree(protected)

        ordinary_unfanout(syndrome)
        h(syndrome[0])
        measurements = measure_array(syndrome)
        for index in range(comptime(nf)):
            if index == 0:
                if measurements[index] != comptime(parity):
                    exit("Syndrome parity error", 1)
            else:
                if measurements[index]:
                    exit("Syndrome extraction error", 1)

    return g, representatives


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
    alpha_gadget, alpha_representatives = gadget(
        n_data,
        alpha_set,
        kp,
        False,
        constant_fan_out=bool(row["constant_fan_out"]),
        data_tree_concentration=bool(row["data_tree_concentration"]),
        n_data_ancilla_connections=int(row["n_data_ancilla_connections"]),
    )
    beta_gadget, beta_representatives = gadget(
        n_data,
        beta_set,
        kp,
        False,
        constant_fan_out=bool(row["constant_fan_out"]),
        data_tree_concentration=bool(row["data_tree_concentration"]),
        n_data_ancilla_connections=int(row["n_data_ancilla_connections"]),
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
        "constant_forward_fanout": bool(row["constant_fan_out"]),
        "ordinary_unfanout": kp > 0,
        "data_tree_concentration": bool(row["data_tree_concentration"]),
    }
    return main.compile(), int(row["n_qubits"]), metadata


if __name__ == "__main__":
    common.run_cli("cpc", CONFIGS[0], compile_program)
