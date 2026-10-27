"""Late-reset state preparation, adapted from Luca's implementation.

The segment ordering and reset placement are unchanged from the hardware copy.
"""

from __future__ import annotations

from typing import Dict, Hashable, List, Optional, Set, Tuple

from guppylang import guppy
from guppylang.std.builtins import array, comptime
from guppylang.std.platform import barrier
from guppylang.std.quantum import qubit, reset
from pytket.circuit import Circuit


def build_membership(base_blocks: List[List[Hashable]]) -> Dict[Hashable, int]:
    member_of = {}
    for gi, group in enumerate(base_blocks):
        for item in group:
            if item in member_of:
                raise ValueError(f"Qubit {item} appears in more than one block.")
            member_of[item] = gi
    return member_of


def classify(
    itemset: List[Hashable], member_of: Dict[Hashable, int]
) -> Tuple[str, Optional[int], Set[int]]:
    blocks = {member_of[item] for item in itemset}
    if len(blocks) == 1:
        block_id = next(iter(blocks))
        return "single", block_id, blocks
    return "multi", None, blocks


def separate_circuit(base_blocks: List[List[Hashable]], stream):
    member_of = build_membership(base_blocks)
    segments = []
    current_items = []
    current_meta = None

    def flush() -> None:
        nonlocal current_items, current_meta
        if current_meta is not None:
            segments.append({"items": current_items, "meta": current_meta})
        current_items = []
        current_meta = None

    for command in stream:
        qubits = command.args
        mode, block_id, touched = classify(qubits, member_of)
        if current_meta is None:
            current_meta = {
                "mode": mode,
                "block_id": block_id,
                "blocks_touched": set(touched),
                "operations": [command],
            }
            current_items = [qubits]
            continue

        if current_meta["mode"] == "single":
            same_segment = mode == "single" and block_id == current_meta["block_id"]
        else:
            same_segment = mode == "multi" and current_meta["blocks_touched"] == touched

        if same_segment:
            current_items.append(qubits)
            current_meta["blocks_touched"].update(touched)
            current_meta["operations"].append(command)
        else:
            flush()
            current_meta = {
                "mode": mode,
                "block_id": block_id,
                "blocks_touched": set(touched),
                "operations": [command],
            }
            current_items = [qubits]

    flush()
    return segments


def combine_segments(segments):
    index_current = 0
    while index_current < len(segments):
        current = segments[index_current]
        index_check = 0
        while index_check < len(segments) - (index_current + 1):
            segment = segments[index_check + index_current + 1]
            if segment["meta"]["blocks_touched"].isdisjoint(
                current["meta"]["blocks_touched"]
            ):
                index_check += 1
            elif segment["meta"]["blocks_touched"] == current["meta"]["blocks_touched"]:
                segments[index_current]["items"].extend(segment["items"])
                segments[index_current]["meta"]["operations"].extend(
                    segment["meta"]["operations"]
                )
                del segments[index_check + index_current + 1]
            else:
                break
        index_current += 1
    return segments


def move_disjoint_single_segments_past_multis(segments):
    reordered = []
    pending_singles = []
    for segment in segments:
        meta = segment["meta"]
        if meta["mode"] == "single":
            pending_singles.append(segment)
            continue
        if meta["mode"] != "multi":
            reordered.extend(pending_singles)
            pending_singles = []
            reordered.append(segment)
            continue

        barrier_blocks = meta["blocks_touched"]
        still_pending = []
        for single in pending_singles:
            if single["meta"]["blocks_touched"].isdisjoint(barrier_blocks):
                still_pending.append(single)
            else:
                reordered.append(single)
        pending_singles = still_pending
        reordered.append(segment)

    reordered.extend(pending_singles)
    return reordered


def rebuild_circuit(qubits, segment) -> Circuit:
    circuit = Circuit()
    for q in qubits:
        circuit.add_qubit(q)
    for operation in segment["meta"]["operations"]:
        circuit.add_gate(operation.op, operation.args)
    return circuit


def prepare_state_late_init(circ: Circuit):
    """Reorder commuting segments and reset each data qubit before first use."""

    base_blocks = [[circ.qubits[i]] for i in range(len(circ.qubits))]
    nq = len(circ.qubits)
    segments = separate_circuit(base_blocks, circ.get_commands())
    segments = combine_segments(segments)
    segments = move_disjoint_single_segments_past_multis(segments)

    guppy_segments = []
    qubit_indices = []
    index_map = {q: i for i, q in enumerate(circ.qubits)}
    for segment_index, segment in enumerate(segments):
        touched = sorted(segment["meta"]["blocks_touched"])
        qubit_indices.append([index_map[base_blocks[i][0]] for i in touched])
        guppy_segment = guppy.load_pytket(
            f"cp_segment_{segment_index}",
            rebuild_circuit(circ.qubits, segment),
            use_arrays=True,
        )
        guppy_segments.append(guppy_segment)

    @guppy
    def barrier_wrapper(qs: array[qubit, comptime(nq)]) -> None:
        barrier(qs)

    @guppy.comptime
    def state_prep(qs: array[qubit, comptime(nq)]) -> None:
        initialized = [0 for _ in range(nq)]
        for indices, guppy_segment in zip(qubit_indices, guppy_segments):
            for index in indices:
                if initialized[index] == 0:
                    initialized[index] = 1
                    barrier_wrapper(qs)
                    reset(qs[index])
            guppy_segment(qs)

        for index in range(nq):
            if initialized[index] == 0:
                reset(qs[index])

    return state_prep
