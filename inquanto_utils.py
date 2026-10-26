"""Minimal InQuanto state helpers used by classical-shadow processing."""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional

from inquanto.operators import FermionOperator
from inquanto.states import FermionState, FermionStateString, QubitState
from zixy.qubit.state import ComplexTermSum as ComplexState
from zixy.qubit.state import RealTermSum as RealState


def eval_mixed_estimator(
    eval_overlaps: Callable,
    f_op: FermionOperator | FermionState,
    ref_state: FermionState,
    tol: float = 0.0,
    eval_inputs: Optional[dict] = None,
):
    """Evaluate a mixed estimator from measured overlaps."""
    if eval_inputs is None:
        eval_inputs = {}

    ref_dets = [
        [occupation for _, occupation in state.items()]
        for state in ref_state.basis_states
    ]
    ref_det_strings = ["".join(map(str, determinant)) for determinant in ref_dets]
    coefficients = ref_state.coefficients

    if isinstance(f_op, FermionOperator):
        h_ref_p = f_op.apply_ket(ref_state)
    elif isinstance(f_op, FermionState):
        h_ref_p = f_op
    else:
        raise TypeError("f_op must be a FermionOperator or FermionState")

    ref_overlaps = eval_overlaps(ref_det_strings, eval_inputs)
    reference_overlap = sum(
        coefficient * overlap
        for coefficient, overlap in zip(coefficients, ref_overlaps)
    )

    corr_vec_strings = [
        ("".join(str(occupation) for _, occupation in state.items()), h_ref_p[state])
        for state in h_ref_p.basis_states
    ]
    corr_overlaps = eval_overlaps(
        [determinant for determinant, _ in corr_vec_strings], eval_inputs
    )
    excited_overlaps = FermionState(
        {
            FermionStateString([int(bit) for bit in determinant]): overlap
            for (determinant, _), overlap in zip(corr_vec_strings, corr_overlaps)
        }
    )
    correlation_energy = h_ref_p.vdot(excited_overlaps)
    return correlation_energy / reference_overlap


def drop_to_inq_by_count(state: FermionState, determinant_count: int) -> FermionState:
    """Keep the ``determinant_count`` largest-magnitude state terms."""
    pruned = FermionState()
    state_terms = sorted(state.items(), key=lambda item: abs(item[1]), reverse=True)
    for basis, coefficient in state_terms[:determinant_count]:
        pruned += FermionState({basis: coefficient})
    return pruned


def permutation_count_bubble(values: list[int]) -> int:
    """Count inversions to determine the fermionic permutation sign."""
    values = values.copy()
    inversions = 0
    for i in range(len(values) - 1):
        for j in range(len(values) - i - 1):
            if values[j] > values[j + 1]:
                values[j], values[j + 1] = values[j + 1], values[j]
                inversions += 1
    return inversions


def permute_fermion_state(
    fermion_state: FermionState, permutation_mapping: dict[int, int]
) -> FermionState:
    """Permute a FermionState's orbitals and apply the fermionic sign."""
    permutated_pairs = []
    inverse_mapping = {value: key for key, value in permutation_mapping.items()}

    for basis, coefficient in fermion_state.items():
        occupation_numbers = list(basis.values())
        permutated_occupation_numbers = [
            occupation_numbers[inverse_mapping[index]]
            for index in range(len(occupation_numbers))
        ]
        occupied_orbitals = [
            index
            for index, occupation in enumerate(occupation_numbers)
            if occupation == 1
        ]
        occupied_permuted = [
            list(permutation_mapping.keys()).index(index) for index in occupied_orbitals
        ]
        sign = (-1) ** (permutation_count_bubble(occupied_permuted) % 2)
        permutated_basis = FermionStateString(
            dict(enumerate(permutated_occupation_numbers))
        )
        permutated_pairs.append((permutated_basis, sign * coefficient))
    return FermionState(dict(permutated_pairs))


def to_fermion_state(state) -> FermionState:
    """Convert a serialized or qubit state to an InQuanto FermionState."""
    qubit_state = QubitState.from_string(str(state))
    return FermionState.from_ndarray(qubit_state.to_ndarray())


def to_complex_state(state: RealState) -> ComplexState:
    """Convert a real Zixy state to a complex-valued Zixy state."""
    return ComplexState.from_iterable(
        [([int(bit) for bit in term.string], complex(term.coeff)) for term in state],
        state.qubits,
    )


__all__ = [
    "drop_to_inq_by_count",
    "eval_mixed_estimator",
    "permutation_count_bubble",
    "permute_fermion_state",
    "to_fermion_state",
    "to_complex_state",
]
