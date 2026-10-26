"""Small utility set shared by compute-uncompute and shadow processing."""

from __future__ import annotations

import json
from math import sqrt
from pathlib import Path
from typing import Any

import numpy
import pandas
from hugr.qsystem.result import QsysResult, QsysShot
from zixy.qubit import Qubits
from zixy.qubit.pauli import I, X, Y, Z
from zixy.qubit.pauli import RealTermSum as RealPauliOperator
from zixy.qubit.state import RealTermSum as RealState
from zixy.qubit.state import Strings as StateList


def save_json(obj: Any, path: str | Path = "data.json", indent: int = 4) -> None:
    """
    Safely save a Python object (usually dict) to a JSON file.
    Creates parent directories automatically.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)
        f.write("\n")  # newline for POSIX compliance


def load_json(path: str | Path) -> Any:
    """
    Safely load a JSON object from file.
    Returns the parsed Python object.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json_out(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_json(data, path, indent=2)
    print(f"# Saved {path}")


def save_text_out(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    print(f"# Saved {path}")


def mixed_estimator_from_absolute_square_counts(
    shift: float,
    scale: float,
    h_zero_counts: tuple[int, int],
    s_zero_counts: tuple[int, int],
    ratio_sign: float,
) -> tuple[float, float, float, float, float, float, float, float, float]:
    h_zero, h_other = h_zero_counts
    s_zero, s_other = s_zero_counts
    n_h = h_zero + h_other
    n_s = s_zero + s_other
    if n_h <= 0 or n_s <= 0:
        raise ValueError("Counts must be positive.")

    h_abs2 = h_zero / n_h
    s_abs2 = s_zero / n_s
    if s_abs2 <= 0.0:
        raise ValueError("s_abs2 is zero; ratio undefined.")
    if ratio_sign not in (-1.0, 1.0):
        raise ValueError("ratio_sign must be +/-1.")

    se_h_abs2 = sqrt(h_abs2 * (1.0 - h_abs2) / n_h)
    se_s_abs2 = sqrt(s_abs2 * (1.0 - s_abs2) / n_s)
    ratio_abs = sqrt(h_abs2 / s_abs2)
    ratio = ratio_sign * ratio_abs

    var_ratio = (1.0 - h_abs2) / (4.0 * s_abs2 * n_h)
    var_ratio += h_abs2 * (1.0 - s_abs2) / (4.0 * (s_abs2**2) * n_s)
    se = abs(scale) * sqrt(var_ratio)

    h_abs = sqrt(h_abs2)
    s_abs = sqrt(s_abs2)
    bias_h = (
        0.0
        if h_abs <= 1e-12
        else -scale * ratio_sign * (1.0 - h_abs2) / (8.0 * h_abs * s_abs * n_h)
    )
    bias_s = (
        3.0 * scale * ratio_sign * h_abs * (1.0 - s_abs2) / (8.0 * (s_abs**3) * n_s)
    )
    bias = bias_h + bias_s

    value = shift + scale * ratio
    value_bc = value - bias
    return value, se, bias, value_bc, h_abs2, s_abs2, se_h_abs2, se_s_abs2, ratio


def from_state_dict(data: dict[str, Any], atol: float = 1e-12) -> RealState:
    """Deserialize a dictionary into a real state.

    Args:
        data: Dictionary with `qubits` and `terms`.
        atol: Imaginary tolerance for real-mode acceptance.

    Returns:
        Real state.

    Raises:
        ValueError: If keys are invalid, lengths mismatch, or an imaginary amplitude is detected.
    """
    n = len(data["qubits"])
    qubits = Qubits.from_count(n)
    real_terms: list[tuple[list[int], float]] = []
    for entry in data["terms"]:
        key = str(entry["key"]).replace("_", "")
        if len(key) != n:
            raise ValueError(
                f"State key '{entry['key']}' has length {len(key)} but expected {n}"
            )
        if any(ch not in ("0", "1") for ch in key):
            raise ValueError(f"Invalid state key '{entry['key']}'")
        imag = float(entry.get("im", 0.0))
        if abs(imag) > atol:
            raise ValueError(f"Real state cannot contain imaginary coefficient {imag}")
        coeff = float(entry.get("re", 0.0))
        real_terms.append(([int(ch) for ch in key], coeff))
    return RealState.from_iterable(real_terms, qubits)


def from_operator_dict(
    data: dict[str, Any], atol: float = 1e-12
) -> RealPauliOperator:
    """Deserialize a JSON-like dictionary into a real Pauli operator."""
    pauli_map = {"I": I, "X": X, "Y": Y, "Z": Z}
    n = len(data["qubits"])
    qubits = Qubits.from_count(n)
    terms: list[tuple[dict[int, Any], float]] = []
    for entry in data["terms"]:
        key = str(entry["key"]).replace("_", "").upper()
        if len(key) != n or any(char not in pauli_map for char in key):
            raise ValueError(f"Invalid {n}-qubit Pauli key: {entry['key']}")
        if abs(float(entry.get("im", 0.0))) > atol:
            raise ValueError("The real Hamiltonian contains an imaginary coefficient")
        terms.append(
            (
                {
                    index: pauli_map[char]
                    for index, char in enumerate(key)
                    if char != "I"
                },
                float(entry.get("re", 0.0)),
            )
        )
    return RealPauliOperator.from_iterable(terms, qubits)


def _to_qubit_pauli_operator(operator: RealPauliOperator):
    from pytket import Qubit
    from pytket.pauli import Pauli, QubitPauliString
    from pytket.utils import QubitPauliOperator

    pauli_map = {"X": Pauli.X, "Y": Pauli.Y, "Z": Pauli.Z}
    terms = {}
    for term in operator:
        paulis = {
            Qubit(index): pauli_map[str(pauli)]
            for index, pauli in enumerate(term.string)
            if str(pauli) != "I"
        }
        string = QubitPauliString(paulis)
        terms[string] = terms.get(string, 0.0j) + complex(term.coeff)
    return QubitPauliOperator(terms)


def _from_qubit_pauli_operator(operator, qubits: Qubits) -> RealPauliOperator:
    from pytket.pauli import Pauli

    pauli_map = {Pauli.X: X, Pauli.Y: Y, Pauli.Z: Z}
    terms = []
    for string, coeff in operator.get_dict().items():
        value = complex(coeff)
        if abs(value.imag) > 1e-12:
            raise ValueError("The real Hamiltonian contains an imaginary coefficient")
        terms.append(
            (
                {
                    qubit.index[0]: pauli_map[pauli]
                    for qubit, pauli in string.map.items()
                },
                value.real,
            )
        )
    return RealPauliOperator.from_iterable(terms, qubits)


def partition_operator(
    operator: RealPauliOperator, strat: Any, method: Any
) -> list[RealPauliOperator]:
    """Partition a real Pauli operator into qubit-wise commuting fragments."""
    from pytket.pauli import QubitPauliString
    from pytket.partition import term_sequence
    from pytket.utils import QubitPauliOperator

    qpo = _to_qubit_pauli_operator(operator)
    identity = QubitPauliString()
    qpo = QubitPauliOperator(
        {pauli: coeff for pauli, coeff in qpo.get_dict().items() if pauli != identity}
    )
    paulis = sorted(qpo.get_dict(), key=lambda pauli: abs(qpo[pauli]), reverse=True)
    return [
        _from_qubit_pauli_operator(
            QubitPauliOperator({pauli: qpo[pauli] for pauli in partition}),
            operator.qubits,
        )
        for partition in term_sequence(paulis, strat=strat, method=method)
    ]


def subspace_from_state(state: RealState, n: int) -> StateList:
    """Build a basis subspace from the ``n`` largest state amplitudes."""
    if n < 1:
        raise ValueError("n must be >= 1")
    subspace = StateList(state.qubits)
    state_terms = sorted(
        (term.clone() for term in state),
        key=lambda term: abs(float(term.coeff)),
        reverse=True,
    )
    for term in state_terms[:n]:
        subspace.append([int(bit) for bit in term.string])
    return subspace


def subspace_eigenstate(
    hamiltonian: RealPauliOperator, subspace: StateList, idx: int = 0
) -> tuple[float, RealState]:
    """Solve the Hamiltonian projected into a basis-state subspace."""
    matrix = numpy.asarray(
        hamiltonian.project_into_ortho_subspace(subspace), dtype=numpy.complex128
    )
    if float(numpy.max(numpy.abs(numpy.imag(matrix)))) > 1e-12:
        raise ValueError("The projected Hamiltonian is not real")
    eigenvalues, eigenvectors = numpy.linalg.eigh(matrix)
    if idx < 0 or idx >= len(eigenvalues):
        raise IndexError(
            f"Eigen-index {idx} out of range for dimension {len(eigenvalues)}"
        )
    vector = numpy.asarray(numpy.real(eigenvectors[:, idx]), dtype=numpy.float64)
    state = RealState.from_iterable(
        [
            ([int(bit) for bit in basis], float(coeff))
            for basis, coeff in zip(subspace, vector)
            if float(coeff) != 0.0
        ],
        hamiltonian.qubits,
    )
    state.l2_normalize()
    return float(numpy.real(eigenvalues[idx])), state


def top_selected_subspace(
    hamiltonian: RealPauliOperator, state: RealState, m: int
) -> tuple[float, RealState]:
    """Return the lowest-energy state in the subspace of the top ``m`` terms."""
    energy, selected = subspace_eigenstate(
        hamiltonian, subspace_from_state(state, m)
    )
    if state.vdot(selected) < 0:
        selected *= -1.0
    return energy, selected


def drop_to_top(state: RealState, k: int) -> RealState:
    """Keep the ``k`` largest state amplitudes and normalize the result."""
    if k < 1:
        raise ValueError("k must be >= 1")
    terms = sorted(
        (term.clone() for term in state),
        key=lambda term: abs(float(term.coeff)),
        reverse=True,
    )
    truncated = RealState.from_iterable(
        [
            ([int(bit) for bit in term.string], float(term.coeff))
            for term in terms[:k]
        ],
        state.qubits,
    )
    truncated.l2_normalize()
    return truncated


def mixed_estimator_pre(
    phi: RealState,
    hamiltonian: RealPauliOperator,
    atol: float = 1e-14,
) -> tuple[float, float, RealState]:
    """Calculate the reference energy and normalized residual state."""
    hphi = hamiltonian.apply(phi)
    imag_norm = hphi.imag_part.l2_norm
    if imag_norm > atol:
        raise ValueError(f"H|phi> has a non-real component with norm {imag_norm}")
    real_hphi = hphi.real_part
    energy = float(phi.vdot(real_hphi))
    residual = real_hphi - phi * energy
    residual_norm = float(residual.l2_norm)
    if residual_norm > atol:
        residual.l2_normalize()
    else:
        residual = RealState.from_iterable([], phi.qubits)
        residual_norm = 0.0
    return energy, residual_norm, residual


def mixed_estimator(
    energy: float,
    residual_norm: float,
    residual: RealState,
    target: RealState,
    reference: RealState,
    atol: float = 1e-14,
) -> tuple[float, float, float]:
    """Evaluate the mixed estimator from target, residual, and reference states."""
    overlap = target.vdot(reference)
    residual_overlap = target.vdot(residual)
    if abs(float(overlap)) <= atol:
        raise ValueError("Overlap <target|reference> is too small")
    estimate = energy + residual_norm * residual_overlap / overlap
    return estimate, residual_overlap, overlap


def get_zero_counts(outcome: pandas.DataFrame, register: str) -> tuple[int, int]:
    col = outcome[register].astype(str)
    n_bit = len(str(col.iloc[0]))
    zero = "0" * n_bit
    valid = (
        ~outcome["_leaked"]
        if "_leaked" in outcome.columns
        else pandas.Series(True, index=outcome.index)
    )
    n_zero = int(((col == zero) & valid).sum())
    n_other = int(valid.sum() - n_zero)
    return n_zero, n_other


def absolute_square_summary_text(
    efci: float,
    shift: float,
    scale: float,
    h_exact: float,
    s_exact: float,
    h_zero_counts: tuple[int, int],
    s_zero_counts: tuple[int, int],
    ratio_sign: float,
) -> str:
    value, se, bias, _, h_abs2_est, s_abs2_est, se_h_abs2, se_s_abs2, ratio_est = (
        mixed_estimator_from_absolute_square_counts(
            shift,
            scale,
            h_zero_counts,
            s_zero_counts,
            ratio_sign,
        )
    )
    h_total = sum(h_zero_counts)
    s_total = sum(s_zero_counts)
    ratio_exact = h_exact / s_exact
    return (
        f"EFCI:         {efci: .6f}\n"
        f"Mixed:        {value: .6f} +- {se:.6f}   (Bias:  {bias: .6f})\n"
        f"Diff:         {value - efci: .6f}\n\n"
        f"h |.|^2 shots: n0: {h_zero_counts[0]:6d}  n1: {h_zero_counts[1]:6d}  nt: {h_total:6d}\n"
        f"h exact:      {h_exact: .6f}\n"
        f"h abs2 exact: {h_exact * h_exact: .6f}\n"
        f"h abs2 est:   {h_abs2_est: .6f} +- {se_h_abs2:.6f}\n"
        f"h abs2 diff:  {h_abs2_est - h_exact * h_exact: .6f}\n\n"
        f"s |.|^2 shots: n0: {s_zero_counts[0]:6d}  n1: {s_zero_counts[1]:6d}  nt: {s_total:6d}\n"
        f"s exact:      {s_exact: .6f}\n"
        f"s abs2 exact: {s_exact * s_exact: .6f}\n"
        f"s abs2 est:   {s_abs2_est: .6f} +- {se_s_abs2:.6f}\n"
        f"s abs2 diff:  {s_abs2_est - s_exact * s_exact: .6f}\n\n"
        f"ratio exact:  {ratio_exact: .6f}\n"
        f"ratio est:    {ratio_est: .6f}\n"
        f"ratio diff:   {ratio_est - ratio_exact: .6f}\n"
    )


def bits_has(s: str, value: int) -> bool:
    """Check whether a bitstring contains a given digit."""
    v = int(value)
    return any(int(c) == v for c in s)


def outcomer(r: QsysResult) -> pandas.DataFrame:
    """Convert a guppy `QsysResult` into a tabular bitstring DataFrame.

    Each quantum register appears as a string column. `_leaked` is `True` when
    any register bitstring contains ternary leakage marker `2`.
    """
    rows = []
    for result in r.results:
        data: dict[str, Any] = {
            key: "".join(map(str, value)) for (key, value) in result.entries
        }
        data["_leaked"] = any(bits_has(s, 2) for s in data.values())
        rows.append(data)
    return pandas.DataFrame(rows)


def load_shottable(filename:str) -> list[QsysShot]:
    raw = load_json(filename)
    shots = [
        QsysShot([(tag, value) for tag, value in shot])
        for shot in raw
    ]
    return shots

