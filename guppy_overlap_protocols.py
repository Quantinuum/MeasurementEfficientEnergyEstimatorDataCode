from __future__ import annotations

from collections.abc import Callable
from math import sqrt
from typing import Any

from hugr.qsystem.result import QsysResult
from pytket._tket.circuit import CircBox
from pytket._tket.unit_id import QubitRegister
from pytket.circuit import Circuit, OpType, QControlBox
from pytket.passes import AutoRebase, DecomposeBoxes

from guppylang import comptime, guppy
from guppylang.std.builtins import array, owned, result
from guppylang.std.platform import barrier, exit
from guppylang.std.qsystem.functional import measure_and_reset_array
from guppylang.std.quantum import (
    cz,
    cx,
    discard_array,
    h,
    measure_array,
    qubit,
    toffoli,
    x,
)

from uncertainties import correlated_values, ufloat
from uncertainties.umath import sqrt as uncertainty_sqrt

SHOTS_CHI = 8000
SHOTS_PHI = 2000
SHOTS = 10000


@guppy
def cswap(control: qubit, q0: qubit, q1: qubit) -> None:
    cx(q1, q0)
    toffoli(control, q0, q1)
    cx(q1, q0)


def controlled_state(circuit: Circuit, control_state: int = 1) -> Circuit:
    """Build a decomposed controlled state-preparation circuit.

    Args:
        circuit: State-preparation circuit to control.
        control_state: Control value that activates the preparation.

    Returns:
        Controlled circuit with register "a" as the control and "q" as data.

    Note:
        This construction is inefficient, but it is useful for testing and for
        noiseless simulation.
    """
    a = QubitRegister("a", 1)
    q = QubitRegister("q", circuit.n_qubits)
    controlled = Circuit()
    controlled.add_q_register(a)
    controlled.add_q_register(q)
    box = QControlBox(CircBox(circuit), 1, control_state)
    controlled.add_qcontrolbox(box, [a[0], *q])
    DecomposeBoxes().apply(controlled)
    AutoRebase({OpType.X, OpType.H, OpType.CX, OpType.Rz, OpType.Rx}).apply(controlled)
    return controlled


def meta_fanout(n_qubits: int, unfanout: bool = False) -> Callable[..., None]:
    """Build a Guppy CX fanout helper rooted at the first qubit.

    Args:
        n_qubits: Number of qubits in the fanout register.
        unfanout: Apply the inverse fanout sequence.

    Returns:
        Guppy function taking one fanout register.
    """
    if n_qubits <= 1:

        @guppy
        def empty_fanout(
            qs: array[qubit, comptime(n_qubits)],
        ) -> None:
            return None

        return empty_fanout

    layers: list[list[tuple[int, int]]] = []
    sources = [0]
    target = 1
    while target < n_qubits:
        layer: list[tuple[int, int]] = []
        new_sources: list[int] = []
        for source in sources:
            if target >= n_qubits:
                break
            layer.append((source, target))
            new_sources.append(target)
            target += 1
        layers.append(layer)
        sources += new_sources

    seq = layers[::-1] if unfanout else layers
    pairs = [pair for layer in seq for pair in layer]

    @guppy
    def fanout(qs: array[qubit, comptime(n_qubits)]) -> None:
        for pair in comptime(pairs):
            source, target = pair
            cx(qs[source], qs[target])

    return fanout


def meta_cnx(n_qubits: int) -> tuple[Callable[..., None], int]:
    n_work = max(n_qubits - 2, 0)

    if n_qubits == 1:

        @guppy
        def cnx(
            qs: array[qubit, comptime(n_qubits)],
            target: array[qubit, comptime(1)],
            work: array[qubit, comptime(n_work)],
        ) -> None:
            cx(qs[0], target[0])

        return cnx, n_work

    if n_qubits == 2:

        @guppy
        def cnx(
            qs: array[qubit, comptime(n_qubits)],
            target: array[qubit, comptime(1)],
            work: array[qubit, comptime(n_work)],
        ) -> None:
            toffoli(qs[0], qs[1], target[0])

        return cnx, n_work

    if n_qubits == 3:

        @guppy
        def cnx(
            qs: array[qubit, comptime(n_qubits)],
            target: array[qubit, comptime(1)],
            work: array[qubit, comptime(n_work)],
        ) -> None:
            toffoli(qs[0], qs[1], work[0])
            toffoli(work[0], qs[2], target[0])
            toffoli(qs[0], qs[1], work[0])

        return cnx, n_work

    middle = list(range(2, n_qubits - 1))

    @guppy
    def cnx(
        qs: array[qubit, comptime(n_qubits)],
        target: array[qubit, comptime(1)],
        work: array[qubit, comptime(n_work)],
    ) -> None:
        toffoli(qs[0], qs[1], work[0])
        for j in comptime(middle):
            toffoli(work[j - 2], qs[j], work[j - 1])
        toffoli(work[comptime(n_qubits - 3)], qs[comptime(n_qubits - 1)], target[0])
        for j in comptime(middle[::-1]):
            toffoli(work[j - 2], qs[j], work[j - 1])
        toffoli(qs[0], qs[1], work[0])

    return cnx, n_work


def meta_parity(
    n_qubits: int,
    qubit_indices: tuple[int, ...],
    n_fanout: int,
    parity: bool,
    result_name: str = "syndrome",
) -> Callable[..., None]:
    """Build a Guppy parity check.

    We implement the parity check of https://arxiv.org/pdf/quant-ph/9903046 fig 4.

    Args:
        n_qubits: Number of protected qubits.
        qubit_indices: Protected-qubit indices included in the parity check.
        n_fanout: Number of syndrome qubits used for fanout.
        parity: Expected parity measurement result.
        result_name: Result key for the measured syndrome register.

    Returns:
        Guppy function taking protected and syndrome registers.
    """
    if n_fanout < 1:
        raise ValueError("n_fanout must be at least 1.")

    fanout = meta_fanout(n_fanout)
    unfanout = meta_fanout(n_fanout, unfanout=True)
    indices = tuple(qubit_indices)
    parity_pairs = [(i % n_fanout, index) for i, index in enumerate(indices)]

    @guppy
    def parity_gadget(
        protected_qubits: array[qubit, comptime(n_qubits)],
        syndrome_qubits: array[qubit, comptime(n_fanout)] @ owned,
    ) -> None:
        barrier(protected_qubits)
        barrier(syndrome_qubits)

        syndrome_qubits, _ = measure_and_reset_array(syndrome_qubits)

        h(syndrome_qubits[0])
        fanout(syndrome_qubits)

        for pair in comptime(parity_pairs):
            syndrome_index, protected_index = pair
            cz(syndrome_qubits[syndrome_index], protected_qubits[protected_index])

        unfanout(syndrome_qubits)
        h(syndrome_qubits[0])

        syndrome = measure_array(syndrome_qubits)
        for i in range(comptime(n_fanout)):
            if i == 0:
                if syndrome[i] != comptime(parity):
                    exit("Syndrome parity error", 1)
            else:
                if syndrome[i]:
                    exit("Syndrome extraction error", 1)
        result(comptime(result_name), syndrome)

    return parity_gadget


def even_parity_checked_state(
    state_prep: Callable[..., None],
    n_qubits: int,
    alpha_indices: tuple[int, ...] | None = None,
    beta_indices: tuple[int, ...] | None = None,
    alpha_n_fanout: int = 1,
    beta_n_fanout: int = 1,
    result_prefix: str = "",
) -> tuple[Callable[..., None], int]:
    """Wrap a Guppy state preparation with even alpha and beta parity checks.

    Args:
        state_prep: Guppy preparation function to wrap.
        n_qubits: Number of system qubits prepared by state_prep.
        alpha_indices: System-qubit indices in the alpha register. Defaults to even indices.
        beta_indices: System-qubit indices in the beta register. Defaults to odd indices.
        alpha_n_fanout: Number of alpha syndrome qubits, including the parity ancilla.
        beta_n_fanout: Number of beta syndrome qubits, including the parity ancilla.
        result_prefix: Prefix for emitted syndrome result keys.

    Returns:
        Tuple ordered as even_parity_checked_state_prep, n_parity_qubits. The
        wrapped preparation has the same call signature as state_prep.
        Add n_parity_qubits to the containing program's emulator qubit count.
    """
    alpha_indices = (
        tuple(range(0, n_qubits, 2)) if alpha_indices is None else alpha_indices
    )
    beta_indices = (
        tuple(range(1, n_qubits, 2)) if beta_indices is None else beta_indices
    )

    alpha_check = meta_parity(
        n_qubits,
        alpha_indices,
        alpha_n_fanout,
        False,
        result_name=result_prefix + "alpha_syndrome",
    )
    beta_check = meta_parity(
        n_qubits,
        beta_indices,
        beta_n_fanout,
        False,
        result_name=result_prefix + "beta_syndrome",
    )
    n_parity_qubits = alpha_n_fanout + beta_n_fanout

    @guppy
    def state_prep_parity_checked(
        qs: array[qubit, comptime(n_qubits)],
    ) -> None:
        state_prep(qs)
        barrier(qs)
        alpha_syndrome = array(qubit() for _ in range(comptime(alpha_n_fanout)))
        alpha_check(qs, alpha_syndrome)
        beta_syndrome = array(qubit() for _ in range(comptime(beta_n_fanout)))
        beta_check(qs, beta_syndrome)

    return state_prep_parity_checked, n_parity_qubits


def even_parity_checked_cstate(
    cstate_prep: Callable[..., None],
    n_qubits: int,
    alpha_indices: tuple[int, ...] | None = None,
    beta_indices: tuple[int, ...] | None = None,
    alpha_n_fanout: int = 1,
    beta_n_fanout: int = 1,
    result_prefix: str = "",
) -> tuple[Callable[..., None], int]:
    """Wrap a one-controlled Guppy state preparation with even parity checks.

    Args:
        cstate_prep: Guppy controlled preparation cstate(a, qs) to wrap.
        n_qubits: Number of system qubits prepared by cstate_prep.
        alpha_indices: System-qubit indices in the alpha register. Defaults to even indices.
        beta_indices: System-qubit indices in the beta register. Defaults to odd indices.
        alpha_n_fanout: Number of alpha syndrome qubits, including the parity ancilla.
        beta_n_fanout: Number of beta syndrome qubits, including the parity ancilla.
        result_prefix: Prefix for emitted syndrome result keys.

    Returns:
        Tuple ordered as even_parity_checked_cstate_prep, n_parity_qubits. The
        wrapped preparation has call signature cstate(a, qs).
        Add n_parity_qubits to the containing program's emulator qubit count.
    """
    alpha_indices = (
        tuple(range(0, n_qubits, 2)) if alpha_indices is None else alpha_indices
    )
    beta_indices = (
        tuple(range(1, n_qubits, 2)) if beta_indices is None else beta_indices
    )

    alpha_check = meta_parity(
        n_qubits,
        alpha_indices,
        alpha_n_fanout,
        False,
        result_name=result_prefix + "alpha_syndrome",
    )
    beta_check = meta_parity(
        n_qubits,
        beta_indices,
        beta_n_fanout,
        False,
        result_name=result_prefix + "beta_syndrome",
    )
    n_parity_qubits = alpha_n_fanout + beta_n_fanout

    @guppy
    def cstate_prep_even_parity_checked(
        a: array[qubit, comptime(1)],
        qs: array[qubit, comptime(n_qubits)],
    ) -> None:
        cstate_prep(a, qs)
        barrier(a, qs)
        alpha_syndrome = array(qubit() for _ in range(comptime(alpha_n_fanout)))
        alpha_check(qs, alpha_syndrome)
        beta_syndrome = array(qubit() for _ in range(comptime(beta_n_fanout)))
        beta_check(qs, beta_syndrome)

    return cstate_prep_even_parity_checked, n_parity_qubits


def meta_state_flag(
    phi: Callable[..., None],
    phi_dagger: Callable[..., None],
    n_qubits: int,
) -> tuple[Callable[..., None], int]:
    """Build a coherent Guppy state-flag helper for the component aligned with phi.

    Args:
        phi: Guppy state-preparation function defining the flagged state.
        phi_dagger: Guppy inverse state-preparation function.
        n_qubits: Number of system qubits.

    Returns:
        Tuple ordered as state_flag, n_work.
    """
    cnx, n_work = meta_cnx(n_qubits)

    @guppy
    def state_flag(
        qs: array[qubit, comptime(n_qubits)],
        flag: array[qubit, comptime(1)],
        work: array[qubit, comptime(n_work)],
    ) -> None:
        phi_dagger(qs)
        for i in range(comptime(n_qubits)):
            x(qs[i])
        cnx(qs, flag, work)
        for i in range(comptime(n_qubits)):
            x(qs[i])
        phi(qs)

    return state_flag, n_work


def meta_cu(
    psi: Callable[..., None],
    phi_dagger: Callable[..., None],
    chi_dagger: Callable[..., None],
    n_qubits: int,
    shots_chi: int = SHOTS_CHI,
    shots_phi: int = SHOTS_PHI,
) -> tuple[
    tuple[Callable[[], None], int, int],
    tuple[Callable[[], None], int, int],
    Callable[..., tuple[Any, ...]],
]:
    """Build compute-uncompute programs for |<psi|chi>|^2 and |<psi|phi>|^2.

    Args:
        psi: Guppy preparation function for the sampled state.
        phi_dagger: Guppy inverse preparation function for the reference state.
        chi_dagger: Guppy inverse preparation function for the correction state.
        n_qubits: Number of system qubits.
        shots_chi: Shots for the |<psi|chi>|^2 program.
        shots_phi: Shots for the |<psi|phi>|^2 program.

    Outcome mapping:
        outcome(chi_outcome, phi_outcome) reduces all-zero events and returns
        chi_counts, phi_counts, h2, s2, and h_over_s = ratio_sign * sqrt(h2 / s2).
        Here h2 = |<psi|chi>|^2 and s2 = |<psi|phi>|^2.

    Returns:
        Tuple ordered as chi_tuple, phi_tuple, outcome. Each program tuple is
        ordered as program, n_qubits, shots.
    """

    def make_program(
        state_dagger: Any, shots: int
    ) -> tuple[Callable[[], None], int, int]:
        @guppy
        def program() -> None:
            qs = array(qubit() for _ in range(comptime(n_qubits)))
            psi(qs)
            barrier(qs)
            state_dagger(qs)
            measured = measure_array(qs)
            result("q", measured)

        return program, n_qubits, shots

    def counts(
        qsys_result: QsysResult, register: str = "q"
    ) -> tuple[int, int, int, Any]:
        n_zero = 0
        n_total = 0
        for shot in qsys_result.results:
            bits = None
            for key, value in shot.entries:
                if key == register:
                    try:
                        bits = "".join(map(str, value))
                    except TypeError:
                        bits = str(value)
                    break
            if bits is None or "2" in bits:
                continue
            if bits == "0" * len(bits):
                n_zero += 1
            n_total += 1
        h2_or_s2 = n_zero / n_total if n_total else 0.0
        stderr = sqrt(h2_or_s2 * (1.0 - h2_or_s2) / n_total) if n_total else 0.0
        return n_zero, n_total - n_zero, n_total, ufloat(h2_or_s2, stderr)

    def outcome(
        chi_outcome: QsysResult,
        phi_outcome: QsysResult,
        ratio_sign: float = -1.0,
    ) -> tuple[tuple[int, int, int, Any], tuple[int, int, int, Any], Any, Any, Any]:
        chi_counts = counts(chi_outcome)
        phi_counts = counts(phi_outcome)
        h2 = chi_counts[3]
        s2 = phi_counts[3]
        if ratio_sign not in (-1.0, 1.0):
            raise ValueError("ratio_sign must be +/-1.")
        if s2.nominal_value <= 0.0:
            raise ValueError("s2 is zero; ratio undefined.")
        h2_for_ratio = ufloat(min(1.0, max(0.0, h2.nominal_value)), h2.std_dev)
        s2_for_ratio = ufloat(min(1.0, max(0.0, s2.nominal_value)), s2.std_dev)
        h_over_s = ratio_sign * uncertainty_sqrt(h2_for_ratio / s2_for_ratio)
        return chi_counts, phi_counts, h2, s2, h_over_s

    return (
        make_program(chi_dagger, shots_chi),
        make_program(phi_dagger, shots_phi),
        outcome,
    )


def meta_fcu(
    psi: Callable[..., None],
    phi: Callable[..., None],
    phi_dagger: Callable[..., None],
    chi_dagger: Callable[..., None],
    n_qubits: int,
    shots: int = SHOTS,
) -> tuple[tuple[Callable[[], None], int, int], Callable[..., tuple[Any, ...]]]:
    """Build a flagged compute-uncompute program for |<psi|chi>|^2 and |<psi|phi>|^2.

    Args:
        psi: Guppy preparation function for the sampled state.
        phi: Guppy preparation function for the flagged reference state.
        phi_dagger: Guppy inverse preparation function for the flagged reference state.
        chi_dagger: Guppy inverse preparation function for the correction state.
        n_qubits: Number of system qubits.
        shots: Shots for the joint |<psi|chi>|^2 and |<psi|phi>|^2 program.

    Outcome mapping:
        outcome(qsys_result) returns h2 = |<psi|chi>|^2 from the q=0,
        flag=0 event, s2 = |<psi|phi>|^2 from the flag=1 event, and
        h_over_s = ratio_sign * sqrt(h2 / s2). The program exits flagged
        shots after reporting flag because q is not needed for s2.
        This assumes <phi|chi> = 0. Otherwise the q=0, flag=0 event
        estimates |<chi|(I - |phi><phi|)|psi>|^2.

    Returns:
        Tuple ordered as program_tuple, outcome. The program tuple is ordered
        as program, n_qubits, shots.
    """
    state_flag, n_work = meta_state_flag(phi, phi_dagger, n_qubits)

    @guppy
    def program() -> None:
        qs = array(qubit() for _ in range(comptime(n_qubits)))
        flag = array(qubit() for _ in range(comptime(1)))
        work = array(qubit() for _ in range(comptime(n_work)))
        psi(qs)
        barrier(qs, flag)
        state_flag(qs, flag, work)
        discard_array(work)
        flag_measured = measure_array(flag)
        result("flag", flag_measured)
        if flag_measured[0]:
            exit("FCU state flag", 1)
        chi_dagger(qs)
        q_measured = measure_array(qs)
        result("q", q_measured)

    def outcome(
        qsys_result: QsysResult,
        data_register: str = "q",
        flag_register: str = "flag",
        ratio_sign: float = -1.0,
    ) -> tuple[int, int, int, int, Any, Any, Any]:
        n_h2 = 0
        n_s2 = 0
        n_total = 0
        for shot in qsys_result.results:
            flag = None
            data = None
            for key, value in shot.entries:
                if key == flag_register:
                    try:
                        flag = "".join(map(str, value))
                    except TypeError:
                        flag = str(value)
                elif key == data_register:
                    try:
                        data = "".join(map(str, value))
                    except TypeError:
                        data = str(value)
            if flag is None or "2" in flag:
                continue
            if flag in {"1", "True", "true"}:
                n_s2 += 1
                n_total += 1
            elif flag in {"0", "False", "false"}:
                if data is None or "2" in data:
                    continue
                if data == "0" * len(data):
                    n_h2 += 1
                n_total += 1
        if ratio_sign not in (-1.0, 1.0):
            raise ValueError("ratio_sign must be +/-1.")
        if n_total:
            h2_nominal = n_h2 / n_total
            s2_nominal = n_s2 / n_total
            h2_variance = h2_nominal * (1.0 - h2_nominal) / n_total
            s2_variance = s2_nominal * (1.0 - s2_nominal) / n_total
            covariance = -h2_nominal * s2_nominal / n_total
            h2, s2 = correlated_values(
                [h2_nominal, s2_nominal],
                [[h2_variance, covariance], [covariance, s2_variance]],
            )
        else:
            h2 = ufloat(0.0, 0.0)
            s2 = ufloat(0.0, 0.0)
        if n_s2:
            h_over_s = ratio_sign * uncertainty_sqrt(h2 / s2)
        else:
            h_over_s = ufloat(float("nan"), 0.0)
        return n_h2, n_s2, n_total - n_h2 - n_s2, n_total, h2, s2, h_over_s

    return (program, n_qubits + 1 + n_work, shots), outcome


def meta_sst(
    psi: Callable[..., None],
    phi: Callable[..., None],
    chi: Callable[..., None],
    n_qubits: int,
    shots_chi: int = SHOTS_CHI,
    shots_phi: int = SHOTS_PHI,
) -> tuple[
    tuple[Callable[[], None], int, int],
    tuple[Callable[[], None], int, int],
    Callable[..., tuple[Any, ...]],
]:
    """Build standard swap test programs for |<psi|chi>|^2 and |<psi|phi>|^2.

    Args:
        psi: Guppy preparation function for the sampled state.
        phi: Guppy preparation function for the reference state.
        chi: Guppy preparation function for the correction state.
        n_qubits: Number of system qubits.
        shots_chi: Shots for the |<psi|chi>|^2 program.
        shots_phi: Shots for the |<psi|phi>|^2 program.

    Outcome mapping:
        outcome(chi_outcome, phi_outcome) reduces ancilla Z means and returns
        chi_counts, phi_counts, h2, s2, and h_over_s = ratio_sign * sqrt(h2 / s2).
        Here h2 = |<psi|chi>|^2 and s2 = |<psi|phi>|^2.

    Returns:
        Tuple ordered as chi_tuple, phi_tuple, outcome. Each program tuple is
        ordered as program, n_qubits, shots.
    """

    def make_program(
        state_circuit: Any,
        shots: int,
        state_register: str,
    ) -> tuple[Callable[[], None], int, int]:
        @guppy
        def program() -> None:
            a = array(qubit() for _ in range(comptime(1)))
            psi_qs = array(qubit() for _ in range(comptime(n_qubits)))
            state_qs = array(qubit() for _ in range(comptime(n_qubits)))
            psi(psi_qs)
            state_circuit(state_qs)
            barrier(a, psi_qs, state_qs)
            h(a[0])
            for i in range(comptime(n_qubits)):
                cswap(a[0], psi_qs[i], state_qs[i])
            h(a[0])
            a_measured = measure_array(a)
            psi_measured = measure_array(psi_qs)
            state_measured = measure_array(state_qs)
            result("a", a_measured)
            result("psi", psi_measured)
            if comptime(state_register == "chi"):
                result("chi", state_measured)
            else:
                result("phi", state_measured)

        return program, 2 * n_qubits + 1, shots

    def counts(
        qsys_result: QsysResult, register: str = "a"
    ) -> tuple[int, int, int, Any]:
        n_zero = 0
        n_one = 0
        for shot in qsys_result.results:
            bits = None
            for key, value in shot.entries:
                if key == register:
                    try:
                        bits = "".join(map(str, value))
                    except TypeError:
                        bits = str(value)
                    break
            if bits is None or "2" in bits:
                continue
            if bits in {"0", "False", "false"}:
                n_zero += 1
            elif bits in {"1", "True", "true"}:
                n_one += 1
        n_total = n_zero + n_one
        h2_or_s2 = (n_zero - n_one) / n_total if n_total else 0.0
        stderr = sqrt((1.0 - h2_or_s2 * h2_or_s2) / n_total) if n_total else 0.0
        return n_zero, n_one, n_total, ufloat(h2_or_s2, stderr)

    def outcome(
        chi_outcome: QsysResult,
        phi_outcome: QsysResult,
        ratio_sign: float = -1.0,
    ) -> tuple[tuple[int, int, int, Any], tuple[int, int, int, Any], Any, Any, Any]:
        chi_counts = counts(chi_outcome)
        phi_counts = counts(phi_outcome)
        h2 = chi_counts[3]
        s2 = phi_counts[3]
        if ratio_sign not in (-1.0, 1.0):
            raise ValueError("ratio_sign must be +/-1.")
        if s2.nominal_value <= 0.0:
            raise ValueError("s2 is zero; ratio undefined.")
        h2_for_ratio = ufloat(min(1.0, max(0.0, h2.nominal_value)), h2.std_dev)
        s2_for_ratio = ufloat(min(1.0, max(0.0, s2.nominal_value)), s2.std_dev)
        h_over_s = ratio_sign * uncertainty_sqrt(h2_for_ratio / s2_for_ratio)
        return chi_counts, phi_counts, h2, s2, h_over_s

    return (
        make_program(chi, shots_chi, "chi"),
        make_program(phi, shots_phi, "phi"),
        outcome,
    )


def meta_dst(
    psi: Callable[..., None],
    phi: Callable[..., None],
    chi: Callable[..., None],
    n_qubits: int,
    shots_chi: int = SHOTS_CHI,
    shots_phi: int = SHOTS_PHI,
) -> tuple[
    tuple[Callable[[], None], int, int],
    tuple[Callable[[], None], int, int],
    Callable[..., tuple[Any, ...]],
]:
    """Build destructive swap test programs for |<psi|chi>|^2 and |<psi|phi>|^2.

    Args:
        psi: Guppy preparation function for the sampled state.
        phi: Guppy preparation function for the reference state.
        chi: Guppy preparation function for the correction state.
        n_qubits: Number of system qubits.
        shots_chi: Shots for the |<psi|chi>|^2 program.
        shots_phi: Shots for the |<psi|phi>|^2 program.

    Outcome mapping:
        outcome(chi_outcome, phi_outcome) reduces Bell-parity means and returns
        chi_counts, phi_counts, h2, s2, and h_over_s = ratio_sign * sqrt(h2 / s2).
        Here h2 = |<psi|chi>|^2 and s2 = |<psi|phi>|^2.

    Returns:
        Tuple ordered as chi_tuple, phi_tuple, outcome. Each program tuple is
        ordered as program, n_qubits, shots.
    """

    def make_program(
        state_circuit: Any,
        shots: int,
        state_register: str,
    ) -> tuple[Callable[[], None], int, int]:
        @guppy
        def program() -> None:
            psi_qs = array(qubit() for _ in range(comptime(n_qubits)))
            state_qs = array(qubit() for _ in range(comptime(n_qubits)))
            psi(psi_qs)
            state_circuit(state_qs)
            barrier(psi_qs, state_qs)
            for i in range(comptime(n_qubits)):
                cx(psi_qs[i], state_qs[i])
                h(psi_qs[i])
            psi_measured = measure_array(psi_qs)
            state_measured = measure_array(state_qs)
            result("psi", psi_measured)
            if comptime(state_register == "chi"):
                result("chi", state_measured)
            else:
                result("phi", state_measured)

        return program, 2 * n_qubits, shots

    def counts(
        qsys_result: QsysResult,
        psi_register: str = "psi",
        state_register: str = "chi",
    ) -> tuple[int, int, int, Any]:
        n_plus = 0
        n_minus = 0
        for shot in qsys_result.results:
            psi_bits = None
            state_bits = None
            for key, value in shot.entries:
                if key == psi_register:
                    try:
                        psi_bits = "".join(map(str, value))
                    except TypeError:
                        psi_bits = str(value)
                elif key == state_register:
                    try:
                        state_bits = "".join(map(str, value))
                    except TypeError:
                        state_bits = str(value)
            if psi_bits is None or state_bits is None:
                continue
            if "2" in psi_bits or "2" in state_bits:
                continue
            parity = sum(int(a) & int(b) for a, b in zip(psi_bits, state_bits)) % 2
            if parity == 0:
                n_plus += 1
            else:
                n_minus += 1
        n_total = n_plus + n_minus
        h2_or_s2 = (n_plus - n_minus) / n_total if n_total else 0.0
        stderr = sqrt((1.0 - h2_or_s2 * h2_or_s2) / n_total) if n_total else 0.0
        return n_plus, n_minus, n_total, ufloat(h2_or_s2, stderr)

    def outcome(
        chi_outcome: QsysResult,
        phi_outcome: QsysResult,
        ratio_sign: float = -1.0,
    ) -> tuple[tuple[int, int, int, Any], tuple[int, int, int, Any], Any, Any, Any]:
        chi_counts = counts(chi_outcome, state_register="chi")
        phi_counts = counts(phi_outcome, state_register="phi")
        h2 = chi_counts[3]
        s2 = phi_counts[3]
        if ratio_sign not in (-1.0, 1.0):
            raise ValueError("ratio_sign must be +/-1.")
        if s2.nominal_value <= 0.0:
            raise ValueError("s2 is zero; ratio undefined.")
        h2_for_ratio = ufloat(min(1.0, max(0.0, h2.nominal_value)), h2.std_dev)
        s2_for_ratio = ufloat(min(1.0, max(0.0, s2.nominal_value)), s2.std_dev)
        h_over_s = ratio_sign * uncertainty_sqrt(h2_for_ratio / s2_for_ratio)
        return chi_counts, phi_counts, h2, s2, h_over_s

    return (
        make_program(chi, shots_chi, "chi"),
        make_program(phi, shots_phi, "phi"),
        outcome,
    )


def meta_rht(
    cpsi: Callable[..., None],
    cphi_dagger: Callable[..., None],
    cchi_dagger: Callable[..., None],
    n_qubits: int,
    shots_chi: int = SHOTS_CHI,
    shots_phi: int = SHOTS_PHI,
) -> tuple[
    tuple[Callable[[], None], int, int],
    tuple[Callable[[], None], int, int],
    Callable[..., tuple[Any, ...]],
]:
    """Build real Hadamard test programs for Re <psi|chi> and Re <psi|phi>.

    Args:
        cpsi: Guppy one-controlled preparation function for the sampled state.
        cphi_dagger: Guppy one-controlled inverse preparation function for the reference state.
        cchi_dagger: Guppy one-controlled inverse preparation function for the correction state.
        n_qubits: Number of system qubits.
        shots_chi: Shots for the Re <psi|chi> program.
        shots_phi: Shots for the Re <psi|phi> program.

    Outcome mapping:
        outcome(chi_outcome, phi_outcome) reduces ancilla Z means and returns
        chi_counts, phi_counts, reh, res, and h_over_s = reh / res.
        Here reh = Re <psi|chi> and res = Re <psi|phi>.

    Returns:
        Tuple ordered as chi_tuple, phi_tuple, outcome. Each program tuple is
        ordered as program, n_qubits, shots.
    """

    def make_program(
        state_dagger_ctrl: Any, shots: int
    ) -> tuple[Callable[[], None], int, int]:
        @guppy
        def program() -> None:
            a = array(qubit() for _ in range(comptime(1)))
            qs = array(qubit() for _ in range(comptime(n_qubits)))
            h(a[0])
            cpsi(a, qs)
            barrier(a, qs)
            state_dagger_ctrl(a, qs)
            h(a[0])
            a_measured = measure_array(a)
            q_measured = measure_array(qs)
            result("a", a_measured)
            result("q", q_measured)

        return program, n_qubits + 1, shots

    def counts(
        qsys_result: QsysResult, register: str = "a"
    ) -> tuple[int, int, int, Any]:
        n_zero = 0
        n_one = 0
        for shot in qsys_result.results:
            bits = None
            for key, value in shot.entries:
                if key == register:
                    try:
                        bits = "".join(map(str, value))
                    except TypeError:
                        bits = str(value)
                    break
            if bits is None or "2" in bits:
                continue
            if bits in {"0", "False", "false"}:
                n_zero += 1
            elif bits in {"1", "True", "true"}:
                n_one += 1
        n_total = n_zero + n_one
        re_overlap = (n_zero - n_one) / n_total if n_total else 0.0
        stderr = sqrt((1.0 - re_overlap * re_overlap) / n_total) if n_total else 0.0
        return n_zero, n_one, n_total, ufloat(re_overlap, stderr)

    def outcome(
        chi_outcome: QsysResult,
        phi_outcome: QsysResult,
    ) -> tuple[tuple[int, int, int, Any], tuple[int, int, int, Any], Any, Any, Any]:
        chi_counts = counts(chi_outcome)
        phi_counts = counts(phi_outcome)
        reh = chi_counts[3]
        res = phi_counts[3]
        if res.nominal_value == 0.0:
            raise ValueError("res is zero; ratio undefined.")
        return chi_counts, phi_counts, reh, res, reh / res

    return (
        make_program(cchi_dagger, shots_chi),
        make_program(cphi_dagger, shots_phi),
        outcome,
    )


def meta_rist(
    cpsi: Callable[..., None],
    cphi: Callable[..., None],
    cchi: Callable[..., None],
    n_qubits: int,
    shots_chi: int = SHOTS_CHI,
    shots_phi: int = SHOTS_PHI,
) -> tuple[
    tuple[Callable[[], None], int, int],
    tuple[Callable[[], None], int, int],
    Callable[..., tuple[Any, ...]],
]:
    """Build real interferometric swap test programs for Re <psi|chi> and Re <psi|phi>.

    Args:
        cpsi: Guppy one-controlled preparation function for the sampled state.
        cphi: Guppy one-controlled preparation function for the reference state.
        cchi: Guppy one-controlled preparation function for the correction state.
        n_qubits: Number of system qubits.
        shots_chi: Shots for the Re <psi|chi> program.
        shots_phi: Shots for the Re <psi|phi> program.

    Outcome mapping:
        outcome(chi_outcome, phi_outcome) reduces ancilla Z means and returns
        chi_counts, phi_counts, reh, res, and h_over_s = reh / res.
        Here reh = Re <psi|chi> and res = Re <psi|phi>.

    Returns:
        Tuple ordered as chi_tuple, phi_tuple, outcome. Each program tuple is
        ordered as program, n_qubits, shots.
    """

    def make_program(
        state_ctrl: Any,
        shots: int,
        state_register: str,
    ) -> tuple[Callable[[], None], int, int]:
        @guppy
        def program() -> None:
            a = array(qubit() for _ in range(comptime(1)))
            psi_ctrl = array(qubit() for _ in range(comptime(1)))
            psi_qs = array(qubit() for _ in range(comptime(n_qubits)))
            state_qs = array(qubit() for _ in range(comptime(n_qubits)))
            h(a[0])
            x(psi_ctrl[0])
            cx(a[0], psi_ctrl[0])
            cpsi(psi_ctrl, psi_qs)
            state_ctrl(a, state_qs)
            cx(a[0], psi_ctrl[0])
            x(psi_ctrl[0])
            discard_array(psi_ctrl)
            barrier(a, psi_qs, state_qs)
            for i in range(comptime(n_qubits)):
                cswap(a[0], psi_qs[i], state_qs[i])
            h(a[0])
            a_measured = measure_array(a)
            psi_measured = measure_array(psi_qs)
            state_measured = measure_array(state_qs)
            result("a", a_measured)
            result("psi", psi_measured)
            if comptime(state_register == "chi"):
                result("chi", state_measured)
            else:
                result("phi", state_measured)

        return program, 2 * n_qubits + 2, shots

    def counts(
        qsys_result: QsysResult, register: str = "a"
    ) -> tuple[int, int, int, Any]:
        n_zero = 0
        n_one = 0
        for shot in qsys_result.results:
            bits = None
            for key, value in shot.entries:
                if key == register:
                    try:
                        bits = "".join(map(str, value))
                    except TypeError:
                        bits = str(value)
                    break
            if bits is None or "2" in bits:
                continue
            if bits in {"0", "False", "false"}:
                n_zero += 1
            elif bits in {"1", "True", "true"}:
                n_one += 1
        n_total = n_zero + n_one
        re_overlap = (n_zero - n_one) / n_total if n_total else 0.0
        stderr = sqrt((1.0 - re_overlap * re_overlap) / n_total) if n_total else 0.0
        return n_zero, n_one, n_total, ufloat(re_overlap, stderr)

    def outcome(
        chi_outcome: QsysResult,
        phi_outcome: QsysResult,
    ) -> tuple[tuple[int, int, int, Any], tuple[int, int, int, Any], Any, Any, Any]:
        chi_counts = counts(chi_outcome)
        phi_counts = counts(phi_outcome)
        reh = chi_counts[3]
        res = phi_counts[3]
        if res.nominal_value == 0.0:
            raise ValueError("res is zero; ratio undefined.")
        return chi_counts, phi_counts, reh, res, reh / res

    return (
        make_program(cchi, shots_chi, "chi"),
        make_program(cphi, shots_phi, "phi"),
        outcome,
    )


def meta_rist_cswap_fanout(
    cpsi: Callable[..., None],
    cphi: Callable[..., None],
    cchi: Callable[..., None],
    n_qubits: int,
    shots_chi: int = SHOTS_CHI,
    shots_phi: int = SHOTS_PHI,
) -> tuple[
    tuple[Callable[[], None], int, int],
    tuple[Callable[[], None], int, int],
    Callable[..., tuple[Any, ...]],
]:
    """Build fanout controlled-swap RIST programs for Re <psi|chi> and Re <psi|phi>.

    Args:
        cpsi: Guppy one-controlled preparation function for the sampled state.
        cphi: Guppy one-controlled preparation function for the reference state.
        cchi: Guppy one-controlled preparation function for the correction state.
        n_qubits: Number of system qubits.
        shots_chi: Shots for the Re <psi|chi> program.
        shots_phi: Shots for the Re <psi|phi> program.

    Outcome mapping:
        outcome(chi_outcome, phi_outcome) reduces ancilla Z means and returns
        chi_counts, phi_counts, reh, res, and h_over_s = reh / res.
        Here reh = Re <psi|chi> and res = Re <psi|phi>.

    Returns:
        Tuple ordered as chi_tuple, phi_tuple, outcome. Each program tuple is
        ordered as program, n_qubits, shots.
    """
    layers: list[list[tuple[int, int]]] = []
    sources = [0]
    target = 1
    while target < n_qubits:
        layer: list[tuple[int, int]] = []
        new_sources: list[int] = []
        for source in sources:
            if target >= n_qubits:
                break
            layer.append((source, target))
            new_sources.append(target)
            target += 1
        layers.append(layer)
        sources += new_sources
    fanout_pairs = [pair for layer in layers for pair in layer] or [(0, 0)]
    unfanout_pairs = [pair for layer in layers[::-1] for pair in layer] or [(0, 0)]
    n_fanout_work = max(n_qubits - 1, 0)

    def make_program(
        state_ctrl: Any,
        shots: int,
        state_register: str,
    ) -> tuple[Callable[[], None], int, int]:
        @guppy
        def program() -> None:
            a = array(qubit() for _ in range(comptime(1)))
            psi_ctrl = array(qubit() for _ in range(comptime(1)))
            psi_qs = array(qubit() for _ in range(comptime(n_qubits)))
            state_qs = array(qubit() for _ in range(comptime(n_qubits)))
            fanout_qs = array(qubit() for _ in range(comptime(n_fanout_work)))
            h(a[0])
            x(psi_ctrl[0])
            cx(a[0], psi_ctrl[0])
            cpsi(psi_ctrl, psi_qs)
            state_ctrl(a, state_qs)
            cx(a[0], psi_ctrl[0])
            x(psi_ctrl[0])
            discard_array(psi_ctrl)
            barrier(a, psi_qs, state_qs)
            if comptime(n_qubits > 1):
                for pair in comptime(fanout_pairs):
                    source, target = pair
                    if source == 0:
                        cx(a[0], fanout_qs[target - 1])
                    else:
                        cx(fanout_qs[source - 1], fanout_qs[target - 1])
            for i in range(comptime(n_qubits)):
                if i == 0:
                    cswap(a[0], psi_qs[i], state_qs[i])
                else:
                    cswap(fanout_qs[i - 1], psi_qs[i], state_qs[i])
            if comptime(n_qubits > 1):
                for pair in comptime(unfanout_pairs):
                    source, target = pair
                    if source == 0:
                        cx(a[0], fanout_qs[target - 1])
                    else:
                        cx(fanout_qs[source - 1], fanout_qs[target - 1])
            discard_array(fanout_qs)
            h(a[0])
            a_measured = measure_array(a)
            psi_measured = measure_array(psi_qs)
            state_measured = measure_array(state_qs)
            result("a", a_measured)
            result("psi", psi_measured)
            if comptime(state_register == "chi"):
                result("chi", state_measured)
            else:
                result("phi", state_measured)

        return program, 3 * n_qubits + 1, shots

    def counts(
        qsys_result: QsysResult, register: str = "a"
    ) -> tuple[int, int, int, Any]:
        n_zero = 0
        n_one = 0
        for shot in qsys_result.results:
            bits = None
            for key, value in shot.entries:
                if key == register:
                    try:
                        bits = "".join(map(str, value))
                    except TypeError:
                        bits = str(value)
                    break
            if bits is None or "2" in bits:
                continue
            if bits in {"0", "False", "false"}:
                n_zero += 1
            elif bits in {"1", "True", "true"}:
                n_one += 1
        n_total = n_zero + n_one
        re_overlap = (n_zero - n_one) / n_total if n_total else 0.0
        stderr = sqrt((1.0 - re_overlap * re_overlap) / n_total) if n_total else 0.0
        return n_zero, n_one, n_total, ufloat(re_overlap, stderr)

    def outcome(
        chi_outcome: QsysResult,
        phi_outcome: QsysResult,
    ) -> tuple[tuple[int, int, int, Any], tuple[int, int, int, Any], Any, Any, Any]:
        chi_counts = counts(chi_outcome)
        phi_counts = counts(phi_outcome)
        reh = chi_counts[3]
        res = phi_counts[3]
        if res.nominal_value == 0.0:
            raise ValueError("res is zero; ratio undefined.")
        return chi_counts, phi_counts, reh, res, reh / res

    return (
        make_program(cchi, shots_chi, "chi"),
        make_program(cphi, shots_phi, "phi"),
        outcome,
    )
