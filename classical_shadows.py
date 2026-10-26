"""Shared post-processing for TUPS and SMCS classical-shadow shots."""

import random
from collections import Counter
from collections.abc import Callable, Iterable
from typing import Any, Optional

import numpy as np
import stim
from hugr.qsystem.result import QsysShot, QsysResult
from qiskit import quantum_info
from pytket.circuit import Circuit
from pytket.extensions.qiskit import qiskit_to_tk

from MeasurementEfficientEnergyEstimatorDataCode.utils import from_state_dict
from MeasurementEfficientEnergyEstimatorDataCode.inquanto_utils import (
    drop_to_inq_by_count,
    eval_mixed_estimator,
    permute_fermion_state,
    to_fermion_state,
)

import json
from qiskit.quantum_info import Clifford


def load_json_info(filename):
    def restore_complex(value):
        if set(value) == {"real", "imag"}:
            return complex(value["real"], value["imag"])
        return value

    with open(filename, encoding="utf-8") as f:
        info = json.load(f, object_hook=restore_complex)

    info["tableaus"] = [Clifford.from_dict(tableau) for tableau in info["tableaus"]]

    for key in ("all_measurements_execute",):
        info[key] = [
            Counter({tuple(map(int, bits)): count for bits, count in counts.items()})
            for counts in info[key]
        ]

    for key in ("execute_results",):
        info[key] = [
            QsysResult(result) if result is not None else None for result in info[key]
        ]

    return info


def slater_det_2_qubit_reorder(slater_det, spinorb_to_qubit_indices):
    qubit_basis_state = "0" * len(slater_det)
    for n in range(len(slater_det)):
        if slater_det[n] == "1":
            # print(n,spinorb_to_qubit_indices[n])
            qubit_basis_state = (
                qubit_basis_state[: spinorb_to_qubit_indices[n]]
                + "1"
                + qubit_basis_state[spinorb_to_qubit_indices[n] + 1 :]
            )
            # print(slater_det,qubit_basis_state)
    return qubit_basis_state


def pauli_mult(a, b):
    if a == "I":
        return [b, 1]
    elif b == "I":
        return [a, 1]
    elif a == b:
        return ["I", 1]
    elif a == "X":
        return ["Y", complex(0, -1)] if b == "Z" else ["Z", complex(0, 1)]
    elif a == "Y":
        return ["X", complex(0, 1)] if b == "Z" else ["Z", complex(0, -1)]
    elif a == "Z":
        return ["Y", complex(0, 1)] if b == "X" else ["X", complex(0, -1)]


def ROWMULT(M, i, j):
    result = []
    phase = M[i][-1] * M[j][-1]  # Multiply phase factors
    for a, b in zip(M[i][:-1], M[j][:-1]):  # Exclude the phase in the multiplication
        p_mult = pauli_mult(a, b)
        result.append(p_mult[0])
        phase = phase * p_mult[1]

    result.append(phase)
    return result


def ROWSWAP(M, i, j):
    M[i], M[j] = M[j], M[i]


def reduce_to_row_echelon(M):
    ### https://arxiv.org/pdf/1711.07848 ###
    n = len(M)
    i = 0

    # Setup X block
    for j in range(n):
        k = next((k for k in range(i, n) if M[k][j] in {"X", "Y"}), None)
        if k is not None:
            ROWSWAP(M, i, k)
            for m in range(n):
                if m != i and M[m][j] in {"X", "Y"}:
                    M[m] = ROWMULT(M, i, m)
            i += 1

    # Setup Z block
    for j in range(n):
        k = next((k for k in range(i, n) if M[k][j] == "Z"), None)
        if k is not None:
            ROWSWAP(M, i, k)
            for m in range(n):
                if m != i and M[m][j] in {"Z", "Y"}:
                    M[m] = ROWMULT(M, i, m)
            i += 1

    return M


def apply_stab_mat_row(pre_state, row, phase):
    post_state = pre_state
    phase = phase * row[-1]
    for n in range(len(post_state)):
        if post_state[n] == "1":
            if row[n] == "X":
                post_state = post_state[:n] + "0" + post_state[n + 1 :]
                phase = phase * 1
            elif row[n] == "Y":
                post_state = post_state[:n] + "0" + post_state[n + 1 :]
                phase = phase * complex(0, -1)
            elif row[n] == "Z":
                post_state = post_state[:n] + "1" + post_state[n + 1 :]
                phase = phase * -1
            elif row[n] == "I":
                post_state = post_state[:n] + "1" + post_state[n + 1 :]
        elif post_state[n] == "0":
            if row[n] == "X":
                post_state = post_state[:n] + "1" + post_state[n + 1 :]
                phase = phase * 1
            elif row[n] == "Y":
                post_state = post_state[:n] + "1" + post_state[n + 1 :]
                phase = phase * complex(0, 1)
            elif row[n] == "Z":
                post_state = post_state[:n] + "0" + post_state[n + 1 :]
                phase = phase * 1
            elif row[n] == "I":
                post_state = post_state[:n] + "0" + post_state[n + 1 :]

    return post_state, phase


def get_ovlp_phase_basis(ref_state, ovlp_state, reduced_stab_matrix):
    # ref state is one of the computational basis states that exist in the stabilizer represented by reduced stab matrix. ovlp_state is a computational basis state
    phase = 1
    interm_state = ref_state
    for n in range(len(ovlp_state)):
        # print(ovlp_state,interm_state,phase)
        # print(ovlp_state[n],interm_state[n])
        if ovlp_state[n] != interm_state[n]:
            applied_q = False
            for m in range(len(ovlp_state)):
                if (
                    (
                        reduced_stab_matrix[m][n] == "Y"
                        or reduced_stab_matrix[m][n] == "X"
                    )
                    and ("X" not in reduced_stab_matrix[m][:n])
                    and ("Y" not in reduced_stab_matrix[m][:n])
                ):
                    # if (reduced_stab_matrix[n][n]=="X" or reduced_stab_matrix[n][n]=="Y"): #if there exists a row with X or Y on the furthest left non I or Z element for the column of interest
                    interm_state, phase = apply_stab_mat_row(
                        interm_state, reduced_stab_matrix[m], phase
                    )
                    applied_q = True
            if not applied_q:
                phase = 0.0
    return phase


def x_rank_of_reduced(reduced_M):
    for m in range(len(reduced_M)):
        if "X" not in reduced_M[m] and "Y" not in reduced_M[m]:
            return m

    return len(reduced_M)


class GlobalCliffordShadow:
    def __init__(self, n_circuits: int, n_qubits: int):
        self.n_circuits = n_circuits
        # self.shots_per_circuit = shots_per_circuit
        self.n_qubits = n_qubits
        self.post_process_db = {}

    def generate_random_cliffords(self, n_U=None):
        ### n_U, number of Cliffords to generate, circuit is the state circuit we want to evaluate overlaps of (only needed for qubit number), backend lets us compile the cliffords to a specific gate set if desired
        if not n_U:
            n_U = self.n_circuits
        shadow_tableau_list = []
        for n in range(n_U):
            random_tableau = stim.Tableau.random(self.n_qubits)
            symp_mat = [
                str(x).replace("_", "I")
                for x in [
                    random_tableau.x_output(k) for k in range(len(random_tableau))
                ]
            ] + [str(x).replace("_", "I") for x in random_tableau.to_stabilizers()]
            rand_clifford = quantum_info.Clifford(symp_mat)
            shadow_tableau_list.append(rand_clifford)

        shadow_circs = []
        for rand_clifford in shadow_tableau_list:
            rand_cliff_circ = rand_clifford.to_circuit()
            tket_clifford = qiskit_to_tk(rand_cliff_circ)

            shadow_circs.append(tket_clifford)

        self.shadow_tableau_list = shadow_tableau_list
        self.shadow_circs = shadow_circs
        # return shadow_tableau_list, shadow_circs

    def set_cliffords_tableaus(self, tableaus):
        self.shadow_tableau_list = tableaus
        self.shadow_circs = [qiskit_to_tk(x.to_circuit()) for x in tableaus]
        self.n_circuits = len(tableaus)

    def set_shadow_outcomes(self, outcomes):
        self.shadow_outcomes = outcomes

    def get_clifford_circuits(self):
        return self.shadow_circs

    def randomise_clifford_meas_lists(self):
        shadow_measurements = []
        for counts in self.shadow_outcomes:
            shadow_measurements.append(
                [list(x) for x in list(counts.keys())[:] for y in range(counts[x])]
            )
        shadow_tableau_list_expanded = [
            self.shadow_tableau_list[x]
            for x in range(len(list(self.shadow_tableau_list)))
            for y in range(len(shadow_measurements[x]))
        ]
        single_list_measurements = [
            x[y] for x in shadow_measurements for y in range(len(x))
        ]
        shuffled_meas_pairs = list(
            zip(shadow_tableau_list_expanded, single_list_measurements)
        )
        random.shuffle(
            shuffled_meas_pairs
        )  #### shuffle all clifford, meaurement pairs randomly
        self.shuffled_Us = [x[0] for x in shuffled_meas_pairs]
        self.shuffled_measurements = [[x[1]] for x in shuffled_meas_pairs]

    def bootstrap_sample(self, n_samples):
        shadow_measurements = []
        for counts in self.shadow_outcomes:
            shadow_measurements.append(
                [list(x) for x in list(counts.keys())[:] for y in range(counts[x])]
            )
        shadow_tableau_list_expanded = [
            self.shadow_tableau_list[x]
            for x in range(len(list(self.shadow_tableau_list)))
            for y in range(len(shadow_measurements[x]))
        ]
        single_list_measurements = [
            x[y] for x in shadow_measurements for y in range(len(x))
        ]
        shuffled_meas_pairs = list(
            zip(shadow_tableau_list_expanded, single_list_measurements)
        )
        # random.shuffle(shuffled_meas_pairs) #### shuffle all clifford, meaurement pairs randomly
        # self.shuffled_Us = [x[0] for x in shuffled_meas_pairs]
        # self.shuffled_measurements = [[x[1]] for x in shuffled_meas_pairs]
        #
        # sample = np.random.choice(single_list_measurements, size=len(single_list_measurements), replace=True)
        idx = np.random.randint(0, len(shuffled_meas_pairs), size=n_samples)
        sample = [shuffled_meas_pairs[i] for i in idx]
        return [x[0] for x in sample], [[x[1]] for x in sample]

    def measure_circuits(
        self,
        state_prep_circuit,
        shots_per_circuit,
        backend,
        shadow_circs_to_use=None,
        seed=123,
    ):
        self.shots_per_circuit = shots_per_circuit
        if not shadow_circs_to_use:
            shadow_circs_to_use = len(self.shadow_tableau_list)

        def sampler(circ: Circuit, n_qubit: int, n_shots: int, seed: int = 0):
            backend.empty_cache()
            circ.measure_all()
            circ = backend.get_compiled_circuit(circ)
            result = backend.run_circuit(circ, n_shots, seed=seed)
            return result.get_counts()

        outcomes = []
        for i, shadow_circuit in enumerate(self.shadow_circs):
            if i < shadow_circs_to_use:
                circ = state_prep_circuit.copy()
                circ.append(shadow_circuit)
                outcome = sampler(circ, self.n_qubits, self.shots_per_circuit, seed)
                outcomes.append(outcome)
        self.shadow_outcomes = outcomes

    def eval_overlaps_w_permutation(
        self, all_addr, spinorb_to_qubit_indices, eval_inputs={}
    ):
        shadow_measurements = []
        for counts in self.shadow_outcomes:
            shadow_measurements.append(
                [list(x) for x in list(counts.keys())[:] for y in range(counts[x])]
            )
        # shadow_tableau_list_expanded = [self.shadow_tableau_list[x] for x in range(len(list(self.shadow_tableau_list))) for y in range(len(shadow_measurements[x]))]
        # single_list_measurements = [x[y] for x in shadow_measurements for y in range(len(x))]
        op_list = []
        for n in all_addr:
            index_1_r = slater_det_2_qubit_reorder(n, spinorb_to_qubit_indices)[
                ::-1
            ]  ##n[::-1]
            op_list.append([1.0, int(index_1_r, 2), 0])

        c_ijab = self.density_mat_stabilizer(
            self.shadow_tableau_list, shadow_measurements, op_list
        )
        total_meas = sum([len(x) for x in shadow_measurements])
        # print("total shots: ",total_meas)
        return [x / total_meas for x in c_ijab]

    def eval_overlaps(self, all_addr, eval_inputs={}):
        shadow_measurements = []
        for counts in self.shadow_outcomes:
            shadow_measurements.append(
                [list(x) for x in list(counts.keys())[:] for y in range(counts[x])]
            )
        # shadow_tableau_list_expanded = [self.shadow_tableau_list[x] for x in range(len(list(self.shadow_tableau_list))) for y in range(len(shadow_measurements[x]))]
        # single_list_measurements = [x[y] for x in shadow_measurements for y in range(len(x))]
        op_list = []
        for n in all_addr:
            index_1_r = n[::-1]  ##n[::-1]
            op_list.append([1.0, int(index_1_r, 2), 0])

        c_ijab = self.density_mat_stabilizer(
            self.shadow_tableau_list, shadow_measurements, op_list
        )
        total_meas = sum([len(x) for x in shadow_measurements])
        # print("total shots: ",total_meas)
        return [x / total_meas for x in c_ijab]

    def eval_overlaps_randomised_part_w_permutation(
        self, all_addr, spinorb_to_qubit_indices, eval_inputs
    ):
        n_groups = eval_inputs["n_groups"]
        index = eval_inputs["index"]
        total_meas = len(self.shuffled_measurements)
        shots_per_std_part = (total_meas) // n_groups
        op_list = []
        for n in all_addr:
            index_1_r = slater_det_2_qubit_reorder(n, spinorb_to_qubit_indices)[
                ::-1
            ]  ##n[::-1]
            op_list.append([1.0, int(index_1_r, 2), 0])

        c_ijab = self.density_mat_stabilizer(
            self.shuffled_Us[
                index * shots_per_std_part : (index + 1) * shots_per_std_part
            ],
            self.shuffled_measurements[
                index * shots_per_std_part : (index + 1) * shots_per_std_part
            ],
            op_list,
        )
        return [x / shots_per_std_part for x in c_ijab]

    def eval_overlaps_randomised_part(self, all_addr, eval_inputs):
        n_groups = eval_inputs["n_groups"]
        index = eval_inputs["index"]
        total_meas = len(self.shuffled_measurements)
        shots_per_std_part = (total_meas) // n_groups
        op_list = []
        for n in all_addr:
            index_1_r = n[::-1]  ##n[::-1]
            op_list.append([1.0, int(index_1_r, 2), 0])

        c_ijab = self.density_mat_stabilizer(
            self.shuffled_Us[
                index * shots_per_std_part : (index + 1) * shots_per_std_part
            ],
            self.shuffled_measurements[
                index * shots_per_std_part : (index + 1) * shots_per_std_part
            ],
            op_list,
        )
        return [x / shots_per_std_part for x in c_ijab]

    def eval_overlaps_bootstrap(self, all_addr, eval_inputs):
        total_meas = len(self.shuffled_measurements)
        # shots_per_std_part = (total_meas)//n_groups
        if "n_samples" in eval_inputs:
            n_samples = eval_inputs["n_samples"]
        else:
            n_samples = total_meas

        op_list = []
        for n in all_addr:
            index_1_r = n[::-1]  ##n[::-1]
            op_list.append([1.0, int(index_1_r, 2), 0])
        bts_Us, bts_samples = self.bootstrap_sample(n_samples)
        c_ijab = self.density_mat_stabilizer(
            bts_Us, bts_samples, op_list
        )  # density_mat_stabilizer(self.shuffled_Us[index*shots_per_std_part:(index+1)*shots_per_std_part], self.shuffled_measurements[index*shots_per_std_part:(index+1)*shots_per_std_part],op_list)
        return [x / n_samples for x in c_ijab]

    def density_mat_stabilizer(self, shadow_tableau_list, shadow_measurements, op_list):
        # self.post_process_db = {}
        num_qubits = len(shadow_measurements[0][0])
        estimates = [0.0] * len(op_list)
        for unitary_count in range(len(shadow_tableau_list)):
            # print("got to here "+str(unitary_count))
            for measurement_count in range(len(shadow_measurements[unitary_count])):
                data = shadow_measurements[unitary_count][measurement_count]
                data = "".join([str(x) for x in data[::-1]])
                rand_clifford = shadow_tableau_list[unitary_count]
                if (
                    tuple(rand_clifford.stab.flatten()),
                    tuple(rand_clifford.destab.flatten()),
                    data,
                ) in self.post_process_db:
                    ovlp_magnitude = self.post_process_db[
                        (
                            tuple(rand_clifford.stab.flatten()),
                            tuple(rand_clifford.destab.flatten()),
                            data,
                        )
                    ][0]
                    reduced_stab_matrix = self.post_process_db[
                        (
                            tuple(rand_clifford.stab.flatten()),
                            tuple(rand_clifford.destab.flatten()),
                            data,
                        )
                    ][1]
                    meas_out = self.post_process_db[
                        (
                            tuple(rand_clifford.stab.flatten()),
                            tuple(rand_clifford.destab.flatten()),
                            data,
                        )
                    ][2]
                else:
                    data_dummy = data.replace("1", "X")
                    data_pauli = data_dummy.replace("0", "I")

                    stab_data_pauli = quantum_info.Pauli(data_pauli)
                    stab_data = quantum_info.StabilizerState(stab_data_pauli)
                    stab_data = stab_data.evolve(rand_clifford.adjoint())
                    stab_matrix = []
                    # phase_vec=[]
                    for n in stab_data.clifford.to_labels(mode="S"):
                        # phase_vec.append(n[0])
                        row = []
                        for m in n[:]:
                            row.append(m)
                        stab_matrix.append(row[1:] + [1 if row[0] == "+" else -1])
                    reduced_stab_matrix = reduce_to_row_echelon(stab_matrix)

                    ovlp_magnitude = 1 / (
                        np.sqrt(2) ** x_rank_of_reduced(reduced_stab_matrix)
                    )
                    meas_out = stab_data.measure()[0]

                    self.post_process_db[
                        (
                            tuple(rand_clifford.stab.flatten()),
                            tuple(rand_clifford.destab.flatten()),
                            data,
                        )
                    ] = (ovlp_magnitude, reduced_stab_matrix, meas_out)
                for n in range(len(op_list)):
                    ovlp_state1 = (
                        "0" * (num_qubits - len(str(bin(op_list[n][1]))[2:]))
                        + str(bin(op_list[n][1]))[2:]
                    )
                    ovlp_state2 = (
                        "0" * (num_qubits - len(str(bin(op_list[n][2]))[2:]))
                        + str(bin(op_list[n][2]))[2:]
                    )
                    # print(ovlp_state1,ovlp_state2,get_ovlp_phase_basis(meas_out,ovlp_state1,reduced_stab_matrix),np.conj(get_ovlp_phase_basis(meas_out,ovlp_state2,reduced_stab_matrix)),ovlp_magnitude)
                    if ovlp_state1 != ovlp_state2:
                        estimates[n] += (
                            2.0
                            * ovlp_magnitude
                            * ovlp_magnitude
                            * ((2**num_qubits) + 1)
                            * (
                                get_ovlp_phase_basis(
                                    meas_out, ovlp_state1, reduced_stab_matrix
                                )
                                * np.conj(
                                    get_ovlp_phase_basis(
                                        meas_out, ovlp_state2, reduced_stab_matrix
                                    )
                                )
                            )
                        )
                    else:
                        estimates[n] += (
                            2.0
                            * ovlp_magnitude
                            * ovlp_magnitude
                            * ((2**num_qubits) + 1)
                            * (
                                get_ovlp_phase_basis(
                                    meas_out, ovlp_state1, reduced_stab_matrix
                                )
                                * np.conj(
                                    get_ovlp_phase_basis(
                                        meas_out, ovlp_state2, reduced_stab_matrix
                                    )
                                )
                            )
                            - 2
                        )
        return estimates


def shot_measurement_counters(shots: list[QsysShot]) -> tuple[list[Counter], int]:
    """Collect per-register, leak-free outcome counters from measurement shots."""
    if not shots:
        raise ValueError("shots must contain at least one QsysShot")
    shot_arrays = np.einsum(
        "ijk->jik",
        [[measurement[1] for measurement in list(shot.entries)] for shot in shots],
    )
    counters = []
    total_shots = 0
    for shots in shot_arrays:
        valid = [tuple(map(int, shot)) for shot in shots if np.sum(shot == 2) == 0]
        counter = Counter(valid)
        counters.append(counter)
        total_shots += counter.total()
    return counters, total_shots


def process_shadow_shots(shots: list[QsysShot], tableaus: Iterable) -> dict[str, Any]:
    """Filter leakage and pair each shot register with its Clifford tableau."""
    tableaus = list(tableaus)
    outcomes, total_shots = shot_measurement_counters(shots)
    if len(outcomes) != len(tableaus):
        raise ValueError(
            f"Found {len(outcomes)} measured registers for {len(tableaus)} tableaus"
        )
    return {
        "outcomes": outcomes,
        "tableaus": tableaus,
        "total_shots": total_shots,
    }


def make_global_clifford_shadow(
    n_qubits: int,
    tableaus: Iterable,
    outcomes: Iterable,
    *,
    number_of_circuits: int = 0,
    randomize: bool = True,
):
    """Build a shared ``GlobalCliffordShadow`` from tableaus and outcomes."""
    shadow = GlobalCliffordShadow(number_of_circuits, n_qubits)
    shadow.set_cliffords_tableaus(list(tableaus))
    shadow.set_shadow_outcomes(list(outcomes))
    if randomize:
        shadow.randomise_clifford_meas_lists()
    return shadow


def evaluate_shadow_reference_series(
    reference_data: dict,
    shadow,
    spinorb_to_qubit_indices: dict[int, int],
    fci_energy: float,
    *,
    max_reference_determinants: int = 8,
    chi_determinants: int = 10,
    bootstrap_samples: int = 400,
    total_shots: Optional[int] = None,
    ci_tolerance: float = 1e-15,
    state_loader: Optional[Callable] = None,
    randomize_each_reference: bool = False,
) -> dict[str, Any]:
    """Estimate the energy and bootstrap error across reference sizes."""
    if state_loader is None:
        state_loader = from_state_dict
    energies = []
    errors = []
    estimates = []

    for reference_count in range(1, max_reference_determinants + 1):
        if randomize_each_reference:
            shadow.randomise_clifford_meas_lists()
        reference = reference_data["refs"][reference_count - 1]
        phi = to_fermion_state(state_loader(reference["phi"]))
        chi = to_fermion_state(state_loader(reference["chi"]))
        chi_truncated = drop_to_inq_by_count(chi, chi_determinants).normalized()
        phi_permuted = permute_fermion_state(phi.normalized(), spinorb_to_qubit_indices)
        chi_permuted = permute_fermion_state(chi_truncated, spinorb_to_qubit_indices)

        correlation = reference["rphi"] * eval_mixed_estimator(
            shadow.eval_overlaps,
            chi_permuted,
            phi_permuted,
            tol=ci_tolerance,
        )
        energy = reference["ephi"] + correlation
        energies.append(energy)

        eval_inputs = {}
        if total_shots is not None:
            eval_inputs["n_samples"] = total_shots
        bootstrap_values = []
        for _ in range(bootstrap_samples):
            bootstrap_value = eval_mixed_estimator(
                shadow.eval_overlaps_bootstrap,
                chi_permuted,
                phi_permuted,
                tol=ci_tolerance,
                eval_inputs=eval_inputs,
            )
            bootstrap_values.append(bootstrap_value.real)
        error = reference["rphi"] * np.std(bootstrap_values)
        errors.append(error)
        estimates.append(
            {
                "reference_count": reference_count,
                "energy": energy,
                "error": error,
                "error_from_fci": energy - fci_energy,
                "determinants_in_chi": len(chi),
                "determinants_after_truncation": len(chi_truncated),
            }
        )

    return {"energies": energies, "errors": errors, "estimates": estimates}


def analyse_shadow_energy_series(
    shots: list[QsysShot],
    tableaus: Iterable,
    reference_data: dict,
    n_qubits: int,
    spinorb_to_qubit_indices: dict[int, int],
    fci_energy: float,
    *,
    max_reference_determinants: int = 8,
    chi_determinants: int = 10,
    bootstrap_samples: int = 100,
    total_shots: Optional[int] = None,
    ci_tolerance: float = 1e-15,
    randomize: bool = True,
    randomize_each_reference: bool = False,
) -> dict[str, Any]:
    """Process measurement shots and calculate the classical-shadow energies.

    ``shots`` is one batch of measurements. TUPS and SMCS use the same measurement
    processing, shadow initialization, and reference-series estimator, so their
    case-specific functions can delegate to this helper.
    """
    if not shots:
        raise ValueError("shots must contain at least one QsysShot")
    processed = process_shadow_shots(shots, tableaus)
    if processed["total_shots"] == 0:
        raise ValueError("No leak-free outcomes were found in the measurement shots")

    shadow = make_global_clifford_shadow(
        n_qubits,
        processed["tableaus"],
        processed["outcomes"],
        number_of_circuits=len(processed["outcomes"]),
        randomize=randomize,
    )

    energy_results = evaluate_shadow_reference_series(
        reference_data,
        shadow,
        spinorb_to_qubit_indices,
        fci_energy,
        max_reference_determinants=max_reference_determinants,
        chi_determinants=chi_determinants,
        bootstrap_samples=bootstrap_samples,
        total_shots=total_shots,
        ci_tolerance=ci_tolerance,
        randomize_each_reference=randomize_each_reference,
    )
    return {**energy_results, "total_shots": processed["total_shots"]}


def main(
    data_directory=None,
    *,
    max_reference_determinants: int = 6,
    chi_determinants: int = 10,
    bootstrap_samples: int = 50,
    seed: Optional[int] = 123,
) -> dict[str, Any]:
    """Calculate T1/Rx (6e, 6o) tUPS and MCS hardware-shadow energies.

    By default, read the four ``helios_plots_*.json`` files under ``code/data``
    and use the local ``states_smcs_m*_k10_l13.json`` FCI references for both
    ansatzes. Follow their
    ``E = ephi + rphi * mixed_estimator`` calculation, retaining the largest
    ``chi_determinants`` terms of chi. Each bootstrap resample contains the
    full number of leak-free hardware shots; error bars are one standard
    deviation of the real energy estimates.

    Print the estimates in Hartree and return them by system and ansatz.
    ``bootstrap_samples`` controls the number of resamples, and ``seed=None``
    leaves the random generators unseeded.
    """
    from pathlib import Path

    if not 1 <= max_reference_determinants <= 8:
        raise ValueError("max_reference_determinants must be between 1 and 8")
    if chi_determinants < 1:
        raise ValueError("chi_determinants must be positive")
    if bootstrap_samples < 2:
        raise ValueError("bootstrap_samples must be at least 2")

    data_directory = (
        Path(data_directory)
        if data_directory is not None
        else Path(__file__).resolve().parents[1] / "data"
    )
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    results = {}
    for system, geometry in (("t166", "T1"), ("rx66", "Rx")):
        results[system] = {}
        protocol_directory = data_directory / system / "protocol"
        for ansatz in ("tups", "smcs"):
            input_path = protocol_directory / f"helios_cs_{ansatz}_shots.json"
            info = load_json_info(input_path)
            tableaus = info["tableaus"]
            outcomes = info["all_measurements_execute"]
            if not tableaus or len(tableaus) != len(outcomes):
                raise ValueError(
                    f"Unmatched or missing tableaus/outcomes in {input_path}"
                )
            total_shots = sum(counts.total() for counts in outcomes)
            if total_shots == 0:
                raise ValueError(f"No leak-free hardware shots in {input_path}")

            references = []
            for reference_count in range(1, max_reference_determinants + 1):
                # These files contain the FCI-based phi/chi shared by both
                # notebook estimators; their psi/psil states are not used here.
                reference_path = protocol_directory / (
                    f"states_smcs_m{reference_count}_k10_l13.json"
                )
                with reference_path.open(encoding="utf-8") as reference_file:
                    references.append(json.load(reference_file))

            n_qubits = tableaus[0].num_qubits
            # The FCI references are unpermuted. Truncate chi in that ordering,
            # then let the estimator apply the tUPS circuit permutation once.
            orbital_order = (
                [0, 10, 2, 8, 4, 6, 7, 5, 9, 3, 11, 1]
                if ansatz == "tups"
                else range(n_qubits)
            )
            spinorb_to_qubit_indices = {
                spinorb: qubit for qubit, spinorb in enumerate(orbital_order)
            }
            shadow = make_global_clifford_shadow(n_qubits, tableaus, outcomes)
            print(
                f"\n{system} {ansatz}: {total_shots} leak-free hardware shots, "
                f"{bootstrap_samples} bootstrap resamples",
                flush=True,
            )
            energy_results = evaluate_shadow_reference_series(
                {"refs": references},
                shadow,
                spinorb_to_qubit_indices,
                references[0]["efci"],
                max_reference_determinants=max_reference_determinants,
                chi_determinants=chi_determinants,
                bootstrap_samples=bootstrap_samples,
                total_shots=total_shots,
            )
            results[system][ansatz] = {**energy_results, "total_shots": total_shots}
            for estimate in energy_results["estimates"]:
                print(
                    f"m={estimate['reference_count']}: "
                    f"E = {estimate['energy'].real:.10f} "
                    f"+/- {estimate['error']:.10f} Ha; "
                    f"E - E_FCI = {estimate['error_from_fci'].real:.10f} Ha",
                    flush=True,
                )

    return results


if __name__ == "__main__":
    main()
