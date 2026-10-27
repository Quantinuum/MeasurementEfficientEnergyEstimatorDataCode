from pathlib import Path

import numpy
from numpy.typing import NDArray
from zixy.fermion import RealTermSum
from zixy.fermion.operator.general import RealTermSum as GeneralTermSum


def load_hamiltonian(path: str | Path) -> RealTermSum:
    with numpy.load(path, allow_pickle=False) as data:
        one_body: NDArray[numpy.float64] = data["one_body"]
        two_body: NDArray[numpy.float64] = data["two_body"]
        terms: list[tuple[list[tuple[int, bool]], float]] = [([], float(data["constant"]))]
    for p, q in numpy.ndindex(one_body.shape):
        for spin in range(2):
            terms.append(([(2 * p + spin, True), (2 * q + spin, False)], float(one_body[p, q])))
    for p, q, r, s in numpy.ndindex(two_body.shape):
        for spin in range(2):
            for other_spin in range(2):
                operators = [(2 * p + spin, True), (2 * r + other_spin, True),
                             (2 * s + other_spin, False), (2 * q + spin, False)]
                terms.append((operators, float(two_body[p, q, r, s]) / 2))
    return GeneralTermSum.from_iterable(terms, 2 * len(one_body), max_string_len=4).to_normal_ordered()
