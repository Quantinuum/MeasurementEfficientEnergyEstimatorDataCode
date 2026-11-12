from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy
from hugr.qsystem.result import QsysResult, QsysShot


def filter_leaked_shots(
    qsys_result: QsysResult,
    leakage_value: int = 2,
) -> tuple[QsysResult, dict[str, int | float]]:
    """Remove shots containing the leakage marker.

    Args:
        qsys_result: Shot results to filter.
        leakage_value: Marker used for leaked measurements.

    Returns:
        Filtered result and counts for total, kept, discarded, and discard rate.
    """

    def has_leakage(value: Any) -> bool:
        if isinstance(value, str):
            return str(leakage_value) in value
        if isinstance(value, bool):
            return False
        if isinstance(value, int):
            return value == leakage_value
        try:
            return any(has_leakage(item) for item in value)
        except TypeError:
            return False

    kept = [
        shot
        for shot in qsys_result.results
        if not any(has_leakage(value) for _tag, value in shot.entries)
    ]
    n_total = len(qsys_result.results)
    n_kept = len(kept)
    n_discarded = n_total - n_kept
    return QsysResult(QsysShot(shot.entries) for shot in kept), {
        "n_total": n_total,
        "n_kept": n_kept,
        "n_discarded": n_discarded,
        "discard_rate": 0.0 if n_total == 0 else n_discarded / n_total,
    }


def bootstrap_qsys_results(
    qsys_result: QsysResult,
    n_bootstrap: int,
    seed: int = 0,
    n_shots: int | None = None,
) -> Iterator[QsysResult]:
    """Resample shots with replacement.

    Args:
        qsys_result: Shot results to resample.
        n_bootstrap: Number of bootstrap result objects.
        seed: Random seed.
        n_shots: Number of shots per sample. Defaults to the input shot count.

    Yields:
        Bootstrap samples as independent result objects.
    """
    shots = qsys_result.results
    if len(shots) == 0:
        raise ValueError("qsys_result must contain at least one shot")
    n_out = len(shots) if n_shots is None else n_shots
    if n_out <= 0:
        raise ValueError("n_shots must be positive")

    rng = numpy.random.default_rng(seed)
    for _sample in range(n_bootstrap):
        yield QsysResult(
            QsysShot(shots[int(idx)].entries)
            for idx in rng.integers(0, len(shots), size=n_out)
        )


def save_qsys_result_json(qsys_result: QsysResult, path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(
            {"results": [shot.entries for shot in qsys_result.results]}, indent=2
        )
        + "\n"
    )


def load_qsys_result_json(path: str | Path) -> QsysResult:
    data = json.loads(Path(path).read_text())
    return QsysResult(
        QsysShot((tag, value) for tag, value in shot) for shot in data["results"]
    )
