"""Compute-uncompute energy post-processing for measurement shots."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from hugr.qsystem.result import QsysResult, QsysShot

from utils import (
    absolute_square_summary_text,
    get_zero_counts,
    load_json,
    load_shottable,
    mixed_estimator_from_absolute_square_counts,
    outcomer,
    save_json_out,
    save_text_out,
)


def save_energies(
    s_results: Sequence[QsysShot],
    h_results: Sequence[QsysShot],
    filename: str,
    *,
    output_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Calculate and save energy estimates using metadata in ``filename``.

    ``filename`` can be either a compute-uncompute overlap JSON or a protocol
    ``states_*.json`` reference file. Outcome CSVs, a result JSON, and a text
    summary are written to ``output_directory`` (or beside the input by default),
    using its stem as the output name prefix.
    """
    overlap_path = Path(filename)
    output_path = (
        overlap_path.parent if output_directory is None else Path(output_directory)
    )
    output_path.mkdir(parents=True, exist_ok=True)
    overlap = load_json(overlap_path)
    measurement = overlap.get("measurement", {})
    protocol = measurement.get("protocol_name", "compute_uncompute")
    if protocol != "compute_uncompute":
        raise ValueError("Only the compute_uncompute protocol is supported.")
    if not s_results or not h_results:
        raise ValueError("Both S and H results must contain at least one shot.")

    outcome_s = outcomer(QsysResult(list(s_results)))
    outcome_h = outcomer(QsysResult(list(h_results)))
    name_s = f"{overlap_path.stem}_outcome_s_abs2.csv"
    name_h = f"{overlap_path.stem}_outcome_h_abs2.csv"
    outcome_s.to_csv(output_path / name_s, index=False)
    outcome_h.to_csv(output_path / name_h, index=False)

    register = str(measurement.get("data_register", "q"))
    s_zero_counts = get_zero_counts(outcome_s, register)
    h_zero_counts = get_zero_counts(outcome_h, register)
    reference = overlap.get("reference", overlap)
    if "ratio_sign" in reference:
        ratio_sign = float(reference["ratio_sign"])
    else:
        ratio_sign = 1.0 if float(reference["h"]) / float(reference["s"]) >= 0 else -1.0
    value, se, bias, value_bc, h_abs2, s_abs2, se_h_abs2, se_s_abs2, ratio = (
        mixed_estimator_from_absolute_square_counts(
            float(reference["ephi"]),
            float(reference["rphi"]),
            h_zero_counts,
            s_zero_counts,
            ratio_sign,
        )
    )

    summary_text = absolute_square_summary_text(
        efci=float(reference["efci"]),
        shift=float(reference["ephi"]),
        scale=float(reference["rphi"]),
        h_exact=float(reference["h"]),
        s_exact=float(reference["s"]),
        h_zero_counts=h_zero_counts,
        s_zero_counts=s_zero_counts,
        ratio_sign=ratio_sign,
    )
    summary_name = f"{overlap_path.stem}_summary.txt"
    result_name = f"{overlap_path.stem}_results.json"
    result = {
        "input_file": str(overlap_path),
        "m": int(overlap["m"]),
        "k": int(overlap["k"]),
        "l": int(overlap["l"]) if "l" in overlap else "N/A",
        "protocol": str(protocol),
        "shots": {"s_abs2": len(s_results), "h_abs2": len(h_results)},
        "estimates": {
            "s_abs2": float(s_abs2),
            "h_abs2": float(h_abs2),
            "se_s_abs2": float(se_s_abs2),
            "se_h_abs2": float(se_h_abs2),
            "ratio": float(ratio),
            "mixed_estimator": {
                "value": float(value),
                "se": float(se),
                "bias": float(bias),
                "value_bc": float(value_bc),
            },
        },
        "outputs_csv": {"s_abs2": name_s, "h_abs2": name_h},
        "summary_txt": summary_name,
        "efci": float(reference["efci"]),
    }

    save_json_out(output_path / result_name, result)
    save_text_out(output_path / summary_name, summary_text)
    return result


def main(
    state_prep: str,
    mkl: tuple[int, int, int],
    geometry: str,
    *,
    output_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Load the H/S shot tables for a case and calculate its energy estimate.

    ``geometry`` is the short geometry name (for example ``"rx"`` or
    ``"t1"``); its protocol data are read from ``data/<geometry>66/protocol``.
    The result JSON, outcome CSVs, and summary are saved in ``output_directory``
    when provided, or beside the state reference file otherwise.
    """
    if len(mkl) != 3:
        raise ValueError("mkl must contain (m, k, l).")
    m, k, l = (int(value) for value in mkl)
    state_prep = state_prep.strip().lower()
    geometry = geometry.strip().lower()
    protocol_directory = (
        Path(__file__).resolve().parent / "data" / f"{geometry}66" / "protocol"
    )
    suffix = f"m{m}_k{k}_l{l}"
    s_path = protocol_directory / f"helios_cu_{state_prep}_s_{suffix}_shots.json"
    h_path = protocol_directory / f"helios_cu_{state_prep}_h_{suffix}_shots.json"
    reference_path = protocol_directory / f"states_{state_prep}_{suffix}.json"

    s_results = load_shottable(s_path)
    h_results = load_shottable(h_path)
    result = save_energies(
        s_results,
        h_results,
        str(reference_path),
        output_directory=output_directory,
    )
    estimate = result["estimates"]["mixed_estimator"]
    print(f"Energy: {estimate['value']:.6f} ± {estimate['se']:.6f} Ha")
    return result


if __name__ == "__main__":
    output_root = (
        Path(__file__).resolve().parent / "sandbox" / "compute_uncompute"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    cases = (
        ("tups", (1, 10, 2)),
        ("tups", (3, 10, 2)),
        ("smcs", (1, 10, 13)),
        ("smcs", (3, 10, 13)),
    )
    records: list[tuple[str, str, tuple[int, int, int], float, float]] = []
    for geometry in ("rx", "t1"):
        for state_prep, mkl in cases:
            result = main(
                state_prep,
                mkl,
                geometry,
                output_directory=output_root / geometry,
            )
            estimate = result["estimates"]["mixed_estimator"]
            records.append(
                (
                    geometry,
                    state_prep,
                    mkl,
                    float(estimate["value"]),
                    float(estimate["se"]),
                )
            )

    report_lines = [
        "Compute-uncompute energies and 1σ error bars (Hartree)",
        "Geometry  State prep  (m, k, l)     Energy          Error bar",
    ]
    for geometry, state_prep, mkl, energy, error_bar in records:
        report_lines.append(
            f"{geometry.upper():8}  {state_prep.upper():10}  {str(mkl):12}  "
            f"{energy: .9f}  {error_bar:.9f}"
        )
    report_path = output_root / "energies_and_error_bars.txt"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"Wrote {report_path}")
