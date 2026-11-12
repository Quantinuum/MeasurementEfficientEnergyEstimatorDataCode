"""Reproduce the Measurement Efficient Energy Estimator (10,10) hardware overlap and energy tables."""

import argparse
import csv
import hashlib
import json
import math
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def count_shots(shots):
    counts = dict(total=len(shots), aborted=0, leaked=0, accepted=0, zero=0)
    n_qubits = None
    missing = object()
    for shot in shots:
        values = next((value for tag, value in shot if tag == "res"), missing)
        if values is missing:
            # A parity failure can end the shot before the data measurement.
            counts["aborted"] += 1
            continue
        if not isinstance(values, list) or not values:
            raise ValueError("Expected a nonempty data-qubit measurement array")
        if n_qubits is None:
            n_qubits = len(values)
        if len(values) != n_qubits or any(v not in (0, 1, 2) for v in values):
            raise ValueError("Inconsistent register length or invalid ternary outcome")
        if 2 in values:
            counts["leaked"] += 1
        else:
            counts["accepted"] += 1
            if all(v == 0 for v in values):
                counts["zero"] += 1
    return counts, n_qubits


def overlap_stats(counts):
    accepted = counts["accepted"]
    if accepted == 0:
        raise ValueError("Cannot estimate an overlap without accepted shots")
    probability = counts["zero"] / accepted
    standard_error = math.sqrt(probability * (1 - probability) / accepted)
    return probability, standard_error


def energy_estimate(s_counts, h_counts, reference):
    p_s, se_s = overlap_stats(s_counts)
    p_h, se_h = overlap_stats(h_counts)
    if p_s == 0 or p_h == 0:
        raise ValueError("The propagated ratio error requires positive overlaps")

    ratio = math.sqrt(p_h / p_s)
    ratio_se = 0.5 * ratio * math.sqrt((se_h / p_h) ** 2 + (se_s / p_s) ** 2)
    # Squared overlaps do not determine the sign: use the recorded classical sign.
    energy = (
        reference["reference_energy_hartree"]
        + reference["overlap_ratio_sign"] * ratio * reference["residual_norm_hartree"]
    )
    energy_se = abs(reference["residual_norm_hartree"]) * ratio_se
    error_mha = 1000 * (energy - reference["casci_energy_hartree"])
    return {
        "p_reference": p_s,
        "p_reference_standard_error": se_s,
        "p_residual": p_h,
        "p_residual_standard_error": se_h,
        "overlap_ratio": ratio,
        "overlap_ratio_standard_error": ratio_se,
        "energy_hartree": energy,
        "energy_standard_error_hartree": energy_se,
        "casci_energy_hartree": reference["casci_energy_hartree"],
        "signed_error_mha": error_mha,
        "absolute_error_mha": abs(error_mha),
    }


def analyse(manifest_path, references_path):
    manifest = json.loads(manifest_path.read_text())
    references = json.loads(references_path.read_text())
    shot_statistics = []
    groups = {}
    seen = set()
    for dataset in manifest["datasets"]:
        key = (dataset["method"], dataset["geometry"], dataset["overlap"])
        if key in seen or key[2] not in ("s", "h"):
            raise ValueError(f"Duplicate or unrecognized circuit: {key}")
        seen.add(key)
        path = manifest_path.parent / dataset["raw_file"]
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != dataset["sha256"]:
            raise ValueError(f"Input checksum does not match: {path}")
        counts, n_qubits = count_shots(json.loads(payload))
        if counts["total"] != dataset["raw_record_count"]:
            raise ValueError(f"Returned-shot count does not match the manifest: {path}")
        probability, probability_se = overlap_stats(counts)
        shot_statistics.append(
            {
                "method": key[0],
                "geometry": key[1],
                "overlap": key[2],
                "submitted_shots": dataset["submitted_shots"],
                "counts": counts,
                "data_qubits": n_qubits,
                "discarded_shots": counts["aborted"] + counts["leaked"],
                "acceptance_fraction": counts["accepted"] / counts["total"],
                "all_zero_probability": probability,
                "probability_standard_error": probability_se,
            }
        )
        groups.setdefault(key[:2], {})[key[2]] = counts

    if len(seen) != manifest["dataset_count"]:
        raise ValueError("Dataset count does not match the manifest")
    if (
        sum(r["counts"]["total"] for r in shot_statistics)
        != manifest["shot_record_count"]
    ):
        raise ValueError("Total returned shots do not match the manifest")

    energies = []
    for (method, geometry), pair in groups.items():
        if set(pair) != {"s", "h"}:
            raise ValueError(f"Both overlaps are required for {method}, {geometry}")
        energies.append(
            {
                "method": method,
                "geometry": geometry,
                "truncation": references[geometry]["truncation"],
                **energy_estimate(pair["s"], pair["h"], references[geometry]),
            }
        )

    differences = []
    for method in dict.fromkeys(row["method"] for row in energies):
        pair = {row["geometry"]: row for row in energies if row["method"] == method}
        rx, r0 = pair["Rx"], pair["R0"]
        difference = rx["energy_hartree"] - r0["energy_hartree"]
        difference_se = math.hypot(
            rx["energy_standard_error_hartree"], r0["energy_standard_error_hartree"]
        )
        casci_difference = rx["casci_energy_hartree"] - r0["casci_energy_hartree"]
        error_mha = 1000 * (difference - casci_difference)
        differences.append(
            {
                "method": method,
                "energy_difference_hartree": difference,
                "energy_difference_standard_error_hartree": difference_se,
                "casci_energy_difference_hartree": casci_difference,
                "signed_error_mha": error_mha,
                "absolute_error_mha": abs(error_mha),
            }
        )
    return {
        "backend": manifest["backend"],
        "energy_difference_convention": "E_Rx - E_R0",
        "uncertainty": "One standard error; independent s/h and geometry measurements",
        "shot_statistics": shot_statistics,
        "energies": energies,
        "energy_differences": differences,
    }, manifest


def rounded(value, places):
    return format(
        Decimal(str(value)).quantize(
            Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP
        ),
        f".{places}f",
    )


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_tables(result, manifest, out):
    counts = {
        (r["method"], r["geometry"], r["overlap"]): r["counts"]
        for r in result["shot_statistics"]
    }
    overlap_rows, energy_rows = [], []
    for row in result["energies"]:
        method, geometry = row["method"], row["geometry"]
        settings = manifest["methods"][method]
        s, h = counts[(method, geometry, "s")], counts[(method, geometry, "h")]
        overlap_rows.append(
            {
                "geometry": geometry,
                "backend": result["backend"],
                "error_detection": settings["error_detection"],
                "total_qubits": settings["total_qubit_allocations"],
                "accepted_s_percent": rounded(
                    Decimal(s["accepted"]) * 100 / s["total"], 1
                ),
                "accepted_h_percent": rounded(
                    Decimal(h["accepted"]) * 100 / h["total"], 1
                ),
                "p_reference": rounded(row["p_reference"], 4),
                "p_reference_standard_error": rounded(
                    row["p_reference_standard_error"], 4
                ),
                "p_residual": rounded(row["p_residual"], 4),
                "p_residual_standard_error": rounded(
                    row["p_residual_standard_error"], 4
                ),
                "overlap_ratio": rounded(row["overlap_ratio"], 4),
                "overlap_ratio_standard_error": rounded(
                    row["overlap_ratio_standard_error"], 4
                ),
            }
        )
        t = row["truncation"]
        energy_rows.append(
            {
                "active_space": manifest["active_space"],
                "geometry": geometry,
                "state_preparation": manifest["state_preparation"],
                "truncation": f"({t['m']},{t['k']},{t['l']})",
                "error_detection": settings["error_detection"],
                "energy_hartree": rounded(row["energy_hartree"], 6),
                "energy_standard_error_hartree": rounded(
                    row["energy_standard_error_hartree"], 6
                ),
                "discarded_h": h["aborted"] + h["leaked"],
                "discarded_s": s["aborted"] + s["leaked"],
            }
        )
    geometries = list(dict.fromkeys(r["geometry"] for r in overlap_rows))
    overlap_rows.sort(key=lambda row: geometries.index(row["geometry"]))
    write_csv(out / "table1_hardware.csv", overlap_rows)
    write_csv(out / "tableS8_hardware.csv", energy_rows)
    write_csv(out / "energy_differences.csv", result["energy_differences"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "manifest.json")
    parser.add_argument("--references", type=Path, default=ROOT / "references.json")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    result, manifest = analyse(args.manifest, args.references)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    write_tables(result, manifest, args.out)
    total = sum(row["counts"]["total"] for row in result["shot_statistics"])
    print(f"{len(result['shot_statistics'])} circuits, {total:,} returned shots")
    print(f"Results: {args.out}")


if __name__ == "__main__":
    main()
