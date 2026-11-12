from __future__ import annotations

import json
import os
import string
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy
from numpy.typing import NDArray

FIGURE_DIR = Path(__file__).resolve().parent
CODE_DIR = FIGURE_DIR
PROJECT_DIR = CODE_DIR.parent
DATA_DIR = CODE_DIR / "data"
RX_CASE_DIR = DATA_DIR / "rx66"
RX_DATA_DIR = RX_CASE_DIR / "stderr"
T1_DATA_DIR = DATA_DIR / "t166" / "stderr"
PAULI_AVERAGING_DATA = DATA_DIR / "pauli_averaging.json"
OUTPUT_PDF = FIGURE_DIR / "fig_stderrs.pdf"
sys.path.insert(0, str(CODE_DIR))

CACHE_DIR = Path(tempfile.gettempdir()) / "fig6-matplotlib"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR / "xdg"))

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot
from matplotlib.axes import Axes
from matplotlib.ticker import FormatStrFormatter

PROTOCOLS = ("FCU", "CU", "DST", "SST", "RHT", "RIST", "CS")
PROTOCOL_LABELS = {
    "FCU": "FCU",
    "CU": "CU",
    "DST": "DST",
    "SST": "SST",
    "RHT": "RHT",
    "RIST": "RIST",
    "CS": "CS",
}
COLOURS = {
    "pumpkin": "#E69F00",
    "aqua": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "navy": "#0072B2",
    "orange": "#D55E00",
    "pink": "#CC79A7",
    "black": "#000000",
}
PROTOCOL_COLORS = {
    "FCU": COLOURS["navy"],
    "CU": COLOURS["pumpkin"],
    "DST": COLOURS["green"],
    "SST": COLOURS["pink"],
    "RHT": COLOURS["orange"],
    "RIST": COLOURS["aqua"],
    "CS": COLOURS["black"],
}
CASE_COLORS = {"Rx": COLOURS["pumpkin"], "T1": COLOURS["navy"]}
EPS_TARGET_MHA = 1.6
PANEL_SHOTS = 10_000
SHOT_M_VALUES = tuple(range(1, 9))
SHOT_K = 10
SHOT_L = 13
SHOT_CURVES = ("FCU", "CU", "DST/SST", "RHT/RIST", "CS")
SHOT_LABELS = {
    "FCU": "FCU",
    "CU": "CU",
    "DST/SST": "DST/SST",
    "RHT/RIST": "RHT/RIST",
    "CS": "CS",
}
SHOT_COLORS = {
    "FCU": PROTOCOL_COLORS["FCU"],
    "CU": PROTOCOL_COLORS["CU"],
    "DST/SST": PROTOCOL_COLORS["SST"],
    "RHT/RIST": PROTOCOL_COLORS["RHT"],
    "CS": PROTOCOL_COLORS["CS"],
}
SHOT_STYLES = {
    "FCU": "-",
    "CU": "--",
    "DST/SST": "-",
    "RHT/RIST": "-",
    "CS": "-",
}

from utils import (
    drop_to_top,
    from_operator_dict,
    from_state_dict,
    mixed_estimator,
    mixed_estimator_pre,
    top_selected_subspace,
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as fobj:
        return [json.loads(line) for line in fobj if line.strip()]


def load_case(data_dir: Path) -> dict[str, Any]:
    rows = load_jsonl(data_dir / "protocol_samples.jsonl")
    rows += load_jsonl(data_dir / "shadow_samples.jsonl")
    return {
        "rows": rows,
        "reference": load_json(data_dir / "reference.json"),
    }


def energies_by_protocol(
    rows: list[dict[str, Any]],
    reference_energy: float,
    protocols: tuple[str, ...],
) -> dict[str, NDArray[numpy.float64]]:
    return {
        protocol: numpy.asarray(
            [
                1000.0 * (float(row["energy"]) - reference_energy)
                for row in rows
                if row["protocol"] == protocol and numpy.isfinite(float(row["energy"]))
            ],
            dtype=float,
        )
        for protocol in protocols
    }


def protocol_stderr(
    case_data: dict[str, Any], protocols: tuple[str, ...]
) -> dict[str, float]:
    return {
        protocol: 1000.0
        * float(
            numpy.std(
                [
                    float(row["energy"])
                    for row in case_data["rows"]
                    if row["protocol"] == protocol
                ],
                ddof=1,
            )
        )
        for protocol in protocols
    }


def plot_ridgelines(
    ax: Axes,
    case_name: str,
    case_data: dict[str, Any],
    protocols: tuple[str, ...],
) -> None:
    reference = case_data["reference"]
    reference_energy = float(reference["energy"])
    efci = float(reference["efci"])
    energies = energies_by_protocol(case_data["rows"], efci, protocols)
    finite = numpy.concatenate(
        [values for values in energies.values() if len(values) > 0]
    )
    statevector_error = 1000.0 * (reference_energy - efci)
    x_values = numpy.asarray([*finite, 0.0, statevector_error], dtype=float)
    lo = float(numpy.percentile(x_values, 0.4))
    hi = float(numpy.percentile(x_values, 99.6))
    margin = max(0.6, 0.10 * (hi - lo))
    bins = numpy.linspace(lo - margin, hi + margin, 76)

    for index, protocol in enumerate(protocols):
        values = energies[protocol]
        if len(values) == 0:
            continue
        density, edges = numpy.histogram(values, bins=bins, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        height = density / max(float(numpy.max(density)), 1e-15) * 0.70
        color = PROTOCOL_COLORS[protocol]
        ax.fill_between(
            centers, index, index + height, color=color, alpha=0.20, linewidth=0.0
        )
        ax.plot(centers, index + height, color=color, linewidth=1.3)
        ax.vlines(
            float(numpy.mean(values)), index, index + 0.74, color=color, linewidth=1.2
        )

    ax.axvline(0.0, color="0.35", linestyle=":", linewidth=1.0, label="CASCI")
    ax.axvline(
        statevector_error,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="(1,10,13) statevector",
    )
    ax.set_yticks(numpy.arange(len(protocols), dtype=float))
    ax.set_yticklabels([PROTOCOL_LABELS[protocol] for protocol in protocols])
    ax.set_ylim(-0.25, len(protocols) - 0.05)
    ax.set_xlim(float(bins[0]), float(bins[-1]))
    ax.set_xlabel("Energy error relative to CASCI (mHa)")
    ax.set_title(f"{case_name}: bootstrap energy distributions")
    ax.grid(axis="x", color="0.90", linewidth=0.5)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.tick_params(direction="out", length=3, width=0.8)


def plot_protocol_stderr(
    ax: Axes,
    rx_stats: dict[str, float],
    t1_stats: dict[str, float],
    rx_analytical: dict[str, float],
    t1_analytical: dict[str, float],
    protocols: tuple[str, ...],
) -> None:
    x = numpy.arange(len(protocols), dtype=float)
    width = 0.36
    rx_values = [rx_stats[protocol] for protocol in protocols]
    t1_values = [t1_stats[protocol] for protocol in protocols]
    rx_analytical_values = [rx_analytical[protocol] for protocol in protocols]
    t1_analytical_values = [t1_analytical[protocol] for protocol in protocols]
    ax.bar(
        x - 0.5 * width,
        rx_values,
        width,
        color=CASE_COLORS["Rx"],
        edgecolor="black",
        linewidth=0.4,
        label=r"$\mathbf{R}_x$",
    )
    ax.bar(
        x + 0.5 * width,
        t1_values,
        width,
        color=CASE_COLORS["T1"],
        edgecolor="black",
        linewidth=0.4,
        label=r"$\mathbf{R}_0$",
    )
    ax.axhline(1.6, color="0.25", linestyle="--", linewidth=1.0, label="1.6 mHa")
    for xpos, height, value in zip(
        x - 0.5 * width, rx_values, rx_analytical_values, strict=True
    ):
        ax.text(
            float(xpos),
            height + 0.08,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )
    for xpos, height, value in zip(
        x + 0.5 * width, t1_values, t1_analytical_values, strict=True
    ):
        ax.text(
            float(xpos),
            height + 0.08,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([PROTOCOL_LABELS[protocol] for protocol in protocols])
    ax.set_ylabel("Standard error (mHa)")
    ax.set_title(r"Standard error for $10^4$ shots")
    ax.set_ylim(0.0, 1.22 * max([*rx_values, *t1_values, 1.6]))
    ax.grid(axis="y", color="0.90", linewidth=0.5)
    ax.legend(framealpha=0.1, ncol=3, loc="upper left")
    ax.tick_params(direction="out", length=3, width=0.8)


def statevector_inputs_by_m(case_dir: Path) -> list[dict[str, float]]:
    jordan_wigner = load_json(case_dir / "jordan_wigner.json")
    hamiltonian = from_operator_dict(jordan_wigner["hamiltonian"])
    psi = from_state_dict(jordan_wigner["state"])
    psil = drop_to_top(psi, SHOT_L)
    rows = []
    for m in SHOT_M_VALUES:
        ephi, phi = top_selected_subspace(hamiltonian, psi, m)
        ephi, rphi, chi = mixed_estimator_pre(phi, hamiltonian)
        chik = drop_to_top(chi, SHOT_K)
        _, h, s = mixed_estimator(ephi, rphi, chik, psil, phi)
        if s < 0.0:
            h = -h
            s = -s
        rows.append(
            {
                "m": float(m),
                "ephi": float(ephi),
                "rphi": float(rphi),
                "h": float(h),
                "s": float(s),
            }
        )
    return rows


def shots_required_by_m(
    rows: list[dict[str, float]], n_qubits: int
) -> dict[str, NDArray[numpy.float64]]:
    epsilon = EPS_TARGET_MHA * 1e-3
    dimension = float(2**n_qubits)
    out = {name: [] for name in SHOT_CURVES}
    for row in rows:
        rphi = abs(float(row["rphi"]))
        h = float(row["h"])
        s = float(row["s"])
        p = h * h
        q = s * s
        out["FCU"].append(
            (rphi * rphi / (4.0 * epsilon * epsilon)) * ((p + q) / (q * q))
        )
        out["CU"].append(
            (
                rphi
                / (2.0 * epsilon)
                * (
                    numpy.sqrt(max(0.0, 1.0 - p)) / numpy.sqrt(q)
                    + numpy.sqrt(p) * numpy.sqrt(max(0.0, 1.0 - q)) / q
                )
            )
            ** 2
        )
        out["DST/SST"].append(
            (
                rphi
                / (2.0 * epsilon)
                * (
                    numpy.sqrt(max(0.0, 1.0 - p * p)) / numpy.sqrt(p * q)
                    + numpy.sqrt(p) * numpy.sqrt(max(0.0, 1.0 - q * q)) / (q**1.5)
                )
            )
            ** 2
        )
        out["RHT/RIST"].append(
            (
                rphi
                / epsilon
                * (
                    numpy.sqrt(max(0.0, 1.0 - h * h)) / abs(s)
                    + abs(h) * numpy.sqrt(max(0.0, 1.0 - s * s)) / (s * s)
                )
            )
            ** 2
        )
        out["CS"].append(
            3.0
            * rphi
            * rphi
            * (dimension + 1.0)
            / (dimension + 2.0)
            * (1.0 / (s * s) + h * h / (s**4))
            / (epsilon * epsilon)
        )
    return {name: numpy.asarray(values, dtype=float) for name, values in out.items()}


def analytical_protocol_stderr(case_dir: Path) -> dict[str, float]:
    curves = shots_required_by_m(statevector_inputs_by_m(case_dir), 12)
    grouped = {
        name: EPS_TARGET_MHA * float(numpy.sqrt(values[0] / PANEL_SHOTS))
        for name, values in curves.items()
    }
    return {
        "FCU": grouped["FCU"],
        "CU": grouped["CU"],
        "DST": grouped["DST/SST"],
        "SST": grouped["DST/SST"],
        "RHT": grouped["RHT/RIST"],
        "RIST": grouped["RHT/RIST"],
        "CS": grouped["CS"],
    }


def plot_shot_requirements(
    ax: Axes,
    rx_pauli_shots: float,
    shot_curves: tuple[str, ...],
) -> None:
    rows = statevector_inputs_by_m(RX_CASE_DIR)
    curves = shots_required_by_m(rows, 12)
    x = numpy.asarray([row["m"] for row in rows], dtype=float)
    for curve_name in shot_curves:
        ax.plot(
            x,
            curves[curve_name],
            color=SHOT_COLORS[curve_name],
            linestyle=SHOT_STYLES[curve_name],
            marker="o",
            markersize=4.0,
            linewidth=1.5,
            alpha=0.95,
            label=SHOT_LABELS[curve_name],
        )

    ax.axhline(rx_pauli_shots, color="0.10", linestyle="--", linewidth=1.2, label="PA")
    ax.set_xlim(0.75, 8.25)
    finite = numpy.concatenate(
        [values[numpy.isfinite(values)] for values in curves.values()]
    )
    finite = numpy.concatenate([finite, numpy.asarray([rx_pauli_shots], dtype=float)])
    ax.set_ylim(100.0, 1.65 * float(numpy.nanmax(finite)))
    ax.set_yscale("log")
    ax.set_yticks([1e2, 1e3, 1e4, 1e5, 1e6])
    ax.set_xticks(SHOT_M_VALUES)
    ax.set_xlabel(r"$m$ in $(m,10,13)$")
    ax.set_ylabel("Required shots")
    ax.set_title(r"Shots for 1.6-mHa precision, $\mathbf{R}_x$")
    ax.grid(axis="both", which="major", color="0.90", linewidth=0.5)
    ax.legend(
        framealpha=0.1,
        ncol=2,
        loc="upper right",
        bbox_to_anchor=(0.99, 0.95),
        fontsize=8,
        handlelength=1.5,
        columnspacing=0.8,
    )
    ax.tick_params(direction="out", length=3, width=0.8)


def make_fig6(
    output_pdf: Path = OUTPUT_PDF,
    protocols: tuple[str, ...] = PROTOCOLS,
    shot_curves: tuple[str, ...] = SHOT_CURVES,
) -> Path:
    rx_case = load_case(RX_DATA_DIR)
    t1_case = load_case(T1_DATA_DIR)
    rx_pauli_shots = float(load_json(PAULI_AVERAGING_DATA)["shots"]["rx66"])
    rx_stats = protocol_stderr(rx_case, protocols)
    t1_stats = protocol_stderr(t1_case, protocols)
    rx_analytical = analytical_protocol_stderr(RX_CASE_DIR)
    t1_analytical = analytical_protocol_stderr(T1_DATA_DIR.parent)

    pyplot.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 10,
            "axes.linewidth": 0.8,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
        }
    )

    figure = pyplot.figure(figsize=(9, 9))
    grid = figure.add_gridspec(
        2, 2, hspace=0.30, wspace=0.30, height_ratios=[1.14, 1.0]
    )
    ax_rx = figure.add_subplot(grid[0, 0])
    ax_t1 = figure.add_subplot(grid[0, 1])
    ax_stderr = figure.add_subplot(grid[1, 0])
    ax_bound = figure.add_subplot(grid[1, 1])

    plot_ridgelines(ax_rx, r"$\mathbf{R}_x$ (6,6)", rx_case, protocols)
    plot_ridgelines(ax_t1, r"$\mathbf{R}_0$ (6,6)", t1_case, protocols)
    plot_protocol_stderr(
        ax_stderr, rx_stats, t1_stats, rx_analytical, t1_analytical, protocols
    )
    plot_shot_requirements(ax_bound, rx_pauli_shots, shot_curves)

    for axes, label in zip(
        (ax_rx, ax_t1, ax_stderr, ax_bound), string.ascii_lowercase[:4], strict=True
    ):
        axes.text(
            -0.05,
            1.04,
            f"{label})",
            transform=axes.transAxes,
            fontsize=14,
            fontweight="bold",
            va="bottom",
            ha="right",
        )

    handles, labels = ax_rx.get_legend_handles_labels()
    casci_legend = ax_t1.legend(
        handles[:1],
        labels[:1],
        loc="upper right",
        bbox_to_anchor=(0.99, 0.99),
        fontsize=8,
        frameon=False,
        handlelength=1.8,
    )
    ax_t1.add_artist(casci_legend)
    ax_t1.legend(
        handles[1:],
        labels[1:],
        loc="center right",
        bbox_to_anchor=(0.99, 0.105),
        fontsize=8,
        frameon=False,
        handlelength=1.8,
    )
    figure.savefig(output_pdf, bbox_inches="tight")
    pyplot.close(figure)
    return output_pdf


if __name__ == "__main__":
    print(make_fig6())
