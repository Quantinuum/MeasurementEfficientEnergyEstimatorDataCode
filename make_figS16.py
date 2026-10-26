from __future__ import annotations

import os
import string
import sys
import tempfile
from pathlib import Path

FIGURE_DIR = Path(__file__).resolve().parent
CODE_DIR = FIGURE_DIR.parent
PROJECT_DIR = CODE_DIR.parent
DATA_DIR = CODE_DIR / "data"
OUTPUT = FIGURE_DIR
sys.path.insert(0, str(CODE_DIR))

CACHE_DIR = Path(tempfile.gettempdir()) / "figS16-matplotlib"
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR / "xdg"))

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from utils import (
    drop_to_top,
    from_operator_dict,
    from_state_dict,
    mixed_estimator,
    mixed_estimator_pre,
    top_selected_subspace,
)


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

M_COLOURS = {
    1: COLOURS["navy"],
    3: COLOURS["pumpkin"],
    5: COLOURS["green"],
    7: COLOURS["pink"],
    9: COLOURS["aqua"],
}

K_VALUES = tuple(range(5, 31))
L_VALUES = tuple(range(5, 31))
CHEMICAL_ACCURACY = 1.6e-3


def load_case(case_dir: Path) -> tuple[object, object, float]:
    import json

    chemistry = json.loads((case_dir / "chemistry.json").read_text())
    encoded = json.loads((case_dir / "jordan_wigner.json").read_text())
    hamiltonian = from_operator_dict(encoded["hamiltonian"])
    target = from_state_dict(encoded["state"])
    return hamiltonian, target, float(chemistry["efci"])


def scan_case(
    case_dir: Path, m_values: tuple[int, ...]
) -> dict[int, list[tuple[int, int, float]]]:
    hamiltonian, target, efci = load_case(case_dir)
    targets = {size: drop_to_top(target, size) for size in L_VALUES}
    accepted: dict[int, list[tuple[int, int, float]]] = {}

    for m in m_values:
        _, reference = top_selected_subspace(hamiltonian, target, m)
        energy, residual_norm, residual = mixed_estimator_pre(reference, hamiltonian)
        residuals = {size: drop_to_top(residual, size) for size in K_VALUES}
        points: list[tuple[int, int, float]] = []
        for k, truncated_residual in residuals.items():
            for l, truncated_target in targets.items():
                estimate, _, _ = mixed_estimator(
                    energy,
                    residual_norm,
                    truncated_residual,
                    truncated_target,
                    reference,
                )
                error = abs(estimate - efci)
                if error <= CHEMICAL_ACCURACY:
                    points.append((k, l, error))
        accepted[m] = points
    return accepted


def plot_case(axes: Axes, accepted: dict[int, list[tuple[int, int, float]]]) -> None:
    axes.scatter(
        [k for k in K_VALUES for _ in L_VALUES],
        [l for _ in K_VALUES for l in L_VALUES],
        color="0.82",
        s=7,
        alpha=0.25,
        linewidths=0,
    )

    for m, points in accepted.items():
        if not points:
            continue
        colour = M_COLOURS[m]
        axes.scatter(
            [point[0] for point in points],
            [point[1] for point in points],
            color=colour,
            s=9,
            alpha=0.18,
            linewidths=0,
        )
        frontier_by_k: dict[int, tuple[int, float]] = {}
        for k, l, error in points:
            if k not in frontier_by_k or (l, error) < frontier_by_k[k]:
                frontier_by_k[k] = (l, error)
        frontier = sorted((k, l) for k, (l, _) in frontier_by_k.items())
        axes.plot(
            [point[0] for point in frontier],
            [point[1] for point in frontier],
            color=colour,
            marker="o",
            markersize=4,
            linewidth=1.5,
        )
        best = min(
            points,
            key=lambda point: (point[0] + point[1], point[2], point[0], point[1]),
        )
        axes.scatter(
            [best[0]],
            [best[1]],
            facecolors="white",
            edgecolors=colour,
            s=70,
            linewidths=1.5,
            zorder=4,
        )

    axes.set_xlim(4.5, 30.5)
    axes.set_ylim(4.5, 30.5)
    axes.set_xticks((5, 10, 15, 20, 25, 30))
    axes.set_yticks((5, 10, 15, 20, 25, 30))
    axes.set(xlabel=r"Residual determinants, $k$", ylabel=r"Target determinants, $l$")
    axes.tick_params(direction="out", length=3, width=0.8)


def make_figS16() -> Path:
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
        }
    )

    cases = (
        ("rx66", (1, 3, 5, 7, 9), r"$\mathbf{R}_x$, (6,6)"),
        ("rx88", (1, 3, 5, 7, 9), r"$\mathbf{R}_x$, (8,8)"),
        ("rx1010", (1, 3, 5, 7, 9), r"$\mathbf{R}_x$, (10,10)"),
        ("t166", (1, 3, 5), r"$\mathbf{R}_0$, (6,6)"),
        ("t188", (1, 3, 5), r"$\mathbf{R}_0$, (8,8)"),
        ("t11010", (1, 3, 5), r"$\mathbf{R}_0$, (10,10)"),
    )

    figure = pyplot.figure(figsize=(12, 8))
    grid = figure.add_gridspec(2, 3, hspace=0.34, wspace=0.30)
    axes = grid.subplots()
    for panel_axes, (directory, m_values, title) in zip(
        axes.ravel(), cases, strict=True
    ):
        plot_case(panel_axes, scan_case(DATA_DIR / directory, m_values))
        panel_axes.set_title(title)

    handles = [
        Line2D(
            [0], [0], color=M_COLOURS[m], marker="o", linewidth=1.5, label=rf"$m={m}$"
        )
        for m in M_COLOURS
    ]
    axes[0, 2].legend(handles=handles, loc="lower right", framealpha=0.1)

    for panel_axes, label in zip(axes.ravel(), string.ascii_lowercase[:6], strict=True):
        panel_axes.text(
            -0.05,
            1.04,
            f"{label})",
            transform=panel_axes.transAxes,
            fontsize=14,
            fontweight="bold",
            va="bottom",
            ha="right",
        )

    figure.savefig(OUTPUT, bbox_inches="tight")
    pyplot.close(figure)
    return OUTPUT


if __name__ == "__main__":
    print(make_figS16())
