from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

FIGURE_DIR = Path(__file__).resolve().parent
CODE_DIR = FIGURE_DIR.parent
SHARED_DATA_DIR = CODE_DIR / "data"
OUTPUT_DIR = FIGURE_DIR
M_VALUES = tuple(range(1, 9))
sys.path.insert(0, str(CODE_DIR))

CACHE_DIR = Path(tempfile.gettempdir()) / "fig2-matplotlib"
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR / "xdg"))

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot
from matplotlib.axes import Axes
from matplotlib.ticker import FormatStrFormatter

from utils import (
    from_operator_dict,
    from_state_dict,
    mixed_estimator,
    mixed_estimator_pre,
    top_selected_subspace,
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def case_rows(case_dir: Path) -> list[dict[str, float]]:
    efci = float(load_json(case_dir / "chemistry.json")["efci"])
    encoded = load_json(case_dir / "jordan_wigner.json")
    hamiltonian = from_operator_dict(encoded["hamiltonian"])
    target = from_state_dict(encoded["state"])
    rows: list[dict[str, float]] = []
    for size in M_VALUES:
        _, reference = top_selected_subspace(hamiltonian, target, size)
        energy, residual_norm, residual = mixed_estimator_pre(reference, hamiltonian)
        mixed_energy, h, s = mixed_estimator(
            energy, residual_norm, residual, target, reference
        )
        if s < 0.0:
            h, s = -h, -s
        rows.append(
            {
                "m": float(size),
                "ephi_minus_efci_mha": 1000.0 * (energy - efci),
                "rphi_mha": 1000.0 * residual_norm,
                "rphi_h_over_s_mha": 1000.0 * residual_norm * h / s,
                "h": h,
                "s": s,
                "h_over_s": h / s,
                "eme_minus_efci_mha": 1000.0 * (mixed_energy - efci),
            }
        )
    return rows


def plot_case(
    axes: Axes,
    rows: list[dict[str, float]],
    key: str,
    color: str,
    label: str,
    marker: str,
) -> None:
    axes.plot(
        [row["m"] for row in rows],
        [row[key] for row in rows],
        color=color,
        marker=marker,
        markersize=4.0,
        markeredgecolor="white",
        markeredgewidth=0.4,
        linewidth=1.2,
        label=label,
    )


def make_fig2() -> Path:
    summary = {
        "rx_rows": case_rows(SHARED_DATA_DIR / "rx66"),
        "t1_rows": case_rows(SHARED_DATA_DIR / "t166"),
    }
    pyplot.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.linewidth": 0.6,
            "axes.titlesize": 7.4,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.6,
            "ytick.labelsize": 6.6,
            "legend.fontsize": 6.3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
        }
    )
    figure, axes = pyplot.subplots(2, 3, figsize=(7.15, 4.6), constrained_layout=True)
    panels = (
        (
            "ephi_minus_efci_mha",
            r"$E_\phi-E_{\mathrm{FCI}}$ (mHa)",
            "a  Reference energy",
            None,
        ),
        ("rphi_mha", r"$r_\phi$ (mHa)", "b  Residual norm", None),
        ("rphi_h_over_s_mha", r"$r_\phi h/s$ (mHa)", "c  Correlation energy", None),
        ("h", r"$h$", "d  Residual state overlap", "%.2f"),
        ("s", r"$s$", "e  Reference state overlap", "%.3f"),
        ("h_over_s", r"$h/s$", "f  Overlap ratio", "%.2f"),
    )
    for panel_axes, (key, ylabel, title, number_format) in zip(
        axes.ravel(), panels, strict=True
    ):
        plot_case(panel_axes, summary["rx_rows"], key, "#4477AA", "Rx (6,6)", "o")
        plot_case(panel_axes, summary["t1_rows"], key, "#CC3311", "T1 (6,6)", "s")
        panel_axes.set(xlabel=r"$m$ (reference size)", ylabel=ylabel, title=title)
        panel_axes.title.set(
            fontweight="bold", position=(0.0, 1.0), horizontalalignment="left"
        )
        panel_axes.set_xticks(M_VALUES)
        panel_axes.grid(axis="both", color="0.88", linewidth=0.45)
        panel_axes.tick_params(axis="both", labelsize=6.6, length=2.2, width=0.55)
        if number_format is not None:
            panel_axes.yaxis.set_major_formatter(FormatStrFormatter(number_format))
        for spine in panel_axes.spines.values():
            spine.set_linewidth(0.6)
    axes.ravel()[0].legend(
        frameon=False, loc="upper right", ncol=2, handlelength=1.7, columnspacing=0.9
    )

    output = OUTPUT_DIR / "fig2_exact_mixed_estimator.pdf"
    figure.savefig(output, bbox_inches="tight")
    pyplot.close(figure)
    return output


if __name__ == "__main__":
    print(make_fig2())
