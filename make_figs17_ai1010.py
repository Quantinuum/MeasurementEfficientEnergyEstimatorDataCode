#!/usr/bin/env python3
"""Make the six-panel Measurement Efficient Energy Estimator (10,10) state-preparation figure."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


HERE = Path(__file__).resolve().parent
DATA_FILES = {
    "Rx": HERE / "data" / "rx1010" / "ai_smcs.json",
    "R0": HERE / "data" / "t11010" / "ai_smcs.json",
}
GEOMETRY_COLORS = {"Rx": "#D55E00", "R0": "#0072B2"}
METHODS = (
    {
        "data_key": "common_tups",
        "label": "Common tUPS",
        "short_label": "Common\ntUPS",
        "n_runs": 1,
        "fill_alpha": 0.28,
        "gap_error_mha": -0.3149460148961225,
        "gap_sd_mha": 0.0,
    },
    {
        "data_key": "per_geometry_tups",
        "label": "Per-geometry tUPS",
        "short_label": "Per-geometry\ntUPS",
        "n_runs": 1,
        "fill_alpha": 0.46,
        "gap_error_mha": -3.2905828161347017,
        "gap_sd_mha": 0.0,
    },
    {
        "data_key": "selected_smcs",
        "label": "Selected sMCS",
        "short_label": "Selected\nsMCS",
        "n_runs": 5,
        "fill_alpha": 1.0,
        "gap_error_mha": 0.19,
        "gap_sd_mha": 0.53,
    },
)
OUTPUT = HERE / "AI_1010_noiselessfinal_main_paper_colors"

INK = "#202124"
GRID = "#E5E7EB"
BASELINE = "#777777"
PAIR = "#72558F"
CHEMICAL_ACCURACY = "#C56A1A"
MONO_FONT = ["DejaVu Sans Mono", "monospace"]


def rgba(hex_color, alpha):
    rgb = tuple(int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5))
    return (*rgb, alpha)


def geometry_legend(rx_color, r0_color):
    return [
        Patch(facecolor=rx_color, edgecolor=rx_color, label=r"$\mathbf{R}_x$ (10,10)"),
        Patch(facecolor=r0_color, edgecolor=r0_color, label=r"$\mathbf{R}_0$ (10,10)"),
    ]


def add_geometry_legend(ax, rx_color, r0_color):
    ax.legend(
        handles=geometry_legend(rx_color, r0_color),
        loc="upper left",
        ncol=2,
        frameon=False,
        fontsize=5.9,
        handlelength=1.4,
        columnspacing=0.9,
        borderaxespad=0.35,
    )


def method_labels(rows):
    return [f"{row['short_label']}\nN={row['n_runs']}" for row in rows]


def residual_text(value):
    if value < 1.0e-3:
        return f"{value:.1e}"
    if value < 0.1:
        return f"{value:.3f}"
    return f"{value:.2f}"


def set_style():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.titleweight": "semibold",
            "axes.linewidth": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.2,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def make_figure(data):
    rows = data["methods"]
    rx_color = data["geometry_colors"]["Rx"]
    r0_color = data["geometry_colors"]["R0"]
    labels = method_labels(rows)

    set_style()
    fig, ax = plt.subplots(3, 2, figsize=(7.2, 7.8))
    fig.subplots_adjust(
        left=0.14,
        right=0.985,
        bottom=0.07,
        top=0.98,
        hspace=0.56,
        wspace=0.52,
    )

    # a) Aggregate ZZPhase-gate count
    a = ax[0, 0]
    y = np.arange(len(rows))
    rx = [row["Rx"]["gates_2q"] for row in rows]
    r0 = [row["R0"]["gates_2q"] for row in rows]
    totals = [x + y0 for x, y0 in zip(rx, r0)]
    a.barh(
        y,
        rx,
        color=[rgba(rx_color, row["fill_alpha"]) for row in rows],
        edgecolor=rx_color,
        linewidth=0.9,
    )
    a.barh(
        y,
        r0,
        left=rx,
        color=[rgba(r0_color, row["fill_alpha"]) for row in rows],
        edgecolor=r0_color,
        linewidth=0.9,
    )
    for yy, rx_value, r0_value, total in zip(y, rx, r0, totals):
        a.text(
            rx_value / 2,
            yy,
            str(rx_value),
            ha="center",
            va="center",
            fontsize=6.4,
            fontfamily=MONO_FONT,
        )
        a.text(
            rx_value + r0_value / 2,
            yy,
            str(r0_value),
            ha="center",
            va="center",
            fontsize=6.4,
            fontfamily=MONO_FONT,
        )
        a.text(
            total + 13,
            yy,
            str(total),
            ha="left",
            va="center",
            fontsize=6.4,
            fontfamily=MONO_FONT,
        )
    a.set_xlim(0, max(totals) * 1.18)
    a.set_ylim(len(rows) - 0.5, -1.05)
    a.set_yticks(y, [row["label"] for row in rows])
    a.set_xlabel(r"Aggregate ZZPhase-gate count, $2G(\psi_l)+G(\chi_{1,k})$")
    a.set_title(
        r"$\bf{a}$  State-preparation gate count", loc="left", pad=7, fontsize=8.2
    )
    a.xaxis.grid(True)
    add_geometry_legend(a, rx_color, r0_color)

    # b) Aggregate two-qubit depth
    b = ax[0, 1]
    height = 0.36
    rx = [row["Rx"]["depth_2q"] for row in rows]
    r0 = [row["R0"]["depth_2q"] for row in rows]
    b.barh(
        y - height / 2,
        rx,
        height=height,
        color=[rgba(rx_color, row["fill_alpha"]) for row in rows],
        edgecolor=rx_color,
        linewidth=0.9,
    )
    b.barh(
        y + height / 2,
        r0,
        height=height,
        color=[rgba(r0_color, row["fill_alpha"]) for row in rows],
        edgecolor=r0_color,
        linewidth=0.9,
    )
    for yy, rx_value, r0_value in zip(y, rx, r0):
        b.text(
            rx_value + 5,
            yy - height / 2,
            str(rx_value),
            va="center",
            fontsize=6.4,
            fontfamily=MONO_FONT,
        )
        b.text(
            r0_value + 5,
            yy + height / 2,
            str(r0_value),
            va="center",
            fontsize=6.4,
            fontfamily=MONO_FONT,
        )
    b.set_xlim(0, max(rx + r0) * 1.20)
    b.set_ylim(len(rows) - 0.5, -1.05)
    b.set_yticks(y, [row["label"] for row in rows])
    b.set_xlabel(r"Aggregate two-qubit depth, $2D(\psi_l)+D(\chi_{1,k})$")
    b.set_title(
        r"$\bf{b}$  State-preparation two-qubit depth", loc="left", pad=7, fontsize=8.2
    )
    b.xaxis.grid(True)
    add_geometry_legend(b, rx_color, r0_color)

    # c) Absolute energy error
    c = ax[1, 0]
    x = np.arange(len(rows))
    width = 0.34
    rx = [row["Rx"]["energy_error_mha"] for row in rows]
    r0 = [row["R0"]["energy_error_mha"] for row in rows]
    rx_sd = [row["Rx"]["energy_error_sd_mha"] for row in rows]
    r0_sd = [row["R0"]["energy_error_sd_mha"] for row in rows]
    c.bar(
        x - width / 2,
        rx,
        width=width,
        yerr=rx_sd,
        color=[rgba(rx_color, row["fill_alpha"]) for row in rows],
        edgecolor=rx_color,
        linewidth=0.9,
        capsize=3,
        ecolor=INK,
        error_kw={"elinewidth": 1.0},
    )
    c.bar(
        x + width / 2,
        r0,
        width=width,
        yerr=r0_sd,
        color=[rgba(r0_color, row["fill_alpha"]) for row in rows],
        edgecolor=r0_color,
        linewidth=0.9,
        capsize=3,
        ecolor=INK,
        error_kw={"elinewidth": 1.0},
    )
    c.axhline(1.6, color=CHEMICAL_ACCURACY, linestyle=(0, (5, 3)), linewidth=1.0)
    for xx, rx_value, r0_value, rx_err, r0_err in zip(x, rx, r0, rx_sd, r0_sd):
        c.text(
            xx - width / 2,
            rx_value + rx_err + 0.16,
            f"{rx_value:.2f}",
            ha="center",
            va="bottom",
            fontsize=6.4,
        )
        c.text(
            xx + width / 2,
            r0_value + r0_err + 0.16,
            f"{r0_value:.2f}",
            ha="center",
            va="bottom",
            fontsize=6.4,
        )
    c.set_xticks(x, labels)
    c.set_ylabel(r"$|E_{\mathrm{ME}}-E_{\mathrm{exact}}|$ (mHa)")
    c.set_ylim(0, max(rx + r0) + 2.4)
    c.set_title(r"$\bf{c}$  Absolute energy error", loc="left", pad=7, fontsize=8.2)
    c.yaxis.grid(True)
    c.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=CHEMICAL_ACCURACY,
                linestyle=(0, (5, 3)),
                linewidth=1.0,
                label="1.6 mHa",
            ),
            *geometry_legend(rx_color, r0_color),
        ],
        loc="upper left",
        ncol=3,
        frameon=False,
        fontsize=5.5,
        handlelength=1.7,
        columnspacing=0.7,
        borderaxespad=0.35,
    )

    # d) Signed gap error
    d = ax[1, 1]
    values = [row["gap_error_mha"] for row in rows]
    errors = [row["gap_sd_mha"] for row in rows]
    d.axvline(0, color=INK, linewidth=1.0)
    d.barh(
        y,
        values,
        xerr=errors,
        color=[rgba(BASELINE, 0.42), rgba(BASELINE, 0.66), rgba(PAIR, 0.96)],
        edgecolor=[BASELINE, BASELINE, PAIR],
        linewidth=0.9,
        capsize=3,
        ecolor=INK,
        error_kw={"elinewidth": 1.0},
    )
    limit = max(abs(value) + error for value, error in zip(values, errors)) * 1.35
    d.set_xlim(-limit, limit)
    d.set_ylim(len(rows) - 0.5, -1.28)
    d.set_yticks(y, [f"{row['label']}\n(N={row['n_runs']})" for row in rows])
    for yy, value, error in zip(y, values, errors):
        offset = error + 0.13
        xpos = value + offset if value >= 0 else value - offset
        d.text(
            xpos,
            yy,
            f"{value:+.2f}",
            ha="left" if value >= 0 else "right",
            va="center",
            fontsize=6.4,
            fontfamily=MONO_FONT,
        )
    d.set_xlabel(r"$\delta(\Delta E_{\mathbf{R}_x-\mathbf{R}_0})$ (mHa)")
    d.set_title(
        r"$\bf{d}$  Signed activation-gap error", loc="left", pad=7, fontsize=8.2
    )
    d.xaxis.grid(True)
    d.legend(
        handles=[
            Patch(
                facecolor=rgba(BASELINE, 0.66),
                edgecolor=BASELINE,
                label="tUPS baselines",
            ),
            Patch(facecolor=rgba(PAIR, 0.96), edgecolor=PAIR, label="selected sMCS"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=2,
        frameon=False,
        fontsize=5.7,
        handlelength=1.25,
        columnspacing=0.75,
        borderaxespad=0.2,
    )

    # e-f) Spin residuals
    spin_panels = [
        (ax[2, 0], "psi_spin_residual", "Prepared-state spin residual", "e", r"\psi_l"),
        (
            ax[2, 1],
            "chi_spin_residual",
            "Residual-state spin residual",
            "f",
            r"\chi_{1,k}",
        ),
    ]
    for axis, key, title, letter, symbol in spin_panels:
        rx = [row["Rx"][key] for row in rows]
        r0 = [row["R0"][key] for row in rows]
        axis.bar(
            x - width / 2,
            rx,
            width=width,
            color=[rgba(rx_color, row["fill_alpha"]) for row in rows],
            edgecolor=rx_color,
            linewidth=0.9,
        )
        axis.bar(
            x + width / 2,
            r0,
            width=width,
            color=[rgba(r0_color, row["fill_alpha"]) for row in rows],
            edgecolor=r0_color,
            linewidth=0.9,
        )
        positive = [value for value in rx + r0 if value > 0]
        axis.set_yscale("log")
        axis.set_ylim(min(positive) * 0.38, max(positive) * 4.8)
        axis.set_xticks(x, labels)
        axis.set_ylabel(
            rf"$\Vert \hat{{S}}^2|{symbol}\rangle-2|{symbol}\rangle\Vert^2$"
        )
        axis.set_title(rf"$\bf{{{letter}}}$  {title}", loc="left", pad=7, fontsize=8.2)
        axis.yaxis.grid(True, which="major")
        for xx, rx_value, r0_value in zip(x, rx, r0):
            axis.text(
                xx - width / 2,
                rx_value * 1.42,
                residual_text(rx_value),
                ha="center",
                va="bottom",
                fontsize=6.2,
            )
            axis.text(
                xx + width / 2,
                r0_value * 1.42,
                residual_text(r0_value),
                ha="center",
                va="bottom",
                fontsize=6.2,
            )
        add_geometry_legend(axis, rx_color, r0_color)

    fig.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.08)
    fig.savefig(
        OUTPUT.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.08
    )
    plt.close(fig)


def load_data():
    geometry_data = {}
    for geometry, path in DATA_FILES.items():
        with path.open(encoding="utf-8") as f:
            geometry_data[geometry] = json.load(f)
        if geometry_data[geometry]["geometry"] != geometry:
            raise ValueError(f"Unexpected geometry data in {path}.")

    rows = []
    for method in METHODS:
        data_key = method["data_key"]
        row = {key: value for key, value in method.items() if key != "data_key"}
        row["Rx"] = geometry_data["Rx"][data_key]
        row["R0"] = geometry_data["R0"][data_key]
        rows.append(row)
    return {"methods": rows, "geometry_colors": GEOMETRY_COLORS}


def main():
    make_figure(load_data())
    print(f"Wrote {OUTPUT.with_suffix('.pdf')}")
    print(f"Wrote {OUTPUT.with_suffix('.png')}")


if __name__ == "__main__":
    main()
