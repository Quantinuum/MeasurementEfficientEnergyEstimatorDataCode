"""Create the classical computational chemistry plots from the notebooks.

Each function below contains the plotting logic from the notebook with the
matching name.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import string


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

DISTANCES = (3.8, 3.6, 3.4, 3.2, 3.0, 2.8, 2.6, 2.4)
IMAGE_NUMBERS = tuple(range(22))

mpl.rcParams["pdf.fonttype"] = 42


def _add_subplot_labels(axes) -> None:
    """Add the letter labels used by the notebooks to each subplot."""
    for axis, label in zip(axes.flat, string.ascii_lowercase):
        axis.text(
            -0.05,
            1.02,
            f"{label})",
            transform=axis.transAxes,
            fontsize=14,
            fontweight="bold",
            va="top",
            ha="right",
        )


def plot_activation_energies():
    """Create and save the activation-energy plot."""


    fig = plt.figure(figsize=(9, 9))
    gs = fig.add_gridspec(2, 1, hspace=0.3, wspace=0.3)
    ax = gs.subplots()

    y1 = (0.45, 0.45, 0.46, 0.57, 0.57, 0.57, 0.66, 0.86)
    y2 = (0.89, 0.89, 0.89, 0.95, 0.95, 0.95, 0.97, 0.92)

    ax[0].set(
        xlabel=r"PS-O$_2$ distance (Å)",
        ylabel="Activation energy (eV)",
        title=r"$\omega$B97X",
    )
    ax[1].set(
        xlabel=r"PS-O$_2$ distance (Å)",
        ylabel="Activation energy (eV)",
        title="B3LYP",
    )

    ax[0].invert_xaxis()
    ax[1].invert_xaxis()
    ax[0].scatter(DISTANCES, y1)
    ax[1].scatter(DISTANCES, y2)
    _add_subplot_labels(ax)
    fig.savefig("activation_energies.pdf")
    return fig


def plot_diabatic_coupl():
    """Create and save the diabatic-couplings plot."""


    fig = plt.figure(figsize=(9, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    ax = gs.subplots()

    y1 = (0.24, 0.48, 0.89, 3.79, 7.34, 7.82, 3.79, 3.79)
    y2 = (0.40, 0.48, 0.89, 2.18, 7.34, 7.90, 6.78, 11.78)
    y3 = (4.19, 6.13, 19.60, 125.18, 125.10, 124.29, 175.67, 98.16)

    ax[0, 0].set(
        xlabel=r"PS-O$_2$ distance (Å)",
        ylabel=r"Diabatic couplings (cm$^{-1}$)",
        title=r"$\omega$B97X",
    )
    ax[0, 1].set(
        xlabel=r"PS-O$_2$ distance (Å)",
        ylabel=r"Diabatic couplings (cm$^{-1}$)",
        title="B3LYP",
    )
    ax[1, 0].set(
        xlabel=r"PS-O$_2$ distance (Å)",
        ylabel=r"Diabatic couplings (cm$^{-1}$)",
        title="PBE",
    )

    ax[0, 0].invert_xaxis()
    ax[0, 1].invert_xaxis()
    ax[1, 0].invert_xaxis()
    ax[0, 0].scatter(DISTANCES, y1)
    ax[0, 1].scatter(DISTANCES, y2)
    ax[1, 0].scatter(DISTANCES, y3)
    ax[1, 1].remove()
    _add_subplot_labels(ax)
    fig.savefig("diabatic_couplings.pdf")
    return fig


def plot_rates():
    """Create and save the rates plot."""

    fig = plt.figure(figsize=(9, 9))
    gs = fig.add_gridspec(2, 1, hspace=0.3, wspace=0.3)
    ax = gs.subplots()

    y1 = (3.86, 14.3, 42.3, 11.7, 42.2, 47.6, 0.0329, 0.00022)
    y2 = (5.03e-07, 6.77e-07, 2.03e-06, 1.38e-06, 1.53e-05, 1.76e-05, 4.93e-06, 1.30e-04)

    ax[0].set(
        xlabel=r"PS-O$_2$ distance (Å)",
        ylabel=r"Rates (sec$^{-1}$)",
        title=r"$\omega$B97X",
    )
    ax[1].set(
        xlabel=r"PS-O$_2$ distance (Å)",
        ylabel=r"Rates (sec$^{-1}$)",
        title="B3LYP",
    )

    ax[0].invert_xaxis()
    ax[1].invert_xaxis()
    ax[0].scatter(DISTANCES, y1)
    ax[1].scatter(DISTANCES, y2)
    _add_subplot_labels(ax)
    fig.savefig("rates.pdf")
    return fig


def plot_spin_images():
    """Create and save the Hirshfeld spin-density plot."""


    fig = plt.figure(figsize=(9, 9))
    gs = fig.add_gridspec(2, 1, hspace=0.3, wspace=0.3)
    ax = gs.subplots()

    y1 = (
        0.872,
        0.8762,
        0.8793,
        0.8812,
        0.8813,
        0.8797,
        0.8777,
        0.8774,
        0.884,
        0.9546,
        1.5728,
        1.5794,
        1.5864,
        1.5939,
        1.6019,
        1.6103,
        1.619,
        1.6281,
        1.6374,
        1.647,
        1.6566,
        1.6662,
    )
    y2 = (
        0.3152,
        0.338,
        0.3634,
        0.392,
        0.4253,
        0.4655,
        0.5262,
        1.0556,
        1.1795,
        1.2854,
        1.3588,
        1.41,
        1.4479,
        1.4776,
        1.5019,
        1.5228,
        1.5412,
        1.5579,
        1.5734,
        1.588,
        1.6018,
        1.6148,
    )

    ax[0].set(
        xlabel="Image number",
        ylabel="Hirshfeld spin density",
        title=r"$\omega$B97X",
    )
    ax[1].set(
        xlabel="Image number",
        ylabel="Hirshfeld spin density",
        title="B3LYP",
    )

    ax[0].scatter(IMAGE_NUMBERS, y1)
    ax[1].scatter(IMAGE_NUMBERS, y2)
    _add_subplot_labels(ax)
    fig.savefig("spin_images.pdf")
    return fig


def plot_splittings():
    """Create and save the singlet-triplet splitting plot."""

    fig = plt.figure(figsize=(9, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    ax = gs.subplots()

    y1 = (
        2.288,
        2.296,
        2.303,
        2.308,
        2.309,
        2.305,
        2.294,
        2.277,
        2.255,
        2.220,
        1.626,
        1.438,
        1.252,
        1.070,
        0.892,
        0.721,
        0.557,
        0.402,
        0.255,
        0.117,
        -0.011,
        -0.129,
    )
    y2 = (
        2.042,
        2.061,
        2.080,
        2.097,
        2.112,
        2.124,
        2.129,
        1.998,
        1.886,
        1.746,
        1.589,
        1.423,
        1.252,
        1.081,
        0.912,
        0.747,
        0.588,
        0.436,
        0.292,
        0.156,
        0.029,
        -0.088,
    )
    y3 = (
        1.711,
        1.724,
        1.736,
        1.746,
        1.754,
        1.758,
        1.758,
        1.737,
        1.676,
        1.586,
        1.469,
        1.336,
        1.194,
        1.047,
        0.899,
        0.751,
        0.607,
        0.467,
        0.332,
        0.205,
        0.085,
        -0.027,
    )

    ax[0, 0].set(
        xlabel="Image number",
        ylabel="S-T splittings (eV)",
        title=r"$\omega$B97X",
    )
    ax[0, 1].set(
        xlabel="Image number",
        ylabel="S-T splittings (eV)",
        title="B3LYP",
    )
    ax[1, 0].set(
        xlabel="Image number",
        ylabel="S-T splittings (eV)",
        title="PBE",
    )

    ax[0, 0].scatter(IMAGE_NUMBERS, y1)
    ax[0, 1].scatter(IMAGE_NUMBERS, y2)
    ax[1, 0].scatter(IMAGE_NUMBERS, y3)
    ax[1, 1].remove()
    _add_subplot_labels(ax)
    fig.savefig("T_S_splittings.pdf")
    return fig


def plot_triplet_energies():
    """Create and save the triplet-energy plot."""


    fig = plt.figure(figsize=(9, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    ax = gs.subplots()

    y1 = (
        0.000,
        0.011,
        0.077,
        0.184,
        0.319,
        0.470,
        0.627,
        0.783,
        0.928,
        1.049,
        0.598,
        0.534,
        0.454,
        0.357,
        0.249,
        0.131,
        0.010,
        -0.109,
        -0.217,
        -0.310,
        -0.495,
        -0.660,
    )
    y2 = (
        0.000,
        0.033,
        0.120,
        0.249,
        0.407,
        0.584,
        0.769,
        0.821,
        0.888,
        0.917,
        0.915,
        0.885,
        0.834,
        0.763,
        0.677,
        0.580,
        0.478,
        0.377,
        0.286,
        0.210,
        0.067,
        -0.052,
    )
    y3 = (
        0.000,
        0.037,
        0.125,
        0.253,
        0.410,
        0.584,
        0.767,
        0.934,
        1.057,
        1.143,
        1.189,
        1.202,
        1.190,
        1.156,
        1.105,
        1.041,
        0.970,
        0.899,
        0.836,
        0.787,
        0.714,
        0.667,
    )

    ax[0, 0].set(
        xlabel="Image number",
        ylabel="Triplet energy (eV)",
        title=r"$\omega$B97X",
    )
    ax[0, 1].set(
        xlabel="Image number",
        ylabel="Triplet energy (eV)",
        title="B3LYP",
    )
    ax[1, 0].set(
        xlabel="Image number",
        ylabel="Triplet energy (eV)",
        title="PBE",
    )

    ax[0, 0].scatter(IMAGE_NUMBERS, y1)
    ax[0, 1].scatter(IMAGE_NUMBERS, y2)
    ax[1, 0].scatter(IMAGE_NUMBERS, y3)
    ax[1, 1].remove()
    _add_subplot_labels(ax)
    fig.savefig("triplet_energies.pdf")
    return fig


if __name__ == "__main__":
    plot_activation_energies()
    plot_diabatic_coupl()
    plot_rates()
    plot_spin_images()
    plot_splittings()
    plot_triplet_energies()