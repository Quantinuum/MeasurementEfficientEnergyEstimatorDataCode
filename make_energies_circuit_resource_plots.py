import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib as mpl
import json
import pathlib as pl
import string
import numpy as np

# Nat Comm defaults
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

mpl.rcParams["pdf.fonttype"] = 42


def read_stateprep_circuit_resources(
    stateprep: str,
) -> dict[str, dict[str, dict[str, dict[str, int]]]]:
    r"""Read the tUPS and sMCS stateprep circuit resources for different truncations.

    Args:
        stateprep: "tups" or "smcs".

    Returns:
        circ_resources: dictionary of keys --- cas, geo, truncation then
            "gates1q", "gates2q", "depth", "depth2q", "ZZPhase_gates", "ZZPhase_depth"
    """
    home_dir = f"{pl.Path(__file__).parent.parent}"
    with open(f"{home_dir}/code/data/circuit_resources/{stateprep}.json", "r") as f:
        data = json.load(f)
    return data


def read_stateprep_statevector_energies(
    stateprep: str,
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    r"""Read the tUPS and sMCS stateprep statevector energies for different truncations.

    Args:
        stateprep: "tups" or "smcs".

    Returns:
        energies: dictionary of keys --- cas, geo, truncation then
            "sv", "casci_overlap"
    """
    home_dir = f"{pl.Path(__file__).parent.parent}"
    with open(f"{home_dir}/code/data/energies/{stateprep}_sv.json", "r") as f:
        data = json.load(f)
    return data


def read_energies(
    method: str,
) -> dict[str, dict[str, dict[str, dict[str, dict[str, float | list[float]]]]]]:
    r"""Read the classical shadows and mixed estimator energies.

    Args:
        method: "cs" or "cu".

    Returns:
        energies: dictionary of keys --- cas, geo, stateprep, device finally "energies"/"errors".
    """
    home_dir = f"{pl.Path(__file__).parent.parent}"
    fname = "classical_shadows" if method == "cs" else "compute_uncompute"
    with open(f"{home_dir}/code/data/energies/{fname}.json", "r") as f:
        data = json.load(f)
    return data


def read_casci_data() -> dict[str, dict[str, dict[str, dict[str, int | float]]]]:
    r"""Read the CASCI state data.

    Returns:
        energies: dictionary of keys --- cas, geo, resource_type (circ_resources/energies).
            The only circ resource is number of determinants. The only energy is the SV energy.
    """
    home_dir = f"{pl.Path(__file__).parent.parent}"
    with open(f"{home_dir}/code/data/energies/casci.json", "r") as f:
        data = json.load(f)
    return data


MC_CIRC = read_stateprep_circuit_resources("smcs")
TUPS_CIRC = read_stateprep_circuit_resources("tups")
MC_ENERGIES = read_stateprep_statevector_energies("smcs")
TUPS_ENERGIES = read_stateprep_statevector_energies("tups")
CASCI = read_casci_data()

CS_ENERGIES = read_energies("cs")
CU_ENERGIES = read_energies("cu")


def plot_cas66_cu_cu_cs_comparison() -> None:
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.25, wspace=0.25)
    ax = gs.subplots(sharex=True, sharey=True)

    rx_tups_em_cs_ens = CS_ENERGIES["66"]["Rx"]["tups"]["emulator"]["energies"][:8]
    rx_tups_em_cs_errs = CS_ENERGIES["66"]["Rx"]["tups"]["emulator"]["errors"][:8]
    r0_tups_em_cs_ens = CS_ENERGIES["66"]["R0"]["tups"]["emulator"]["energies"][:8]
    r0_tups_em_cs_errs = CS_ENERGIES["66"]["R0"]["tups"]["emulator"]["errors"][:8]

    rx_smcs_em_cs_ens = CS_ENERGIES["66"]["Rx"]["smcs"]["emulator"]["energies"][:8]
    rx_smcs_em_cs_errs = CS_ENERGIES["66"]["Rx"]["smcs"]["emulator"]["errors"][:8]
    r0_smcs_em_cs_ens = CS_ENERGIES["66"]["R0"]["smcs"]["emulator"]["energies"][:8]
    r0_smcs_em_cs_errs = CS_ENERGIES["66"]["R0"]["smcs"]["emulator"]["errors"][:8]

    rx_tups_hard_cs_ens = CS_ENERGIES["66"]["Rx"]["tups"]["hardware"]["energies"][:8]
    rx_tups_hard_cs_errs = CS_ENERGIES["66"]["Rx"]["tups"]["hardware"]["errors"][:8]
    r0_tups_hard_cs_ens = CS_ENERGIES["66"]["R0"]["tups"]["hardware"]["energies"][:8]
    r0_tups_hard_cs_errs = CS_ENERGIES["66"]["R0"]["tups"]["hardware"]["errors"][:8]

    rx_smcs_hard_cs_ens = CS_ENERGIES["66"]["Rx"]["smcs"]["hardware"]["energies"][:8]
    rx_smcs_hard_cs_errs = CS_ENERGIES["66"]["Rx"]["smcs"]["hardware"]["errors"][:8]
    r0_smcs_hard_cs_ens = CS_ENERGIES["66"]["R0"]["smcs"]["hardware"]["energies"][:8]
    r0_smcs_hard_cs_errs = CS_ENERGIES["66"]["R0"]["smcs"]["hardware"]["errors"][:8]

    rx_tups_em_cu_ens = CU_ENERGIES["66"]["Rx"]["tups"]["emulator"]["energies"]
    rx_tups_em_cu_errs = CU_ENERGIES["66"]["Rx"]["tups"]["emulator"]["errors"]
    r0_tups_em_cu_ens = CU_ENERGIES["66"]["R0"]["tups"]["emulator"]["energies"]
    r0_tups_em_cu_errs = CU_ENERGIES["66"]["R0"]["tups"]["emulator"]["errors"]

    rx_smcs_em_cu_ens = CU_ENERGIES["66"]["Rx"]["smcs"]["emulator"]["energies"]
    rx_smcs_em_cu_errs = CU_ENERGIES["66"]["Rx"]["smcs"]["emulator"]["errors"]
    r0_smcs_em_cu_ens = CU_ENERGIES["66"]["R0"]["smcs"]["emulator"]["energies"]
    r0_smcs_em_cu_errs = CU_ENERGIES["66"]["R0"]["smcs"]["emulator"]["errors"]

    rx_tups_hard_cu_ens = CU_ENERGIES["66"]["Rx"]["tups"]["hardware"]["energies"]
    rx_tups_hard_cu_errs = CU_ENERGIES["66"]["Rx"]["tups"]["hardware"]["errors"]
    r0_tups_hard_cu_ens = CU_ENERGIES["66"]["R0"]["tups"]["hardware"]["energies"]
    r0_tups_hard_cu_errs = CU_ENERGIES["66"]["R0"]["tups"]["hardware"]["errors"]

    rx_smcs_hard_cu_ens = CU_ENERGIES["66"]["Rx"]["smcs"]["hardware"]["energies"]
    rx_smcs_hard_cu_errs = CU_ENERGIES["66"]["Rx"]["smcs"]["hardware"]["errors"]
    r0_smcs_hard_cu_ens = CU_ENERGIES["66"]["R0"]["smcs"]["hardware"]["energies"]
    r0_smcs_hard_cu_errs = CU_ENERGIES["66"]["R0"]["smcs"]["hardware"]["errors"]

    rx_casci_en, r0_casci_en = (
        CASCI["66"]["Rx"]["energies"]["sv"],
        CASCI["66"]["R0"]["energies"]["sv"],
    )

    rx_tups_sv = CU_ENERGIES["66"]["Rx"]["tups"]["statevector"]["energies"]
    r0_tups_sv = CU_ENERGIES["66"]["R0"]["tups"]["statevector"]["energies"]
    rx_smcs_sv = CU_ENERGIES["66"]["Rx"]["smcs"]["statevector"]["energies"]
    r0_smcs_sv = CU_ENERGIES["66"]["R0"]["smcs"]["statevector"]["energies"]

    m, hard_m = list(range(1, 9)), [1, 3]
    ax[0, 0].errorbar(
        m,
        rx_tups_em_cu_ens,
        rx_tups_em_cu_errs,
        marker="o",
        markeredgecolor=COLOURS["orange"],
        markerfacecolor="none",
        color=COLOURS["orange"],
        capsize=3,
        linestyle="",
    )
    ax[0, 0].errorbar(
        m,
        r0_tups_em_cu_ens,
        r0_tups_em_cu_errs,
        marker="o",
        markeredgecolor=COLOURS["navy"],
        markerfacecolor="none",
        color=COLOURS["navy"],
        capsize=3,
        linestyle="",
    )
    ax[1, 0].errorbar(
        m,
        rx_smcs_em_cu_ens,
        rx_smcs_em_cu_errs,
        marker="o",
        markeredgecolor=COLOURS["orange"],
        markerfacecolor="none",
        color=COLOURS["orange"],
        capsize=3,
        linestyle="",
    )
    ax[1, 0].errorbar(
        m,
        r0_smcs_em_cu_ens,
        r0_smcs_em_cu_errs,
        marker="o",
        markeredgecolor=COLOURS["navy"],
        markerfacecolor="none",
        color=COLOURS["navy"],
        capsize=3,
        linestyle="",
    )

    ax[0, 0].errorbar(
        hard_m,
        rx_tups_hard_cu_ens,
        rx_tups_hard_cu_errs,
        marker="s",
        markeredgecolor=COLOURS["pumpkin"],
        markerfacecolor="none",
        color=COLOURS["pumpkin"],
        capsize=3,
        linestyle="",
    )
    ax[0, 0].errorbar(
        hard_m,
        r0_tups_hard_cu_ens,
        r0_tups_hard_cu_errs,
        marker="s",
        markeredgecolor=COLOURS["aqua"],
        markerfacecolor="none",
        color=COLOURS["aqua"],
        capsize=3,
        linestyle="",
    )
    ax[1, 0].errorbar(
        hard_m,
        rx_smcs_hard_cu_ens,
        rx_smcs_hard_cu_errs,
        marker="s",
        markeredgecolor=COLOURS["pumpkin"],
        markerfacecolor="none",
        color=COLOURS["pumpkin"],
        capsize=3,
        linestyle="",
    )
    ax[1, 0].errorbar(
        hard_m,
        r0_smcs_hard_cu_ens,
        r0_smcs_hard_cu_errs,
        marker="s",
        markeredgecolor=COLOURS["aqua"],
        markerfacecolor="none",
        color=COLOURS["aqua"],
        capsize=3,
        linestyle="",
    )

    ax[0, 1].errorbar(
        m,
        rx_tups_em_cs_ens,
        rx_tups_em_cs_errs,
        marker="o",
        markeredgecolor=COLOURS["orange"],
        markerfacecolor=COLOURS["orange"],
        color=COLOURS["orange"],
        capsize=3,
        linestyle="",
    )
    ax[0, 1].errorbar(
        m,
        r0_tups_em_cs_ens,
        r0_tups_em_cs_errs,
        marker="o",
        markeredgecolor=COLOURS["navy"],
        markerfacecolor=COLOURS["navy"],
        color=COLOURS["navy"],
        capsize=3,
        linestyle="",
    )

    ax[0, 2].errorbar(
        m,
        rx_tups_hard_cs_ens,
        rx_tups_hard_cs_errs,
        marker="s",
        markeredgecolor=COLOURS["pumpkin"],
        markerfacecolor=COLOURS["pumpkin"],
        color=COLOURS["pumpkin"],
        capsize=3,
        linestyle="",
    )
    ax[0, 2].errorbar(
        m,
        r0_tups_hard_cs_ens,
        r0_tups_hard_cs_errs,
        marker="s",
        markeredgecolor=COLOURS["aqua"],
        markerfacecolor=COLOURS["aqua"],
        color=COLOURS["aqua"],
        capsize=3,
        linestyle="",
    )

    ax[1, 1].errorbar(
        m,
        rx_smcs_em_cs_ens,
        rx_smcs_em_cs_errs,
        marker="o",
        markeredgecolor=COLOURS["orange"],
        markerfacecolor=COLOURS["orange"],
        color=COLOURS["orange"],
        capsize=3,
        linestyle="",
    )
    ax[1, 1].errorbar(
        m,
        r0_smcs_em_cs_ens,
        r0_smcs_em_cs_errs,
        marker="o",
        markeredgecolor=COLOURS["navy"],
        markerfacecolor=COLOURS["navy"],
        color=COLOURS["navy"],
        capsize=3,
        linestyle="",
    )

    ax[1, 2].errorbar(
        m,
        rx_smcs_hard_cs_ens,
        rx_smcs_hard_cs_errs,
        marker="s",
        markeredgecolor=COLOURS["pumpkin"],
        markerfacecolor=COLOURS["pumpkin"],
        color=COLOURS["pumpkin"],
        capsize=3,
        linestyle="",
    )
    ax[1, 2].errorbar(
        m,
        r0_smcs_hard_cs_ens,
        r0_smcs_hard_cs_errs,
        marker="s",
        markeredgecolor=COLOURS["aqua"],
        markerfacecolor=COLOURS["aqua"],
        color=COLOURS["aqua"],
        capsize=3,
        linestyle="",
    )

    ax[0, 0].plot(m, [rx_casci_en] * len(m), color=COLOURS["orange"], linestyle="solid")
    ax[0, 0].plot(m, [r0_casci_en] * len(m), color=COLOURS["navy"], linestyle="solid")
    ax[0, 1].plot(m, [rx_casci_en] * len(m), color=COLOURS["orange"], linestyle="solid")
    ax[0, 1].plot(m, [r0_casci_en] * len(m), color=COLOURS["navy"], linestyle="solid")
    ax[0, 2].plot(m, [rx_casci_en] * len(m), color=COLOURS["orange"], linestyle="solid")
    ax[0, 2].plot(m, [r0_casci_en] * len(m), color=COLOURS["navy"], linestyle="solid")
    ax[1, 0].plot(m, [rx_casci_en] * len(m), color=COLOURS["orange"], linestyle="solid")
    ax[1, 0].plot(m, [r0_casci_en] * len(m), color=COLOURS["navy"], linestyle="solid")
    ax[1, 1].plot(m, [rx_casci_en] * len(m), color=COLOURS["orange"], linestyle="solid")
    ax[1, 1].plot(m, [r0_casci_en] * len(m), color=COLOURS["navy"], linestyle="solid")
    ax[1, 2].plot(m, [rx_casci_en] * len(m), color=COLOURS["orange"], linestyle="solid")
    ax[1, 2].plot(m, [r0_casci_en] * len(m), color=COLOURS["navy"], linestyle="solid")

    ax[0, 0].plot(m, rx_tups_sv, color=COLOURS["orange"], linestyle="dashed")
    ax[0, 0].plot(m, r0_tups_sv, color=COLOURS["navy"], linestyle="dashed")
    ax[0, 1].plot(m, rx_tups_sv, color=COLOURS["orange"], linestyle="dashed")
    ax[0, 1].plot(m, r0_tups_sv, color=COLOURS["navy"], linestyle="dashed")
    ax[0, 2].plot(m, rx_tups_sv, color=COLOURS["orange"], linestyle="dashed")
    ax[0, 2].plot(m, r0_tups_sv, color=COLOURS["navy"], linestyle="dashed")
    ax[1, 0].plot(m, rx_smcs_sv, color=COLOURS["orange"], linestyle="dashed")
    ax[1, 0].plot(m, r0_smcs_sv, color=COLOURS["navy"], linestyle="dashed")
    ax[1, 1].plot(m, rx_smcs_sv, color=COLOURS["orange"], linestyle="dashed")
    ax[1, 1].plot(m, r0_smcs_sv, color=COLOURS["navy"], linestyle="dashed")
    ax[1, 2].plot(m, rx_smcs_sv, color=COLOURS["orange"], linestyle="dashed")
    ax[1, 2].plot(m, r0_smcs_sv, color=COLOURS["navy"], linestyle="dashed")

    ax[0, 0].fill_between(
        m,
        rx_casci_en + 1.6e-3,
        rx_casci_en - 1.6e-3,
        color=COLOURS["orange"],
        alpha=0.1,
    )
    ax[0, 0].fill_between(
        m, r0_casci_en + 1.6e-3, r0_casci_en - 1.6e-3, color=COLOURS["navy"], alpha=0.1
    )
    ax[0, 1].fill_between(
        m,
        rx_casci_en + 1.6e-3,
        rx_casci_en - 1.6e-3,
        color=COLOURS["orange"],
        alpha=0.1,
    )
    ax[0, 1].fill_between(
        m, r0_casci_en + 1.6e-3, r0_casci_en - 1.6e-3, color=COLOURS["navy"], alpha=0.1
    )
    ax[0, 2].fill_between(
        m,
        rx_casci_en + 1.6e-3,
        rx_casci_en - 1.6e-3,
        color=COLOURS["orange"],
        alpha=0.1,
    )
    ax[0, 2].fill_between(
        m, r0_casci_en + 1.6e-3, r0_casci_en - 1.6e-3, color=COLOURS["navy"], alpha=0.1
    )
    ax[1, 0].fill_between(
        m,
        rx_casci_en + 1.6e-3,
        rx_casci_en - 1.6e-3,
        color=COLOURS["orange"],
        alpha=0.1,
    )
    ax[1, 0].fill_between(
        m, r0_casci_en + 1.6e-3, r0_casci_en - 1.6e-3, color=COLOURS["navy"], alpha=0.1
    )
    ax[1, 1].fill_between(
        m,
        rx_casci_en + 1.6e-3,
        rx_casci_en - 1.6e-3,
        color=COLOURS["orange"],
        alpha=0.1,
    )
    ax[1, 1].fill_between(
        m, r0_casci_en + 1.6e-3, r0_casci_en - 1.6e-3, color=COLOURS["navy"], alpha=0.1
    )

    ax[1, 2].fill_between(
        m,
        rx_casci_en + 1.6e-3,
        rx_casci_en - 1.6e-3,
        color=COLOURS["orange"],
        alpha=0.1,
    )
    ax[1, 2].fill_between(
        m, r0_casci_en + 1.6e-3, r0_casci_en - 1.6e-3, color=COLOURS["navy"], alpha=0.1
    )

    ax[0, 0].set(xlabel="(m,k=10,l=2)", ylabel="E (Ha)", title="CU Emulator")
    ax[1, 0].set(xlabel="(m,k=10,l=13)", ylabel="E (Ha)")
    ax[0, 1].set(xlabel="(m,k=10,l=2)", ylabel="", title="CS Emulator")
    ax[1, 1].set(xlabel="(m,k=10,l=13)", ylabel="")
    ax[0, 2].set(xlabel="(m,k=10,l=2)", ylabel="", title="CS Hardware")
    ax[1, 2].set(xlabel="(m,k=10,l=13)", ylabel="")

    axtop, axbot = ax[0, 2].twinx(), ax[1, 2].twinx()
    axtop.set_ylabel(
        "pp-tUPS", rotation=270, labelpad=20, fontsize=12, fontweight="bold"
    )
    axbot.set_ylabel("sMCS", rotation=270, labelpad=20, fontsize=12, fontweight="bold")
    axtop.set_yticks([])
    axbot.set_yticks([])

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            markeredgecolor=COLOURS["orange"],
            markerfacecolor="none",
            label=r"$\mathbf{R}_x$",
            linestyle="",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            markeredgecolor=COLOURS["navy"],
            markerfacecolor="none",
            label=r"$\mathbf{R}_0$",
            linestyle="",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            markeredgecolor=COLOURS["pumpkin"],
            markerfacecolor="none",
            label=r"$\mathbf{R}_x$" + " Hardware",
            linestyle="",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            markeredgecolor=COLOURS["aqua"],
            markerfacecolor="none",
            label=r"$\mathbf{R}_0$" + " Hardware",
            linestyle="",
        ),
    ]
    ax[0, 0].legend(handles=handles, fontsize=6, loc="lower right", framealpha=0.1)

    handles = [
        Line2D(
            [0],
            [0],
            label=r"$\mathbf{R}_x$" + " CASCI",
            linestyle="solid",
            color=COLOURS["orange"],
        ),
        Line2D(
            [0],
            [0],
            label=r"$\mathbf{R}_0$" + " CASCI",
            linestyle="solid",
            color=COLOURS["navy"],
        ),
        Line2D(
            [0],
            [0],
            label=r"$\mathbf{R}_x$" + " ME SV",
            linestyle="dashed",
            color=COLOURS["orange"],
        ),
        Line2D(
            [0],
            [0],
            label=r"$\mathbf{R}_0$" + " ME SV",
            linestyle="dashed",
            color=COLOURS["navy"],
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            markeredgecolor=COLOURS["orange"],
            markerfacecolor=COLOURS["orange"],
            label=r"$\mathbf{R}_x$",
            linestyle="",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            markeredgecolor=COLOURS["navy"],
            markerfacecolor=COLOURS["navy"],
            label=r"$\mathbf{R}_0$",
            linestyle="",
        ),
    ]
    ax[0, 1].legend(
        handles=handles, fontsize=6, loc="lower right", framealpha=0.1, ncol=2
    )

    handles = [
        Line2D(
            [0],
            [0],
            label=r"$\mathbf{R}_x$" + " CASCI",
            linestyle="solid",
            color=COLOURS["orange"],
        ),
        Line2D(
            [0],
            [0],
            label=r"$\mathbf{R}_0$" + " CASCI",
            linestyle="solid",
            color=COLOURS["navy"],
        ),
        Line2D(
            [0],
            [0],
            label=r"$\mathbf{R}_x$" + " ME SV",
            linestyle="dashed",
            color=COLOURS["orange"],
        ),
        Line2D(
            [0],
            [0],
            label=r"$\mathbf{R}_0$" + " ME SV",
            linestyle="dashed",
            color=COLOURS["navy"],
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            markeredgecolor=COLOURS["pumpkin"],
            markerfacecolor=COLOURS["pumpkin"],
            label=r"$\mathbf{R}_x$",
            linestyle="",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            markeredgecolor=COLOURS["aqua"],
            markerfacecolor=COLOURS["aqua"],
            label=r"$\mathbf{R}_0$",
            linestyle="",
        ),
    ]
    ax[0, 2].legend(
        handles=handles, fontsize=6, loc="lower right", framealpha=0.1, ncol=2
    )

    # add subplot labels
    for a, label in zip(ax.flat, string.ascii_lowercase):
        a.text(
            -0.05,
            1.06,
            f"{label})",
            transform=a.transAxes,
            fontsize=14,
            fontweight="bold",
            va="top",
            ha="right",
        )

    # Keep tick marks on every subplot, but hide labels
    for a in ax.flat:
        a.tick_params(axis="x", which="both", bottom=True, labelbottom=False)

    # Show labels only on the bottom row
    for a in ax[-1, :]:
        a.tick_params(axis="x", which="both", labelbottom=True)

        a.set_xticks([1, 2, 3, 4, 5, 6, 7, 8])
        a.set_xticklabels([1, 2, 3, 4, 5, 6, 7, 8])

    home_dir = f"{pl.Path(__file__).parent.parent}"
    plt.savefig(f"{home_dir}/code/cas66_cu_cs_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_expectation_value_mixed_estimator_convergence() -> None:
    rx_mc88_me = [
        MC_ENERGIES["88"]["Rx"][str(det)]["mixed_estimator"] for det in range(1, 101)
    ]

    r0_mc88_me = [
        MC_ENERGIES["88"]["R0"][str(det)]["mixed_estimator"] for det in range(1, 101)
    ]

    rx_mc1010_me = [
        MC_ENERGIES["1010"]["Rx"][str(det)]["mixed_estimator"] for det in range(1, 101)
    ]

    r0_mc1010_me = [
        MC_ENERGIES["1010"]["R0"][str(det)]["mixed_estimator"] for det in range(1, 101)
    ]

    rx_tups66_me = [
        TUPS_ENERGIES["66"]["Rx"][f"L{l}pp"]["mixed_estimator"] for l in range(1, 4)
    ]

    r0_tups66_me = [
        TUPS_ENERGIES["66"]["R0"][f"L{l}pp"]["mixed_estimator"] for l in range(1, 4)
    ]

    rx_mc66_me = [
        MC_ENERGIES["66"]["Rx"][str(det)]["mixed_estimator"] for det in range(1, 101)
    ]

    r0_mc66_me = [
        MC_ENERGIES["66"]["R0"][str(det)]["mixed_estimator"] for det in range(1, 101)
    ]

    rx_mc88_ev = [MC_ENERGIES["88"]["Rx"][str(det)]["exp_val"] for det in range(1, 101)]

    r0_mc88_ev = [MC_ENERGIES["88"]["R0"][str(det)]["exp_val"] for det in range(1, 101)]

    rx_mc1010_ev = [
        MC_ENERGIES["1010"]["Rx"][str(det)]["exp_val"] for det in range(1, 101)
    ]

    r0_mc1010_ev = [
        MC_ENERGIES["1010"]["R0"][str(det)]["exp_val"] for det in range(1, 101)
    ]

    rx_tups66_ev = [
        TUPS_ENERGIES["66"]["Rx"][f"L{l}pp"]["exp_val"] for l in range(1, 4)
    ]

    r0_tups66_ev = [
        TUPS_ENERGIES["66"]["R0"][f"L{l}pp"]["exp_val"] for l in range(1, 4)
    ]

    rx_mc66_ev = [MC_ENERGIES["66"]["Rx"][str(det)]["exp_val"] for det in range(1, 101)]

    r0_mc66_ev = [MC_ENERGIES["66"]["R0"][str(det)]["exp_val"] for det in range(1, 101)]

    rx_casci66_en, r0_casci66_en = (
        CASCI["66"]["Rx"]["energies"]["sv"],
        CASCI["66"]["R0"]["energies"]["sv"],
    )

    rx_casci88_en, r0_casci88_en = (
        CASCI["88"]["Rx"]["energies"]["sv"],
        CASCI["88"]["R0"]["energies"]["sv"],
    )

    rx_casci1010_en, r0_casci1010_en = (
        CASCI["1010"]["Rx"]["energies"]["sv"],
        CASCI["1010"]["R0"]["energies"]["sv"],
    )

    fig = plt.figure(figsize=(9, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    ax = gs.subplots()
    layers, dets = [1, 2, 3], list(range(1, 101))
    cas6688_det, cas1010_det = 13, 24

    ax[0, 0].scatter(
        layers,
        rx_tups66_me,
        marker="o",
        edgecolors=COLOURS["orange"],
        facecolors="none",
    )
    ax[0, 0].scatter(
        layers,
        r0_tups66_me,
        marker="o",
        edgecolors=COLOURS["navy"],
        facecolors="none",
    )
    ax[0, 0].scatter(
        layers,
        rx_tups66_ev,
        marker="d",
        edgecolors=COLOURS["orange"],
        facecolors="none",
    )
    ax[0, 0].scatter(
        layers,
        r0_tups66_ev,
        marker="d",
        edgecolors=COLOURS["navy"],
        facecolors="none",
    )
    ax[0, 0].plot(layers, rx_tups66_me, color=COLOURS["orange"], linestyle="dashed")
    ax[0, 0].plot(layers, r0_tups66_me, color=COLOURS["navy"], linestyle="dashed")
    ax[0, 0].plot(layers, rx_tups66_ev, color=COLOURS["orange"], linestyle="dotted")
    ax[0, 0].plot(layers, r0_tups66_ev, color=COLOURS["navy"], linestyle="dotted")

    ax[0, 1].plot(dets, rx_mc66_me, color=COLOURS["orange"], linestyle="dashed")
    ax[0, 1].plot(dets, r0_mc66_me, color=COLOURS["navy"], linestyle="dashed")
    ax[0, 1].plot(dets, rx_mc66_ev, color=COLOURS["orange"], linestyle="dotted")
    ax[0, 1].plot(dets, r0_mc66_ev, color=COLOURS["navy"], linestyle="dotted")

    ax[0, 1].scatter(
        dets[cas6688_det - 1],
        rx_mc66_me[cas6688_det - 1],
        marker="o",
        edgecolors=COLOURS["orange"],
        facecolors="none",
    )
    ax[0, 1].scatter(
        dets[cas6688_det - 1],
        r0_mc66_me[cas6688_det - 1],
        marker="o",
        edgecolors=COLOURS["navy"],
        facecolors="none",
    )
    ax[0, 1].scatter(
        dets[cas6688_det - 1],
        rx_mc66_ev[cas6688_det - 1],
        marker="d",
        edgecolors=COLOURS["orange"],
        facecolors="none",
    )
    ax[0, 1].scatter(
        dets[cas6688_det - 1],
        r0_mc66_ev[cas6688_det - 1],
        marker="d",
        edgecolors=COLOURS["navy"],
        facecolors="none",
    )

    ax[1, 0].plot(dets, rx_mc88_me, color=COLOURS["orange"], linestyle="dashed")
    ax[1, 0].plot(dets, r0_mc88_me, color=COLOURS["navy"], linestyle="dashed")
    ax[1, 0].plot(dets, rx_mc88_ev, color=COLOURS["orange"], linestyle="dotted")
    ax[1, 0].plot(dets, r0_mc88_ev, color=COLOURS["navy"], linestyle="dotted")

    ax[1, 0].scatter(
        dets[cas6688_det - 1],
        rx_mc88_me[cas6688_det - 1],
        marker="o",
        edgecolors=COLOURS["orange"],
        facecolors="none",
    )
    ax[1, 0].scatter(
        dets[cas6688_det - 1],
        r0_mc88_me[cas6688_det - 1],
        marker="o",
        edgecolors=COLOURS["navy"],
        facecolors="none",
    )
    ax[1, 0].scatter(
        dets[cas6688_det - 1],
        rx_mc88_ev[cas6688_det - 1],
        marker="d",
        edgecolors=COLOURS["orange"],
        facecolors="none",
    )
    ax[1, 0].scatter(
        dets[cas6688_det - 1],
        r0_mc88_ev[cas6688_det - 1],
        marker="d",
        edgecolors=COLOURS["navy"],
        facecolors="none",
    )

    ax[1, 1].plot(dets, rx_mc1010_me, color=COLOURS["orange"], linestyle="dashed")
    ax[1, 1].plot(dets, r0_mc1010_me, color=COLOURS["navy"], linestyle="dashed")
    ax[1, 1].plot(dets, rx_mc1010_ev, color=COLOURS["orange"], linestyle="dotted")
    ax[1, 1].plot(dets, r0_mc1010_ev, color=COLOURS["navy"], linestyle="dotted")

    ax[1, 1].scatter(
        dets[cas1010_det - 1],
        rx_mc1010_me[cas1010_det - 1],
        marker="o",
        edgecolors=COLOURS["orange"],
        facecolors="none",
    )
    ax[1, 1].scatter(
        dets[cas1010_det - 1],
        r0_mc1010_me[cas1010_det - 1],
        marker="o",
        edgecolors=COLOURS["navy"],
        facecolors="none",
    )
    ax[1, 1].scatter(
        dets[cas1010_det - 1],
        rx_mc1010_ev[cas1010_det - 1],
        marker="d",
        edgecolors=COLOURS["orange"],
        facecolors="none",
    )
    ax[1, 1].scatter(
        dets[cas1010_det - 1],
        r0_mc1010_ev[cas1010_det - 1],
        marker="d",
        edgecolors=COLOURS["navy"],
        facecolors="none",
    )

    ax[0, 0].plot(
        layers,
        [rx_casci66_en] * len(layers),
        color=COLOURS["orange"],
        linestyle="solid",
    )
    ax[0, 0].plot(
        layers, [r0_casci66_en] * len(layers), color=COLOURS["navy"], linestyle="solid"
    )
    ax[0, 1].plot(
        dets, [rx_casci66_en] * len(dets), color=COLOURS["orange"], linestyle="solid"
    )
    ax[0, 1].plot(
        dets, [r0_casci66_en] * len(dets), color=COLOURS["navy"], linestyle="solid"
    )
    ax[1, 0].plot(
        dets, [rx_casci88_en] * len(dets), color=COLOURS["orange"], linestyle="solid"
    )
    ax[1, 0].plot(
        dets, [r0_casci88_en] * len(dets), color=COLOURS["navy"], linestyle="solid"
    )
    ax[1, 1].plot(
        dets, [rx_casci1010_en] * len(dets), color=COLOURS["orange"], linestyle="solid"
    )
    ax[1, 1].plot(
        dets, [r0_casci1010_en] * len(dets), color=COLOURS["navy"], linestyle="solid"
    )

    ax[0, 0].fill_between(
        dets,
        rx_casci66_en + 1.6e-3,
        rx_casci66_en - 1.6e-3,
        color=COLOURS["orange"],
        alpha=0.1,
    )
    ax[0, 0].fill_between(
        dets,
        r0_casci66_en + 1.6e-3,
        r0_casci66_en - 1.6e-3,
        color=COLOURS["navy"],
        alpha=0.1,
    )

    ax[0, 1].fill_between(
        dets,
        rx_casci66_en + 1.6e-3,
        rx_casci66_en - 1.6e-3,
        color=COLOURS["orange"],
        alpha=0.1,
    )
    ax[0, 1].fill_between(
        dets,
        r0_casci66_en + 1.6e-3,
        r0_casci66_en - 1.6e-3,
        color=COLOURS["navy"],
        alpha=0.1,
    )

    ax[1, 0].fill_between(
        dets,
        rx_casci88_en + 1.6e-3,
        rx_casci88_en - 1.6e-3,
        color=COLOURS["orange"],
        alpha=0.1,
    )
    ax[1, 0].fill_between(
        dets,
        r0_casci88_en + 1.6e-3,
        r0_casci88_en - 1.6e-3,
        color=COLOURS["navy"],
        alpha=0.1,
    )

    ax[1, 1].fill_between(
        dets,
        rx_casci1010_en + 1.6e-3,
        rx_casci1010_en - 1.6e-3,
        color=COLOURS["orange"],
        alpha=0.1,
    )
    ax[1, 1].fill_between(
        dets,
        r0_casci1010_en + 1.6e-3,
        r0_casci1010_en - 1.6e-3,
        color=COLOURS["navy"],
        alpha=0.1,
    )

    ax[0, 0].set(
        xlabel="Layers", ylabel="E (Ha)", title="pp-tUPS (6,6)", xlim=(0.8, 3.2)
    )
    ax[0, 1].set(xlabel="Determinants", ylabel="", title="sMCS (6,6)")
    ax[1, 0].set(xlabel="Determinants", ylabel="E (Ha)", title="sMCS (8,8)")
    ax[1, 1].set(xlabel="Determinants", ylabel="", title="sMCS (10,10)")
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            markeredgecolor=COLOURS["orange"],
            markerfacecolor="none",
            label=r"$\mathbf{R}_x$" + " ME SV",
            linestyle="dashed",
            color=COLOURS["orange"],
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            markeredgecolor=COLOURS["navy"],
            markerfacecolor="none",
            label=r"$\mathbf{R}_0$" + " ME SV",
            linestyle="dashed",
            color=COLOURS["navy"],
        ),
        Line2D(
            [0],
            [0],
            marker="d",
            markeredgecolor=COLOURS["orange"],
            markerfacecolor="none",
            label=r"$\mathbf{R}_x$" + " EV SV",
            linestyle="dotted",
            color=COLOURS["orange"],
        ),
        Line2D(
            [0],
            [0],
            marker="d",
            markeredgecolor=COLOURS["navy"],
            markerfacecolor="none",
            label=r"$\mathbf{R}_0$" + " EV SV",
            linestyle="dotted",
            color=COLOURS["navy"],
        ),
    ]
    ax[0, 0].legend(handles=handles, fontsize=6, loc="upper right", framealpha=0.1)

    handles = [
        # Line2D(
        #     [0],
        #     [0],
        #     label="ME",
        #     linestyle="dashed",
        #     color=COLOURS["black"]
        # ),
        # Line2D(
        #     [0],
        #     [0],
        #     label="EV SV",
        #     linestyle="dotted",
        #     color=COLOURS["black"]
        # ),
        # Line2D(
        #     [0],
        #     [0],
        #     label="CASCI",
        #     linestyle="solid",
        #     color=COLOURS["black"]
        # ),
        Line2D(
            [0],
            [0],
            color=COLOURS["orange"],
            label=r"$\mathbf{R}_x$ CASCI",
            linestyle="solid",
        ),
        Line2D(
            [0],
            [0],
            color=COLOURS["navy"],
            label=r"$\mathbf{R}_0$ CASCI",
            linestyle="solid",
        ),
        Line2D(
            [0],
            [0],
            color=COLOURS["orange"],
            label=r"$\mathbf{R}_x$ ME",
            linestyle="dashed",
        ),
        Line2D(
            [0],
            [0],
            color=COLOURS["navy"],
            label=r"$\mathbf{R}_0$ ME",
            linestyle="dashed",
        ),
        Line2D(
            [0],
            [0],
            color=COLOURS["orange"],
            label=r"$\mathbf{R}_x$ SV",
            linestyle="dotted",
        ),
        Line2D(
            [0],
            [0],
            color=COLOURS["navy"],
            label=r"$\mathbf{R}_0$ SV",
            linestyle="dotted",
        ),
    ]
    ax[0, 1].legend(handles=handles, fontsize=6, loc="upper right", framealpha=0.1)

    # add subplot labels
    for a, label in zip(ax.flat, string.ascii_lowercase):
        a.text(
            -0.05,
            1.02,
            f"{label})",
            transform=a.transAxes,
            fontsize=14,
            fontweight="bold",
            va="top",
            ha="right",
        )

    ax[0, 0].set_xticks([1, 2, 3])
    ax[0, 0].set_xticklabels([1, 2, 3])

    home_dir = f"{pl.Path(__file__).parent.parent}"
    plt.savefig(
        f"{home_dir}/code/expectation_value_mixed_estimator_convergence.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_compressed_tups_multiconfig_stateprep66() -> None:
    fig = plt.figure(figsize=(9, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    ax = gs.subplots()

    rx_tups_ov = [
        TUPS_ENERGIES["66"]["Rx"][f"L{l}pp"]["casci_overlap"] for l in range(1, 4)
    ]
    rx_tups_zzg = [
        TUPS_CIRC["66"]["Rx"][f"L{l}pp"]["ZZPhase_gates"] for l in range(1, 4)
    ]
    rx_tups_zzd = [
        TUPS_CIRC["66"]["Rx"][f"L{l}pp"]["ZZPhase_depth"] for l in range(1, 4)
    ]

    r0_tups_ov = [
        TUPS_ENERGIES["66"]["R0"][f"L{l}pp"]["casci_overlap"] for l in range(1, 4)
    ]
    r0_tups_zzg = [
        TUPS_CIRC["66"]["R0"][f"L{l}pp"]["ZZPhase_gates"] for l in range(1, 4)
    ]
    r0_tups_zzd = [
        TUPS_CIRC["66"]["R0"][f"L{l}pp"]["ZZPhase_depth"] for l in range(1, 4)
    ]

    rx_smcs_ov = [
        MC_ENERGIES["66"]["Rx"][str(det)]["casci_overlap"] for det in range(1, 101)
    ]
    rx_smcs_zzg = [
        MC_CIRC["66"]["Rx"][str(det)]["ZZPhase_gates"] for det in range(1, 101)
    ]
    rx_smcs_zzd = [
        MC_CIRC["66"]["Rx"][str(det)]["ZZPhase_depth"] for det in range(1, 101)
    ]

    r0_smcs_ov = [
        MC_ENERGIES["66"]["R0"][str(det)]["casci_overlap"] for det in range(1, 101)
    ]
    r0_smcs_zzg = [
        MC_CIRC["66"]["R0"][str(det)]["ZZPhase_gates"] for det in range(1, 101)
    ]
    r0_smcs_zzd = [
        MC_CIRC["66"]["R0"][str(det)]["ZZPhase_depth"] for det in range(1, 101)
    ]

    layers, dets = [1, 2, 3], list(range(1, 101))

    ax[0, 0].scatter(
        layers,
        rx_tups_ov,
        marker="x",
        color=COLOURS["orange"],
    )
    ax[0, 0].scatter(
        layers,
        r0_tups_ov,
        marker="x",
        color=COLOURS["navy"],
    )
    ax[0, 0].plot(layers, rx_tups_ov, color=COLOURS["orange"])
    ax[0, 0].plot(layers, r0_tups_ov, color=COLOURS["navy"])

    ax[0, 1].scatter(
        layers,
        rx_tups_zzg,
        marker="h",
        edgecolors=COLOURS["orange"],
        facecolors="none",
    )
    ax[0, 1].scatter(
        layers,
        r0_tups_zzg,
        marker="h",
        edgecolors=COLOURS["navy"],
        facecolors="none",
    )
    ax[0, 1].plot(
        layers,
        rx_tups_zzg,
        ls="solid",
        color=COLOURS["orange"],
    )
    ax[0, 1].plot(
        layers,
        r0_tups_zzg,
        ls="solid",
        color=COLOURS["navy"],
    )

    ax_tups_depth = ax[0, 1].twinx()
    ax_tups_depth.scatter(
        layers,
        rx_tups_zzd,
        marker="8",
        edgecolors=COLOURS["orange"],
        facecolors="none",
    )
    ax_tups_depth.scatter(
        layers,
        r0_tups_zzd,
        marker="8",
        edgecolors=COLOURS["navy"],
        facecolors="none",
    )
    ax_tups_depth.plot(
        layers,
        rx_tups_zzd,
        ls="dotted",
        color=COLOURS["orange"],
    )
    ax_tups_depth.plot(
        layers,
        r0_tups_zzd,
        ls="dotted",
        color=COLOURS["navy"],
    )

    ax[1, 0].plot(dets, rx_smcs_ov, color=COLOURS["orange"])
    ax[1, 0].plot(dets, r0_smcs_ov, color=COLOURS["navy"])
    ax_smcs_depth = ax[1, 1].twinx()
    ax_smcs_depth.plot(
        dets,
        rx_smcs_zzd,
        ls="dotted",
        color=COLOURS["orange"],
    )
    ax_smcs_depth.plot(
        dets,
        r0_smcs_zzd,
        ls="dotted",
        color=COLOURS["navy"],
    )
    ax[1, 1].plot(
        dets,
        rx_smcs_zzg,
        ls="solid",
        color=COLOURS["orange"],
    )
    ax[1, 1].plot(
        dets,
        r0_smcs_zzg,
        ls="solid",
        color=COLOURS["navy"],
    )

    ax[0, 0].set(
        xlabel="Layers", ylabel=r"$|<\mathrm{CASCI}|\Psi>|^2$", title="pp-tUPS Overlap"
    )
    ax[0, 1].set(xlabel="Layers", ylabel="ZZPhase Count", title="pp-tUPS Resources")
    ax_tups_depth.set(ylabel="ZZPhase Depth")

    ax[1, 0].set(
        xlabel="Determinants",
        ylabel=r"$|<\mathrm{CASCI}|\Psi>|^2$",
        title="sMCS Overlap",
    )
    ax[1, 1].set(xlabel="Determinants", ylabel="ZZPhase Count", title="sMCS Resources")
    ax_smcs_depth.set(ylabel="ZZPhase Depth")

    tups_handles = [
        Line2D(
            [0],
            [0],
            marker="x",
            markeredgecolor=COLOURS["orange"],
            # markerfacecolor="none",
            label=r"$\mathbf{R}_x$",
            linestyle="solid",
            color=COLOURS["orange"],
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            markeredgecolor=COLOURS["navy"],
            # markerfacecolor="none",
            label=r"$\mathbf{R}_0$",
            linestyle="solid",
            color=COLOURS["navy"],
        ),
    ]
    ax[0, 0].legend(handles=tups_handles, fontsize=6, loc="upper right", framealpha=0.1)

    tups_handles = [
        Line2D(
            [0],
            [0],
            marker="h",
            markeredgecolor=COLOURS["orange"],
            markerfacecolor="none",
            label=r"$\mathbf{R}_x$ ZZPhase Count",
            linestyle="solid",
            color=COLOURS["orange"],
        ),
        Line2D(
            [0],
            [0],
            marker="8",
            markeredgecolor=COLOURS["orange"],
            markerfacecolor="none",
            label=r"$\mathbf{R}_x$ ZZPhase Depth",
            linestyle="dotted",
            color=COLOURS["orange"],
        ),
        Line2D(
            [0],
            [0],
            marker="h",
            markeredgecolor=COLOURS["navy"],
            markerfacecolor="none",
            label=r"$\mathbf{R}_0$ ZZPhase Count",
            linestyle="solid",
            color=COLOURS["navy"],
        ),
        Line2D(
            [0],
            [0],
            marker="8",
            markeredgecolor=COLOURS["navy"],
            markerfacecolor="none",
            label=r"$\mathbf{R}_0$ ZZPhase Depth",
            linestyle="dotted",
            color=COLOURS["navy"],
        ),
    ]
    ax[0, 1].legend(handles=tups_handles, fontsize=6, loc="lower right", framealpha=0.1)

    mc_handles = [
        Line2D(
            [0],
            [0],
            color=COLOURS["orange"],
            linestyle="solid",
            label=r"$\mathbf{R}_x$",
        ),
        Line2D(
            [0], [0], color=COLOURS["navy"], linestyle="solid", label=r"$\mathbf{R}_0$"
        ),
    ]
    ax[1, 0].legend(handles=mc_handles, fontsize=6, loc="lower right", framealpha=0.1)

    mc_handles = [
        Line2D(
            [0],
            [0],
            color=COLOURS["orange"],
            linestyle="solid",
            label=r"$\mathbf{R}_x$ ZZPhase Count",
        ),
        Line2D(
            [0],
            [0],
            color=COLOURS["orange"],
            linestyle="dotted",
            label=r"$\mathbf{R}_x$ ZZPhase Depth",
        ),
        Line2D(
            [0],
            [0],
            color=COLOURS["navy"],
            linestyle="solid",
            label=r"$\mathbf{R}_0$ ZZPhase Count",
        ),
        Line2D(
            [0],
            [0],
            color=COLOURS["navy"],
            linestyle="dotted",
            label=r"$\mathbf{R}_0$ ZZPhase Depth",
        ),
    ]
    ax[1, 1].legend(handles=mc_handles, fontsize=6, loc="lower right", framealpha=0.1)

    ax[0, 0].set_xticks([1, 2, 3])
    ax[0, 0].set_xticklabels([1, 2, 3])
    ax[0, 1].set_xticks([1, 2, 3])
    ax[0, 1].set_xticklabels([1, 2, 3])

    for a, label in zip(ax.flat, string.ascii_lowercase):
        a.text(
            -0.05,
            1.06,
            f"{label})",
            transform=a.transAxes,
            fontsize=14,
            fontweight="bold",
            va="top",
            ha="right",
        )

    # # Show labels only on the bottom row
    # for a in ax[0, :]:
    #     a.tick_params(axis="x", which="both", labelbottom=True)

    #     a.set_xticks([1, 2, 3])
    #     a.set_xticklabels([1, 2, 3])

    home_dir = f"{pl.Path(__file__).parent.parent}"
    plt.savefig(
        f"{home_dir}/code/cas66_stateprep_resources_compressed.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_compressed_multiconfig_stateprep_881010() -> None:
    fig = plt.figure(figsize=(9, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    ax = gs.subplots(sharex=True)

    rx_mc88_ov = [
        MC_ENERGIES["88"]["Rx"][str(det)]["casci_overlap"] for det in range(1, 101)
    ]
    rx_mc88_zzg = [
        MC_CIRC["88"]["Rx"][str(det)]["ZZPhase_gates"] for det in range(1, 101)
    ]
    rx_mc88_zzd = [
        MC_CIRC["88"]["Rx"][str(det)]["ZZPhase_depth"] for det in range(1, 101)
    ]

    r0_mc88_ov = [
        MC_ENERGIES["88"]["R0"][str(det)]["casci_overlap"] for det in range(1, 101)
    ]
    r0_mc88_zzg = [
        MC_CIRC["88"]["R0"][str(det)]["ZZPhase_gates"] for det in range(1, 101)
    ]
    r0_mc88_zzd = [
        MC_CIRC["88"]["R0"][str(det)]["ZZPhase_depth"] for det in range(1, 101)
    ]

    rx_mc1010_ov = [
        MC_ENERGIES["1010"]["Rx"][str(det)]["casci_overlap"] for det in range(1, 101)
    ]
    rx_mc1010_zzg = [
        MC_CIRC["1010"]["Rx"][str(det)]["ZZPhase_gates"] for det in range(1, 101)
    ]
    rx_mc1010_zzd = [
        MC_CIRC["1010"]["Rx"][str(det)]["ZZPhase_depth"] for det in range(1, 101)
    ]

    r0_mc1010_ov = [
        MC_ENERGIES["1010"]["R0"][str(det)]["casci_overlap"] for det in range(1, 101)
    ]
    r0_mc1010_zzg = [
        MC_CIRC["1010"]["R0"][str(det)]["ZZPhase_gates"] for det in range(1, 101)
    ]
    r0_mc1010_zzd = [
        MC_CIRC["1010"]["R0"][str(det)]["ZZPhase_depth"] for det in range(1, 101)
    ]

    dets = list(range(1, 101))

    ax[1, 0].plot(dets, rx_mc1010_ov, ls="solid", color=COLOURS["orange"])
    ax[1, 0].plot(dets, r0_mc1010_ov, ls="solid", color=COLOURS["navy"])
    ax[1, 1].plot(dets, rx_mc1010_zzg, ls="solid", color=COLOURS["orange"])
    ax[1, 1].plot(dets, r0_mc1010_zzg, ls="solid", color=COLOURS["navy"])
    ax_mc1010_depth = ax[1, 1].twinx()
    ax_mc1010_depth.plot(dets, rx_mc1010_zzd, ls="dotted", color=COLOURS["orange"])
    ax_mc1010_depth.plot(dets, r0_mc1010_zzd, ls="dotted", color=COLOURS["navy"])

    ax[0, 0].plot(dets, rx_mc88_ov, ls="solid", color=COLOURS["orange"])
    ax[0, 0].plot(dets, r0_mc88_ov, ls="solid", color=COLOURS["navy"])
    ax[0, 1].plot(dets, rx_mc88_zzg, ls="solid", color=COLOURS["orange"])
    ax[0, 1].plot(dets, r0_mc88_zzg, ls="solid", color=COLOURS["navy"])
    ax_mc88_depth = ax[0, 1].twinx()
    ax_mc88_depth.plot(dets, rx_mc88_zzd, ls="dotted", color=COLOURS["orange"])
    ax_mc88_depth.plot(dets, r0_mc88_zzd, ls="dotted", color=COLOURS["navy"])

    ax[0, 0].set(
        xlabel="Determinants",
        ylabel=r"$|<\mathrm{CASCI}|\Psi>|^2$",
        title="(8,8) Overlap",
    )
    ax[0, 1].set(xlabel="Determinants", ylabel="ZZPhase Count", title="(8,8) Resources")
    # ax[0, 2].set(xlabel="Determinants", ylabel="Count", title="ZZPhase Depth")
    # ax[1, 0].set(xlabel="Determinants", ylabel="E/Ha")
    ax[1, 0].set(
        xlabel="Determinants",
        ylabel=r"$|<\mathrm{CASCI}|\Psi>|^2$",
        title="(10,10) Overlap",
    )
    ax[1, 1].set(
        xlabel="Determinants", ylabel="ZZPhase Count", title="(10,10) Resources"
    )
    ax_mc88_depth.set(ylabel="ZZPhase Depth")
    ax_mc1010_depth.set(ylabel="ZZPhase Depth")
    # ax[1, 2].set(xlabel="Determinants", ylabel="Count")

    mc_handles = [
        Line2D(
            [0],
            [0],
            color=COLOURS["orange"],
            linestyle="solid",
            label=r"$\mathbf{R}_x$",
        ),
        Line2D(
            [0], [0], color=COLOURS["navy"], linestyle="solid", label=r"$\mathbf{R}_0$"
        ),
    ]
    ax[1, 0].legend(handles=mc_handles, fontsize=6, loc="lower right", framealpha=0.1)

    mc_handles = [
        Line2D(
            [0],
            [0],
            color=COLOURS["orange"],
            linestyle="solid",
            label=r"$\mathbf{R}_x$ ZZPhase Count",
        ),
        Line2D(
            [0],
            [0],
            color=COLOURS["orange"],
            linestyle="dotted",
            label=r"$\mathbf{R}_x$ ZZPhase Depth",
        ),
        Line2D(
            [0],
            [0],
            color=COLOURS["navy"],
            linestyle="solid",
            label=r"$\mathbf{R}_0$ ZZPhase Count",
        ),
        Line2D(
            [0],
            [0],
            color=COLOURS["navy"],
            linestyle="dotted",
            label=r"$\mathbf{R}_0$ ZZPhase Depth",
        ),
    ]
    ax[1, 1].legend(handles=mc_handles, fontsize=6, loc="lower right", framealpha=0.1)

    for a, label in zip(ax.flat, string.ascii_lowercase):
        a.text(
            -0.05,
            1.06,
            f"{label})",
            transform=a.transAxes,
            fontsize=14,
            fontweight="bold",
            va="top",
            ha="right",
        )

    home_dir = f"{pl.Path(__file__).parent.parent}"
    plt.savefig(
        f"{home_dir}/code/cas881010_multiconfig_resources_compressed.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_tups_smcs_ai_activation_energy_66_1010() -> None:
    fig = plt.figure(figsize=(9, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    ax = gs.subplots()

    rx_tups_em66_cs_ens = np.array(
        CS_ENERGIES["66"]["Rx"]["tups"]["emulator"]["energies"][:8]
    )
    rx_tups_em66_cs_errs = np.array(
        CS_ENERGIES["66"]["Rx"]["tups"]["emulator"]["errors"][:8]
    )
    r0_tups_em66_cs_ens = np.array(
        CS_ENERGIES["66"]["R0"]["tups"]["emulator"]["energies"][:8]
    )
    r0_tups_em66_cs_errs = np.array(
        CS_ENERGIES["66"]["R0"]["tups"]["emulator"]["errors"][:8]
    )
    # Rx/T1 = r/teal
    # CU/CS = empty/filled
    # emulator/hardware = o/s
    delta = 0.1

    ax[0, 0].errorbar(
        [0 - delta, 1 - delta],
        rx_tups_em66_cs_ens[[0, 2]],
        rx_tups_em66_cs_errs[[0, 2]],
        marker="o",
        markeredgecolor=COLOURS["orange"],
        markerfacecolor=COLOURS["orange"],
        ecolor=COLOURS["orange"],
        capsize=4,
        ls="",
    )
    ax[0, 0].errorbar(
        [0 - delta, 1 - delta],
        r0_tups_em66_cs_ens[[0, 2]],
        r0_tups_em66_cs_errs[[0, 2]],
        marker="o",
        markeredgecolor=COLOURS["navy"],
        markerfacecolor=COLOURS["navy"],
        ecolor=COLOURS["navy"],
        capsize=4,
        ls="",
    )

    ax[1, 0].errorbar(
        [0 - delta, 1 - delta],
        rx_tups_em66_cs_ens[[0, 2]] - r0_tups_em66_cs_ens[[0, 2]],
        np.sqrt(r0_tups_em66_cs_errs[[0, 2]] ** 2 + rx_tups_em66_cs_errs[[0, 2]] ** 2),
        marker="o",
        markeredgecolor=COLOURS["green"],
        markerfacecolor=COLOURS["green"],
        ecolor=COLOURS["green"],
        capsize=4,
        ls="",
    )

    rx_smcs_em66_cs_ens = np.array(
        CS_ENERGIES["66"]["Rx"]["smcs"]["emulator"]["energies"][:8]
    )
    rx_smcs_em66_cs_errs = np.array(
        CS_ENERGIES["66"]["Rx"]["smcs"]["emulator"]["errors"][:8]
    )
    r0_smcs_em66_cs_ens = np.array(
        CS_ENERGIES["66"]["R0"]["smcs"]["emulator"]["energies"][:8]
    )
    r0_smcs_em66_cs_errs = np.array(
        CS_ENERGIES["66"]["R0"]["smcs"]["emulator"]["errors"][:8]
    )
    ax[0, 0].errorbar(
        [2 - delta, 3 - delta],
        rx_smcs_em66_cs_ens[[0, 2]],
        rx_smcs_em66_cs_errs[[0, 2]],
        marker="o",
        markeredgecolor=COLOURS["orange"],
        markerfacecolor=COLOURS["orange"],
        ecolor=COLOURS["orange"],
        capsize=4,
        ls="",
    )
    ax[0, 0].errorbar(
        [2 - delta, 3 - delta],
        r0_smcs_em66_cs_ens[[0, 2]],
        r0_smcs_em66_cs_errs[[0, 2]],
        marker="o",
        markeredgecolor=COLOURS["navy"],
        markerfacecolor=COLOURS["navy"],
        ecolor=COLOURS["navy"],
        capsize=4,
        ls="",
    )

    ax[1, 0].errorbar(
        [2 - delta, 3 - delta],
        rx_smcs_em66_cs_ens[[0, 2]] - r0_smcs_em66_cs_ens[[0, 2]],
        np.sqrt(r0_smcs_em66_cs_errs[[0, 2]] ** 2 + rx_smcs_em66_cs_errs[[0, 2]] ** 2),
        marker="o",
        markeredgecolor=COLOURS["green"],
        markerfacecolor=COLOURS["green"],
        ecolor=COLOURS["green"],
        capsize=4,
        ls="",
    )

    rx_tups_hard66_cs_ens = np.array(
        CS_ENERGIES["66"]["Rx"]["tups"]["hardware"]["energies"][:8]
    )
    rx_tups_hard66_cs_errs = np.array(
        CS_ENERGIES["66"]["Rx"]["tups"]["hardware"]["errors"][:8]
    )
    r0_tups_hard66_cs_ens = np.array(
        CS_ENERGIES["66"]["R0"]["tups"]["hardware"]["energies"][:8]
    )
    r0_tups_hard66_cs_errs = np.array(
        CS_ENERGIES["66"]["R0"]["tups"]["hardware"]["errors"][:8]
    )
    ax[0, 0].errorbar(
        [0 - 2 * delta, 1 - 2 * delta],
        rx_tups_hard66_cs_ens[[0, 2]],
        rx_tups_hard66_cs_errs[[0, 2]],
        marker="s",
        markeredgecolor=COLOURS["pumpkin"],
        markerfacecolor=COLOURS["pumpkin"],
        ecolor=COLOURS["pumpkin"],
        capsize=4,
        ls="",
    )
    ax[0, 0].errorbar(
        [0 - 2 * delta, 1 - 2 * delta],
        r0_tups_hard66_cs_ens[[0, 2]],
        r0_tups_hard66_cs_errs[[0, 2]],
        marker="s",
        markeredgecolor=COLOURS["aqua"],
        markerfacecolor=COLOURS["aqua"],
        ecolor=COLOURS["aqua"],
        capsize=4,
        ls="",
    )

    ax[1, 0].errorbar(
        [0 - 2 * delta, 1 - 2 * delta],
        rx_tups_hard66_cs_ens[[0, 2]] - r0_tups_hard66_cs_ens[[0, 2]],
        np.sqrt(
            r0_tups_hard66_cs_errs[[0, 2]] ** 2 + rx_tups_hard66_cs_errs[[0, 2]] ** 2
        ),
        marker="s",
        markeredgecolor=COLOURS["pink"],
        markerfacecolor=COLOURS["pink"],
        ecolor=COLOURS["pink"],
        capsize=4,
        ls="",
    )

    rx_smcs_hard66_cs_ens = np.array(
        CS_ENERGIES["66"]["Rx"]["smcs"]["hardware"]["energies"][:8]
    )
    rx_smcs_hard66_cs_errs = np.array(
        CS_ENERGIES["66"]["Rx"]["smcs"]["hardware"]["errors"][:8]
    )
    r0_smcs_hard66_cs_ens = np.array(
        CS_ENERGIES["66"]["R0"]["smcs"]["hardware"]["energies"][:8]
    )
    r0_smcs_hard66_cs_errs = np.array(
        CS_ENERGIES["66"]["R0"]["smcs"]["hardware"]["errors"][:8]
    )

    ax[0, 0].errorbar(
        [2 - 2 * delta, 3 - 2 * delta],
        rx_smcs_hard66_cs_ens[[0, 2]],
        rx_smcs_hard66_cs_errs[[0, 2]],
        marker="s",
        markeredgecolor=COLOURS["orange"],
        markerfacecolor=COLOURS["orange"],
        ecolor=COLOURS["orange"],
        capsize=4,
        ls="",
    )
    ax[0, 0].errorbar(
        [2 - 2 * delta, 3 - 2 * delta],
        r0_smcs_hard66_cs_ens[[0, 2]],
        r0_smcs_hard66_cs_errs[[0, 2]],
        marker="s",
        markeredgecolor=COLOURS["navy"],
        markerfacecolor=COLOURS["navy"],
        ecolor=COLOURS["navy"],
        capsize=4,
        ls="",
    )

    ax[1, 0].errorbar(
        [2 - 2 * delta, 3 - 2 * delta],
        rx_smcs_hard66_cs_ens[[0, 2]] - r0_smcs_hard66_cs_ens[[0, 2]],
        np.sqrt(
            r0_smcs_hard66_cs_errs[[0, 2]] ** 2 + rx_smcs_hard66_cs_errs[[0, 2]] ** 2
        ),
        marker="s",
        markeredgecolor=COLOURS["pink"],
        markerfacecolor=COLOURS["pink"],
        ecolor=COLOURS["pink"],
        capsize=4,
        ls="",
    )

    rx_tups_em66_cu_ens = np.array(
        CU_ENERGIES["66"]["Rx"]["tups"]["emulator"]["energies"]
    )
    rx_tups_em66_cu_errs = np.array(
        CU_ENERGIES["66"]["Rx"]["tups"]["emulator"]["errors"]
    )
    r0_tups_em66_cu_ens = np.array(
        CU_ENERGIES["66"]["R0"]["tups"]["emulator"]["energies"]
    )
    r0_tups_em66_cu_errs = np.array(
        CU_ENERGIES["66"]["R0"]["tups"]["emulator"]["errors"]
    )
    ax[0, 0].errorbar(
        [0 + delta, 1 + delta],
        rx_tups_em66_cu_ens[[0, 2]],
        rx_tups_em66_cu_errs[[0, 2]],
        marker="o",
        markeredgecolor=COLOURS["orange"],
        markerfacecolor="none",
        ecolor=COLOURS["orange"],
        capsize=4,
        ls="",
    )
    ax[0, 0].errorbar(
        [0 + delta, 1 + delta],
        r0_tups_em66_cu_ens[[0, 2]],
        r0_tups_em66_cu_errs[[0, 2]],
        marker="o",
        markeredgecolor=COLOURS["navy"],
        markerfacecolor="none",
        ecolor=COLOURS["navy"],
        capsize=4,
        ls="",
    )

    ax[1, 0].errorbar(
        [0 + delta, 1 + delta],
        rx_tups_em66_cu_ens[[0, 2]] - r0_tups_em66_cu_ens[[0, 2]],
        np.sqrt(r0_tups_em66_cu_errs[[0, 2]] ** 2 + rx_tups_em66_cu_errs[[0, 2]] ** 2),
        marker="o",
        markeredgecolor=COLOURS["green"],
        markerfacecolor="none",
        ecolor=COLOURS["green"],
        capsize=4,
        ls="",
    )

    rx_smcs_em66_cu_ens = np.array(
        CU_ENERGIES["66"]["Rx"]["smcs"]["emulator"]["energies"]
    )
    rx_smcs_em66_cu_errs = np.array(
        CU_ENERGIES["66"]["Rx"]["smcs"]["emulator"]["errors"]
    )
    r0_smcs_em66_cu_ens = np.array(
        CU_ENERGIES["66"]["R0"]["smcs"]["emulator"]["energies"]
    )
    r0_smcs_em66_cu_errs = np.array(
        CU_ENERGIES["66"]["R0"]["smcs"]["emulator"]["errors"]
    )
    ax[0, 0].errorbar(
        [2 + delta, 3 + delta],
        rx_smcs_em66_cu_ens[[0, 2]],
        rx_smcs_em66_cu_errs[[0, 2]],
        marker="o",
        markeredgecolor=COLOURS["orange"],
        markerfacecolor="none",
        ecolor=COLOURS["orange"],
        capsize=4,
        ls="",
    )
    ax[0, 0].errorbar(
        [2 + delta, 3 + delta],
        r0_smcs_em66_cu_ens[[0, 2]],
        r0_smcs_em66_cu_errs[[0, 2]],
        marker="o",
        markeredgecolor=COLOURS["navy"],
        markerfacecolor="none",
        ecolor=COLOURS["navy"],
        capsize=4,
        ls="",
    )

    ax[1, 0].errorbar(
        [2 + delta, 3 + delta],
        rx_smcs_em66_cu_ens[[0, 2]] - r0_smcs_em66_cu_ens[[0, 2]],
        np.sqrt(r0_smcs_em66_cu_errs[[0, 2]] ** 2 + rx_smcs_em66_cu_errs[[0, 2]] ** 2),
        marker="o",
        markeredgecolor=COLOURS["green"],
        markerfacecolor="none",
        ecolor=COLOURS["green"],
        capsize=4,
        ls="",
    )

    rx_tups_hard66_cu_ens = np.array(
        CU_ENERGIES["66"]["Rx"]["tups"]["hardware"]["energies"]
    )
    rx_tups_hard66_cu_errs = np.array(
        CU_ENERGIES["66"]["Rx"]["tups"]["hardware"]["errors"]
    )
    r0_tups_hard66_cu_ens = np.array(
        CU_ENERGIES["66"]["R0"]["tups"]["hardware"]["energies"]
    )
    r0_tups_hard66_cu_errs = np.array(
        CU_ENERGIES["66"]["R0"]["tups"]["hardware"]["errors"]
    )
    ax[0, 0].errorbar(
        [0 + 2 * delta, 1 + 2 * delta],
        rx_tups_hard66_cu_ens[[0, 1]],
        rx_tups_hard66_cu_errs[[0, 1]],
        marker="s",
        markeredgecolor=COLOURS["pumpkin"],
        markerfacecolor="none",
        ecolor=COLOURS["pumpkin"],
        capsize=4,
        ls="",
    )
    ax[0, 0].errorbar(
        [0 + 2 * delta, 1 + 2 * delta],
        r0_tups_hard66_cu_ens[[0, 1]],
        r0_tups_hard66_cu_errs[[0, 1]],
        marker="s",
        markeredgecolor=COLOURS["aqua"],
        markerfacecolor="none",
        ecolor=COLOURS["aqua"],
        capsize=4,
        ls="",
    )

    ax[1, 0].errorbar(
        [0 + 2 * delta, 1 + 2 * delta],
        rx_tups_hard66_cu_ens[[0, 1]] - r0_tups_hard66_cu_ens[[0, 1]],
        np.sqrt(
            r0_tups_hard66_cu_errs[[0, 1]] ** 2 + rx_tups_hard66_cu_errs[[0, 1]] ** 2
        ),
        marker="s",
        markeredgecolor=COLOURS["pink"],
        markerfacecolor="none",
        ecolor=COLOURS["pink"],
        capsize=4,
        ls="",
    )

    rx_smcs_hard66_cu_ens = np.array(
        CU_ENERGIES["66"]["Rx"]["smcs"]["hardware"]["energies"]
    )
    rx_smcs_hard66_cu_errs = np.array(
        CU_ENERGIES["66"]["Rx"]["smcs"]["hardware"]["errors"]
    )
    r0_smcs_hard66_cu_ens = np.array(
        CU_ENERGIES["66"]["R0"]["smcs"]["hardware"]["energies"]
    )
    r0_smcs_hard66_cu_errs = np.array(
        CU_ENERGIES["66"]["R0"]["smcs"]["hardware"]["errors"]
    )
    ax[0, 0].errorbar(
        [2 + 2 * delta, 3 + 2 * delta],
        rx_smcs_hard66_cu_ens[[0, 1]],
        rx_smcs_hard66_cu_errs[[0, 1]],
        marker="s",
        markeredgecolor=COLOURS["pumpkin"],
        markerfacecolor="none",
        ecolor=COLOURS["pumpkin"],
        capsize=4,
        ls="",
    )
    ax[0, 0].errorbar(
        [2 + 2 * delta, 3 + 2 * delta],
        r0_smcs_hard66_cu_ens[[0, 1]],
        r0_smcs_hard66_cu_errs[[0, 1]],
        marker="s",
        markeredgecolor=COLOURS["aqua"],
        markerfacecolor="none",
        ecolor=COLOURS["aqua"],
        capsize=4,
        ls="",
    )

    ax[1, 0].errorbar(
        [2 + 2 * delta, 3 + 2 * delta],
        rx_smcs_hard66_cu_ens[[0, 1]] - r0_smcs_hard66_cu_ens[[0, 1]],
        np.sqrt(
            r0_smcs_hard66_cu_errs[[0, 1]] ** 2 + rx_smcs_hard66_cu_errs[[0, 1]] ** 2
        ),
        marker="s",
        markeredgecolor=COLOURS["pink"],
        markerfacecolor="none",
        ecolor=COLOURS["pink"],
        capsize=4,
        ls="",
    )

    rx_smcs_em1010_cu_ens = CU_ENERGIES["1010"]["Rx"]["smcs"]["emulator"]["energies"]
    rx_smcs_em1010_cu_errs = CU_ENERGIES["1010"]["Rx"]["smcs"]["emulator"]["errors"]
    r0_smcs_em1010_cu_ens = CU_ENERGIES["1010"]["R0"]["smcs"]["emulator"]["energies"]
    r0_smcs_em1010_cu_errs = CU_ENERGIES["1010"]["R0"]["smcs"]["emulator"]["errors"]
    ax[0, 1].errorbar(
        0,
        rx_smcs_em1010_cu_ens,
        rx_smcs_em1010_cu_errs,
        marker="o",
        markeredgecolor=COLOURS["orange"],
        markerfacecolor="none",
        ecolor=COLOURS["orange"],
        capsize=4,
        ls="",
    )
    ax[0, 1].errorbar(
        0,
        r0_smcs_em1010_cu_ens,
        r0_smcs_em1010_cu_errs,
        marker="o",
        markeredgecolor=COLOURS["navy"],
        markerfacecolor="none",
        ecolor=COLOURS["navy"],
        capsize=4,
        ls="",
    )
    ax[1, 1].errorbar(
        0,
        rx_smcs_em1010_cu_ens - r0_smcs_em1010_cu_ens,
        np.sqrt(r0_smcs_em1010_cu_errs**2 + rx_smcs_em1010_cu_errs**2),
        marker="o",
        markeredgecolor=COLOURS["green"],
        markerfacecolor="none",
        ecolor=COLOURS["green"],
        capsize=4,
        ls="",
    )

    rx_ai_em1010_cu_ens = np.array(
        CU_ENERGIES["1010"]["Rx"]["ai"]["emulator"]["energies"]
    )[1:]
    rx_ai_em1010_cu_errs = np.array(
        CU_ENERGIES["1010"]["Rx"]["ai"]["emulator"]["errors"]
    )[1:]
    r0_ai_em1010_cu_ens = np.array(
        CU_ENERGIES["1010"]["R0"]["ai"]["emulator"]["energies"]
    )[1:]
    r0_ai_em1010_cu_errs = np.array(
        CU_ENERGIES["1010"]["R0"]["ai"]["emulator"]["errors"]
    )[1:]
    ax[0, 1].errorbar(
        [1 - delta, 2 - delta],
        rx_ai_em1010_cu_ens,
        rx_ai_em1010_cu_errs,
        marker="o",
        markeredgecolor=COLOURS["orange"],
        markerfacecolor="none",
        ecolor=COLOURS["orange"],
        capsize=4,
        ls="",
    )
    ax[0, 1].errorbar(
        [1 - delta, 2 - delta],
        r0_ai_em1010_cu_ens,
        r0_ai_em1010_cu_errs,
        marker="o",
        markeredgecolor=COLOURS["navy"],
        markerfacecolor="none",
        ecolor=COLOURS["navy"],
        capsize=4,
        ls="",
    )

    ax[1, 1].errorbar(
        [1 - delta, 2 - delta],
        rx_ai_em1010_cu_ens - r0_ai_em1010_cu_ens,
        np.sqrt(r0_ai_em1010_cu_errs**2 + rx_ai_em1010_cu_errs**2),
        marker="o",
        markeredgecolor=COLOURS["green"],
        markerfacecolor="none",
        ecolor=COLOURS["green"],
        capsize=4,
        ls="",
    )

    rx_ai_hard1010_cu_ens = np.array(
        CU_ENERGIES["1010"]["Rx"]["ai"]["hardware"]["energies"]
    )
    rx_ai_hard1010_cu_errs = np.array(
        CU_ENERGIES["1010"]["Rx"]["ai"]["hardware"]["errors"]
    )
    r0_ai_hard1010_cu_ens = np.array(
        CU_ENERGIES["1010"]["R0"]["ai"]["hardware"]["energies"]
    )
    r0_ai_hard1010_cu_errs = np.array(
        CU_ENERGIES["1010"]["R0"]["ai"]["hardware"]["errors"]
    )
    ax[0, 1].errorbar(
        [1 + delta, 2 + delta, 3 + delta],
        rx_ai_hard1010_cu_ens,
        rx_ai_hard1010_cu_errs,
        marker="s",
        markeredgecolor=COLOURS["pumpkin"],
        markerfacecolor="none",
        ecolor=COLOURS["pumpkin"],
        capsize=4,
        ls="",
    )
    ax[0, 1].errorbar(
        [1 + delta, 2 + delta, 3 + delta],
        r0_ai_hard1010_cu_ens,
        r0_ai_hard1010_cu_errs,
        marker="s",
        markeredgecolor=COLOURS["aqua"],
        markerfacecolor="none",
        ecolor=COLOURS["aqua"],
        capsize=4,
        ls="",
    )

    ax[1, 1].errorbar(
        [1 + delta, 2 + delta, 3 + delta],
        rx_ai_hard1010_cu_ens - r0_ai_hard1010_cu_ens,
        np.sqrt(r0_ai_hard1010_cu_errs**2 + rx_ai_hard1010_cu_errs**2),
        marker="s",
        markeredgecolor="violet",
        markerfacecolor="none",
        ecolor="violet",
        capsize=4,
        ls="",
    )

    rx_casci66_en, r0_casci66_en = (
        CASCI["66"]["Rx"]["energies"]["sv"],
        CASCI["66"]["R0"]["energies"]["sv"],
    )

    rx_casci1010_en, r0_casci1010_en = (
        CASCI["1010"]["Rx"]["energies"]["sv"],
        CASCI["1010"]["R0"]["energies"]["sv"],
    )

    activation_energy_66 = rx_casci66_en - r0_casci66_en
    activation_energy_1010 = rx_casci1010_en - r0_casci1010_en

    cas66_x, cas1010_x = [0 - 2 * delta, 1, 2, 3 + 2 * delta], [0, 1, 2, 3 + delta]

    ax[0, 0].plot(
        cas66_x,
        [rx_casci66_en] * len(cas66_x),
        color=COLOURS["orange"],
        linestyle="solid",
    )
    ax[0, 0].plot(
        cas66_x,
        [r0_casci66_en] * len(cas66_x),
        color=COLOURS["navy"],
        linestyle="solid",
    )
    ax[1, 0].plot(
        cas66_x,
        [activation_energy_66] * len(cas66_x),
        color=COLOURS["black"],
        linestyle="solid",
    )

    ax[0, 1].plot(
        cas1010_x,
        [rx_casci1010_en] * len(cas1010_x),
        color=COLOURS["orange"],
        linestyle="solid",
    )
    ax[0, 1].plot(
        cas1010_x,
        [r0_casci1010_en] * len(cas1010_x),
        color=COLOURS["navy"],
        linestyle="solid",
    )
    ax[1, 1].plot(
        cas1010_x,
        [activation_energy_1010] * len(cas1010_x),
        color=COLOURS["black"],
        linestyle="solid",
    )

    ax[0, 0].fill_between(
        cas66_x,
        rx_casci66_en + 1.6e-3,
        rx_casci66_en - 1.6e-3,
        color=COLOURS["orange"],
        alpha=0.1,
    )
    ax[0, 0].fill_between(
        cas66_x,
        r0_casci66_en + 1.6e-3,
        r0_casci66_en - 1.6e-3,
        color=COLOURS["navy"],
        alpha=0.1,
    )

    ax[0, 1].fill_between(
        cas1010_x,
        rx_casci1010_en + 1.6e-3,
        rx_casci1010_en - 1.6e-3,
        color=COLOURS["orange"],
        alpha=0.1,
    )
    ax[0, 1].fill_between(
        cas1010_x,
        r0_casci1010_en + 1.6e-3,
        r0_casci1010_en - 1.6e-3,
        color=COLOURS["navy"],
        alpha=0.1,
    )

    ax[1, 0].fill_between(
        cas66_x,
        activation_energy_66 + 1.6e-3,
        activation_energy_66 - 1.6e-3,
        color=COLOURS["black"],
        alpha=0.1,
    )

    ax[1, 1].fill_between(
        cas1010_x,
        activation_energy_1010 + 1.6e-3,
        activation_energy_1010 - 1.6e-3,
        color=COLOURS["black"],
        alpha=0.1,
    )

    ax[0, 0].set(ylabel="E (Ha)", title="(6,6) Energy")
    ax[0, 1].set(ylabel="", title="(10,10) Energy")
    ax[1, 0].set(ylabel=r"$\Delta$E (Ha)", title="(6,6) Energy Difference")
    ax[1, 1].set(ylabel="", title="(10,10) Energy Difference")

    for a, label in zip(ax.flat, string.ascii_lowercase):
        a.text(
            -0.05,
            1.06,
            f"{label})",
            transform=a.transAxes,
            fontsize=14,
            fontweight="bold",
            va="top",
            ha="right",
        )

    ax[0, 0].set_xticks([0, 1, 2, 3], ["", "", "", ""])
    ax[0, 1].set_xticks([0, 1, 2, 3], ["", "", "", ""])
    ax[1, 0].set_xticks(
        [0, 1, 2, 3],
        [
            "pp-tUPS\n(1,10,2)",
            "pp-tUPS\n(3,10,2)",
            "sMCS\n(1,10,13)",
            "sMCS\n(3,10,13)",
        ],
    )
    ax[1, 1].set_xticks(
        [0, 1, 2, 3],
        [
            "sMCS\n(1,21,24)",
            "AI-sMCS\nNo PC",
            "AI-sMCS\nCPC",
            "AI-sMCS\nMBPC",
        ],
    )
    ax[1, 0].tick_params(axis="x", labelsize=8, labelrotation=0)
    ax[1, 1].tick_params(axis="x", labelsize=8, labelrotation=0)

    # Rx/T1 = r/teal
    # CU/CS = empty/filled
    # emulator/hardware = o/s

    cucs_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            markeredgecolor=COLOURS["orange"],
            markerfacecolor="none",
            label=r"$\mathbf{R}_x$ CU Emulator",
            linestyle="",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            markeredgecolor=COLOURS["navy"],
            markerfacecolor="none",
            label=r"$\mathbf{R}_0$ CU Emulator",
            linestyle="",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            markeredgecolor=COLOURS["pumpkin"],
            markerfacecolor="none",
            label=r"$\mathbf{R}_x$ CU Hardware",
            linestyle="",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            markeredgecolor=COLOURS["aqua"],
            markerfacecolor="none",
            label=r"$\mathbf{R}_0$ CU Hardware",
            linestyle="",
        ),
        Line2D(
            [0],
            [0],
            color=COLOURS["orange"],
            label=r"$\mathbf{R}_x$ CASCI",
        ),
        Line2D(
            [0],
            [0],
            color=COLOURS["navy"],
            label=r"$\mathbf{R}_0$ CASCI",
        ),
    ]
    ax00_leg1 = ax[0, 0].legend(
        handles=cucs_handles, fontsize=6, loc="lower right", framealpha=0.1, ncol=2
    )
    ax[0, 0].add_artist(ax00_leg1)

    cucs_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            markeredgecolor=COLOURS["orange"],
            markerfacecolor=COLOURS["orange"],
            label=r"$\mathbf{R}_x$ CS Emulator",
            linestyle="",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            markeredgecolor=COLOURS["navy"],
            markerfacecolor=COLOURS["navy"],
            label=r"$\mathbf{R}_0$ CS Emulator",
            linestyle="",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            markeredgecolor=COLOURS["pumpkin"],
            markerfacecolor=COLOURS["pumpkin"],
            label=r"$\mathbf{R}_x$ CS Hardware",
            linestyle="",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            markeredgecolor=COLOURS["aqua"],
            markerfacecolor=COLOURS["aqua"],
            label=r"$\mathbf{R}_0$ CS Hardware",
            linestyle="",
        ),
    ]
    ax00_leg2 = ax[0, 0].legend(
        handles=cucs_handles, fontsize=6, loc="upper right", framealpha=0.1, ncol=2
    )
    ax[0, 0].add_artist(ax00_leg2)

    rxr0_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            markeredgecolor=COLOURS["orange"],
            markerfacecolor="none",
            label=r"$\mathbf{R}_x$" + " CU Emulator",
            linestyle="",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            markeredgecolor=COLOURS["navy"],
            markerfacecolor="none",
            label=r"$\mathbf{R}_0$" + " CU Emulator",
            linestyle="",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            markeredgecolor=COLOURS["pumpkin"],
            markerfacecolor="none",
            label=r"$\mathbf{R}_x$" + " CU Hardware",
            linestyle="",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            markeredgecolor=COLOURS["aqua"],
            markerfacecolor="none",
            label=r"$\mathbf{R}_0$" + " CU Hardware",
            linestyle="",
        ),
    ]
    ax01_leg1 = ax[0, 1].legend(
        handles=rxr0_handles, fontsize=6, loc="upper left", framealpha=0.1
    )
    ax[0, 1].add_artist(ax01_leg1)
    casci_handles = [
        Line2D(
            [0],
            [0],
            color=COLOURS["orange"],
            label=r"$\mathbf{R}_x$ CASCI",
        ),
        Line2D(
            [0],
            [0],
            color=COLOURS["navy"],
            label=r"$\mathbf{R}_0$ CASCI",
        ),
    ]
    ax01_leg2 = ax[0, 1].legend(
        handles=casci_handles, fontsize=6, loc="upper right", framealpha=0.1
    )
    ax[0, 1].add_artist(ax01_leg2)

    activation_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            markeredgecolor=COLOURS["green"],
            markerfacecolor="none",
            label="CU Emulator",
            linestyle="",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            markeredgecolor=COLOURS["pink"],
            markerfacecolor="none",
            label="CU Hardware",
            linestyle="",
        ),
        Line2D(
            [0],
            [0],
            color=COLOURS["black"],
            label="CASCI",
            linestyle="solid",
        ),
    ]

    ax[1, 1].legend(
        handles=activation_handles, fontsize=6, loc="lower left", framealpha=0.1
    )

    activation_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            markeredgecolor=COLOURS["green"],
            markerfacecolor="none",
            label="CU Emulator",
            linestyle="",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            markeredgecolor=COLOURS["pink"],
            markerfacecolor="none",
            label="CU Hardware",
            linestyle="",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            markeredgecolor=COLOURS["green"],
            markerfacecolor=COLOURS["green"],
            label="CS Emulator",
            linestyle="",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            markeredgecolor=COLOURS["pink"],
            markerfacecolor=COLOURS["pink"],
            label="CS Hardware",
            linestyle="",
        ),
        Line2D(
            [0],
            [0],
            color=COLOURS["black"],
            label="CASCI",
            linestyle="solid",
        ),
    ]

    ax[1, 0].legend(
        handles=activation_handles, fontsize=6, loc="upper left", framealpha=0.1, ncol=2
    )

    # for a in ax[0, :]:
    #     a.tick_params(axis="x", which="both", labelbottom=True)

    #     a.set_xticks([1, 2, 3])
    #     a.set_xticklabels([1, 2, 3])

    home_dir = f"{pl.Path(__file__).parent.parent}"
    plt.savefig(
        f"{home_dir}/code/activation_energy_tups_multiconfig_ai_cas661010.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


if __name__ == "__main__":
    x = 1

    plot_cas66_cu_cu_cs_comparison()
    plot_expectation_value_mixed_estimator_convergence()
    plot_compressed_tups_multiconfig_stateprep66()
    plot_compressed_multiconfig_stateprep_881010()
    plot_tups_smcs_ai_activation_energy_66_1010()
