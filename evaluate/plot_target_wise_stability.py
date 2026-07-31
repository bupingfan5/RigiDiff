import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.size"] = 7.5
plt.rcParams["axes.labelsize"] = 8
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["xtick.labelsize"] = 7
plt.rcParams["ytick.labelsize"] = 7
plt.rcParams["xtick.major.width"] = 0.7
plt.rcParams["ytick.major.width"] = 0.7
plt.rcParams["xtick.major.size"] = 3
plt.rcParams["ytick.major.size"] = 3
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["legend.frameon"] = False
plt.rcParams["legend.fontsize"] = 7


VIOLET = "#6F3C8E"
ORANGE = "#D17A22"
NEUTRAL = "#606060"


def target_index(name):
    match = re.search(r"(\d+)$", str(name))
    if match is None:
        raise ValueError(f"Cannot parse target index from {name!r}")
    return int(match.group(1))


def build_source_data(attempts_path, vina_path):
    attempts = pd.read_csv(attempts_path)
    vina = pd.read_csv(vina_path, low_memory=False)

    attempts["target_id"] = attempts["target_name"].map(target_index)
    attempts["assembly_success_bool"] = (
        attempts["assembly_success"].astype(str).str.lower().eq("true")
    )
    generation = (
        attempts.groupby(["target_id", "target_name"], as_index=False)
        .agg(
            generation_attempts=("assembly_success_bool", "size"),
            assembly_successes=("assembly_success_bool", "sum"),
        )
        .sort_values("target_id")
    )
    generation["generation_success_rate"] = (
        generation["assembly_successes"] / generation["generation_attempts"]
    )

    vina["target_id"] = vina["target_name"].map(target_index)
    vina["vina_dock"] = pd.to_numeric(vina["vina_dock"], errors="coerce")

    docking_rows = []
    for (identifier, name), frame in vina.groupby(["target_id", "target_name"]):
        scores = frame["vina_dock"].dropna().sort_values().to_numpy(dtype=float)
        if scores.size == 0:
            raise ValueError(f"No valid Vina scores for {name}")
        docking_rows.append(
            {
                "target_id": int(identifier),
                "target_name": name,
                "docked_molecules": int(scores.size),
                "mean_vina": float(scores.mean()),
                "top5_mean_vina": float(scores[: min(5, scores.size)].mean()),
            }
        )
    docking = pd.DataFrame(docking_rows).sort_values("target_id")

    source = generation.merge(
        docking,
        on=["target_id", "target_name"],
        how="inner",
        validate="one_to_one",
    )
    if len(source) != 99:
        raise ValueError(f"Expected 99 targets, found {len(source)}")
    if not (source["generation_attempts"] == 100).all():
        raise ValueError("Each target must contain exactly 100 generation attempts")

    expected = {
        "generation_success_rate": 0.535959595959596,
        "mean_vina": -6.80346070707071,
        "top5_mean_vina": -9.86120808080808,
    }
    for column, reference in expected.items():
        observed = float(source[column].mean())
        if not np.isclose(observed, reference, atol=1e-10):
            raise ValueError(
                f"{column} mismatch: observed {observed}, expected {reference}"
            )
    return source


def add_panel_label(axis, label):
    axis.text(
        -0.12,
        1.03,
        label,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
    )


def plot(source, output_base):
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(7.20, 2.65),
        gridspec_kw={"wspace": 0.30},
    )
    x = source["target_id"].to_numpy()

    success = source["generation_success_rate"].to_numpy()
    success_mean = float(success.mean())
    axes[0].scatter(
        x,
        success,
        s=15,
        marker="o",
        color=VIOLET,
        alpha=0.82,
        linewidths=0,
        rasterized=False,
        zorder=3,
    )
    axes[0].axhline(
        success_mean,
        color=NEUTRAL,
        linewidth=1.0,
        linestyle=(0, (4, 2)),
        zorder=2,
    )
    axes[0].text(
        0.98,
        0.96,
        f"Macro mean = {success_mean:.3f}",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        color=NEUTRAL,
        fontsize=7,
    )
    axes[0].set_xlabel("Test pocket index")
    axes[0].set_ylabel("Generation success rate")
    axes[0].set_xlim(-2, 100)
    axes[0].set_ylim(0.35, 0.75)
    axes[0].set_xticks([0, 20, 40, 60, 80, 98])
    axes[0].set_yticks([0.4, 0.5, 0.6, 0.7])
    add_panel_label(axes[0], "a")

    mean_vina = source["mean_vina"].to_numpy()
    top5_vina = source["top5_mean_vina"].to_numpy()
    mean_vina_macro = float(mean_vina.mean())
    top5_vina_macro = float(top5_vina.mean())

    axes[1].scatter(
        x,
        mean_vina,
        s=15,
        marker="o",
        color=VIOLET,
        alpha=0.82,
        linewidths=0,
        label="Mean Vina",
        zorder=3,
    )
    axes[1].scatter(
        x,
        top5_vina,
        s=18,
        marker="^",
        color=ORANGE,
        alpha=0.82,
        linewidths=0,
        label="Top-5 mean",
        zorder=3,
    )
    axes[1].axhline(
        mean_vina_macro,
        color=VIOLET,
        linewidth=0.9,
        linestyle=(0, (4, 2)),
        alpha=0.8,
        zorder=2,
    )
    axes[1].axhline(
        top5_vina_macro,
        color=ORANGE,
        linewidth=0.9,
        linestyle=(0, (4, 2)),
        alpha=0.8,
        zorder=2,
    )
    axes[1].set_xlabel("Test pocket index")
    axes[1].set_ylabel(r"Vina Score (kcal mol$^{-1}$)")
    axes[1].set_xlim(-2, 100)
    axes[1].set_ylim(-12.2, -5.3)
    axes[1].set_xticks([0, 20, 40, 60, 80, 98])
    axes[1].set_yticks([-12, -10, -8, -6])
    axes[1].legend(
        loc="lower right",
        borderaxespad=0.3,
        handletextpad=0.4,
        labelspacing=0.35,
    )
    add_panel_label(axes[1], "b")

    for axis in axes:
        axis.tick_params(direction="out")
        axis.spines["left"].set_color("#333333")
        axis.spines["bottom"].set_color("#333333")

    figure.subplots_adjust(left=0.09, right=0.995, bottom=0.21, top=0.95)

    output_base = Path(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_base.with_suffix(".svg"),
        bbox_inches="tight",
        pad_inches=0.03,
    )
    figure.savefig(
        output_base.with_suffix(".pdf"),
        bbox_inches="tight",
        pad_inches=0.03,
    )
    figure.savefig(
        output_base.with_suffix(".png"),
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.03,
    )
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--attempts",
        default=r"E:\fragdiff1\fragdiff\samples\final run\generation_attempts.csv",
    )
    parser.add_argument(
        "--vina",
        default=(
            r"E:\fragdiff1\fragdiff"
            r"\vina_full_protein_allchains_openbabel_test99_seed2025.csv"
        ),
    )
    parser.add_argument(
        "--output",
        default=r"E:\fragdiff1\figures\target_wise_stability",
    )
    args = parser.parse_args()

    source = build_source_data(args.attempts, args.vina)
    output_base = Path(args.output)
    source.to_csv(
        output_base.with_name(output_base.name + "_source.csv"),
        index=False,
    )
    plot(source, output_base)

    print(f"targets={len(source)}")
    print(f"generation_success_macro={source['generation_success_rate'].mean():.6f}")
    print(f"mean_vina_macro={source['mean_vina'].mean():.6f}")
    print(f"top5_vina_macro={source['top5_mean_vina'].mean():.6f}")
    for suffix in (".svg", ".pdf", ".png"):
        print(output_base.with_suffix(suffix))


if __name__ == "__main__":
    main()
