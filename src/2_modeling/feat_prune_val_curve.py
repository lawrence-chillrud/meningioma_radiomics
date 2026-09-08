# %% Imports
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase
import matplotlib as mpl

from src.utils.plotting import *
from src.utils import MODELING_DIR

# User defined variables to select proper experiment
PREDICTION_TASK = "Chr22q"  # "MethylationSubgroup", "Chr22q", or "Chr1p"
FEATURE_SET = "collage"  # can be one of "pyradiomics" or "collage"
EXP_DIRS = sorted(
    [d for d in (MODELING_DIR / FEATURE_SET / PREDICTION_TASK).iterdir() if d.is_dir()]
)

print(PREDICTION_TASK)

FONT_SIZE = 14

mpl.rcParams.update(
    {
        "font.size": FONT_SIZE - 4,
        "axes.titlesize": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE - 4, # for MethylationSubgroup tasks,
        "figure.titlesize": FONT_SIZE,
    }
)

class HandlerLineMarkerPatch(HandlerBase):
    """Custom handler to draw line + marker + patch in one legend entry."""

    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        line_color, marker_style, patch_alpha = orig_handle
        # Make rectangle taller and centered
        patch_height = height * 1
        patch = Rectangle(
            [xdescent, ydescent + (height - patch_height) / 2],  # center it
            width,
            patch_height,
            facecolor=line_color,
            alpha=patch_alpha,
            transform=trans,
        )
        # Line through center
        line = Line2D(
            [xdescent, xdescent + width],
            [ydescent + height / 2] * 2,
            color=line_color,
            lw=2,
            transform=trans,
        )
        # Marker at center
        marker = Line2D(
            [xdescent + width * 0.5],
            [ydescent + height / 2],
            color=line_color,
            marker=marker_style,
            markersize=8,
            markeredgecolor="black",
            transform=trans,
            linestyle="",
        )
        return [patch, line, marker]

dirs_to_plot = [EXP_DIRS[i] for i in range(-8, 0)]

fig, ax = plt.subplots()
palette = sns.color_palette("tab10", len(dirs_to_plot))
legend_handles = []
corr_feats_threshs = []
means = []
stds = []
for color, dtp in zip(palette, dirs_to_plot):
    run_metadata_df = pd.read_csv(dtp / "run_metadata_df.csv")
    if "CORRELATED_FEATS_THRESH" not in run_metadata_df.columns:
        CORRELATED_FEATS_THRESH = 1
    else:
        CORRELATED_FEATS_THRESH = (
            run_metadata_df["CORRELATED_FEATS_THRESH"].values[0]
            if not run_metadata_df["CORRELATED_FEATS_THRESH"].isna().values[0]
            else 1
        )
    if CORRELATED_FEATS_THRESH == 0:
        CORRELATED_FEATS_THRESH = 1
    corr_feats_threshs.append(CORRELATED_FEATS_THRESH)
    val_loop = pd.read_csv(dtp / "validation_loop.csv")

    val_summary = (
        val_loop.groupby(["test_idx", "lambda_i"])[["train_loss", "val_loss"]]
        .mean()
        .reset_index()
    )
    val_summary = val_summary.rename(
        columns={"train_loss": "Training", "val_loss": "Validation"}
    )
    val_summary_long = val_summary.melt(
        id_vars=["test_idx", "lambda_i"], var_name="Dataset split", value_name="loss"
    )
    df_val_summary_agg = (
        val_summary_long.groupby(["lambda_i", "Dataset split"])
        .agg(
            mean_loss=("loss", "mean"),
            std_loss=("loss", "std"),
            n=("loss", "count"),
        )
        .reset_index()
    )
    # 95% CI using std-dev
    df_val_summary_agg["ci_low"] = df_val_summary_agg[
        "mean_loss"
    ] - 1.96 * df_val_summary_agg["std_loss"] / np.sqrt(df_val_summary_agg["n"])
    df_val_summary_agg["ci_high"] = df_val_summary_agg[
        "mean_loss"
    ] + 1.96 * df_val_summary_agg["std_loss"] / np.sqrt(df_val_summary_agg["n"])

    df_s = df_val_summary_agg[df_val_summary_agg["Dataset split"] == "Validation"]

    min_idx = np.argmin(df_s["mean_loss"])
    means.append(df_s["mean_loss"].iloc[min_idx])
    stds.append(df_s["std_loss"].iloc[min_idx])

    # Ribbon
    ax.fill_between(
        df_s["lambda_i"],
        df_s["mean_loss"] - df_s["std_loss"],
        df_s["mean_loss"] + df_s["std_loss"],
        alpha=0.25,
        color=color,
        linewidth=0,
        zorder=1,
    )

    # Line
    sns.lineplot(
        data=df_s,
        x="lambda_i",
        y="mean_loss",
        ax=ax,
        color=color,
        zorder=2,
    )

    # Marker
    sns.scatterplot(
        data=df_s,
        x="lambda_i",
        y="mean_loss",
        ax=ax,
        color=color,
        marker="o",
        s=40,
        edgecolor="black",
        zorder=3,
    )

    # Legend handle combines all three
    legend_handles.append((color, "o", 0.25))

ax.set_xlabel("Inverse regularization strength (λ)")
ax.set_ylabel("Validation log loss")
ax.legend(
    legend_handles,
    corr_feats_threshs,
    handler_map={tuple: HandlerLineMarkerPatch()},
    title="Pruning threshold τ",
)
plt.show()
plt.close()

df_to_plot = pd.DataFrame({
    "mean": means, "std": stds, "corr": corr_feats_threshs
    }).sort_values(by=["corr"])

fig, ax = plt.subplots()

# Ribbon
ax.fill_between(
    df_to_plot["corr"],
    df_to_plot["mean"] - df_to_plot["std"],
    df_to_plot["mean"] + df_to_plot["std"],
    alpha=0.25,
    color="tab:green",
    linewidth=0,
    zorder=1,
)

# Line
sns.lineplot(
    data=df_to_plot,
    x="corr",
    y="mean",
    ax=ax,
    color="tab:green",
    zorder=2,
)

# Marker
sns.scatterplot(
    data=df_to_plot,
    x="corr",
    y="mean",
    ax=ax,
    color="tab:green",
    marker="o",
    s=40,
    edgecolor="black",
    zorder=3,
)

ax.set_xlabel("Feature pruning correlation threshold (τ)")
ax.set_ylabel("Validation log loss")
plt.show()
plt.close()

# %%
