# %% Imports
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase
import seaborn as sns
from pathlib import Path

from src.utils.plotting import *
from src.utils import MODELING_DIR, get_feats

# User defined variables to select proper experiment
PREDICTION_TASK = "Chr1p"  # can be one of "MethylationSubgroup", "Chr22q", or "Chr1p"
FEATURE_SET = "collage"  # can be one of "pyradiomics" or "collage"
EXP_DIRS = sorted(
    [d for d in (MODELING_DIR / FEATURE_SET / PREDICTION_TASK).iterdir() if d.is_dir()]
)
EXP_DIR = EXP_DIRS[-5]  # pyrad: -12 for MethylationSubgroup, -8 for Chr1p, -3 for Chr22q, collage: -5 for Chr1p, else -8
OUTPUT_DIR = Path(f"data/paper_figs/{FEATURE_SET}_{PREDICTION_TASK}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Read in all experiment metadata and results
run_metadata_df = pd.read_csv(EXP_DIR / "run_metadata_df.csv")
SCALER = run_metadata_df["SCALER"].values[0]
LOW_VAR_THRESH = (
    run_metadata_df["LOW_VAR_THRESH"].values[0]
    if not run_metadata_df["LOW_VAR_THRESH"].isna().values[0]
    else None
)
if "CORRELATED_FEATS_THRESH" not in run_metadata_df.columns:
    CORRELATED_FEATS_THRESH = None
else:
    CORRELATED_FEATS_THRESH = (
        run_metadata_df["CORRELATED_FEATS_THRESH"].values[0]
        if not run_metadata_df["CORRELATED_FEATS_THRESH"].isna().values[0]
        else None
    )
LAMBDAS = np.fromstring(run_metadata_df["LAMBDAS"].values[0].strip("[]"), sep=" ")
LR_PARAMS = run_metadata_df["LR_PARAMS"].values[0]
if FEATURE_SET == "pyradiomics":
    PYRAD_FILE = run_metadata_df["PYRAD_FILE"].values[0]
else:
    COLLAGE_DIR = Path(run_metadata_df["COLLAGE_DIR"].values[0])
LABELS_FILE = run_metadata_df["LABELS_FILE"].values[0]
METADATA_FILE = run_metadata_df["METADATA_FILE"].values[0]

best_lambdas = pd.read_csv(EXP_DIR / "best_lambdas.csv")
mean_best_lambda = best_lambdas["lambda_i"].mean()
print("PREDICTION TASK: ", PREDICTION_TASK)
print("Mean best lambda: ", mean_best_lambda)
if FEATURE_SET == "collage":
    print("Mean best win_size: ", best_lambdas["win_size"].mean())
    print("Mean best bin_size: ", best_lambdas["bin_size"].mean())

subset = (
    ["lambda_i", "win_size", "bin_size"] if (FEATURE_SET == "collage") else ["lambda_i"]
)
best_lambdas_counts = best_lambdas.value_counts(subset=subset)
print(best_lambdas_counts)
val_loop = pd.read_csv(EXP_DIR / "validation_loop.csv")
test_loop = pd.read_csv(EXP_DIR / "testing_loop.csv")

test_metrics = pd.read_csv(EXP_DIR / "testing_metrics.csv")
coefs = {}
if PREDICTION_TASK != "MethylationSubgroup":
    coefs[PREDICTION_TASK] = pd.read_csv(EXP_DIR / "coefs.csv", index_col=0)
else:
    coef_files = [f for f in EXP_DIR.iterdir() if f.name.endswith("coefs.csv")]
    for f in coef_files:
        coef_name = f.name.replace("_coefs.csv", "")
        coefs[coef_name] = pd.read_csv(f, index_col=0)
print("All available test metrics:")
print(test_metrics.T)
print()
print("Formatted test metrics:")

def metric_mean(value):
    return float(str(value).split()[0])

for n in [
    "AUC" if PREDICTION_TASK != "MethylationSubgroup" else "Macro AUC",
    "Balanced Accuracy",
    (
        "Binary Recall (Sensitivity)"
        if PREDICTION_TASK != "MethylationSubgroup"
        else "Macro Recall (Sensitivity)"
    ),
    "Specificity" if PREDICTION_TASK != "MethylationSubgroup" else "Macro Specificity",
    (
        "Binary Precision"
        if PREDICTION_TASK != "MethylationSubgroup"
        else "Macro Precision"
    ),
    "Binary F1" if PREDICTION_TASK != "MethylationSubgroup" else "Macro F1",
    "MCC",
]:
    print(f"\t{n}: {test_metrics[n].round(3).item()}")

metadata_df = pd.read_csv(METADATA_FILE)

if FEATURE_SET == "pyradiomics":
    if PREDICTION_TASK == "MethylationSubgroup":
        LOW_VAR_THRESH = None
    X, y, SUBJECTS = get_feats(
        prediction_task=PREDICTION_TASK,
        features_path=PYRAD_FILE,
        labels_path=LABELS_FILE,
        scaler=SCALER,
        low_var_thresh=LOW_VAR_THRESH,
        remove_correlated_feats=CORRELATED_FEATS_THRESH,
    )
else:
    win_size = best_lambdas_counts.index[0][1]
    bin_size = best_lambdas_counts.index[0][2]
    features_path = [
        f
        for f in COLLAGE_DIR.rglob(
            f"*wide-features-collage_win-{win_size}_bin-{bin_size}.csv"
        )
    ][0]
    X, y, SUBJECTS = get_feats(
        prediction_task=PREDICTION_TASK,
        features_path=features_path,
        labels_path=LABELS_FILE,
        scaler=SCALER,
        low_var_thresh=LOW_VAR_THRESH,
        remove_correlated_feats=CORRELATED_FEATS_THRESH,
    )

N = len(X)
N_CLASSES = len(set(y))
CLASS_IDS = ["Intact", "Lost"]
if N_CLASSES == 3:
    CLASS_IDS = ["Merlin Intact", "Immune Enriched", "Hypermetabolic"]

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


fig, ax = plt.subplots()

splits = df_val_summary_agg["Dataset split"].unique()
palette = sns.color_palette("tab10", len(splits))
markers = ["o", "s"]  # train, val
marker_map = dict(zip(splits, markers))

legend_handles = []

for color, split in zip(palette, splits):
    df_s = df_val_summary_agg[df_val_summary_agg["Dataset split"] == split]
    marker = marker_map[split]

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
        marker=marker,
        s=40,
        edgecolor="black",
        zorder=3,
    )

    # Legend handle combines all three
    legend_handles.append((color, marker, 0.25))

ax.set_xlabel("Inverse regularization strength (λ)")
ax.set_ylabel("Log loss")
ax.legend(
    legend_handles,
    splits,
    handler_map={tuple: HandlerLineMarkerPatch()},
    title="NLTOCV split\n±1 std. dev.",
)
plt.show()
plt.close()

df_val_summary_agg[df_val_summary_agg["Dataset split"] == "Validation"].mean_loss.plot()
min_val_loss = df_val_summary_agg[
    df_val_summary_agg["Dataset split"] == "Validation"
].mean_loss.min()
print(
    f"MIN AVG VAL LOSS FOR CORR FEAT THRESH {CORRELATED_FEATS_THRESH}: {min_val_loss}"
)

if FEATURE_SET == "collage":
    # Collage cross-val plot (only one test index for the below... needs to be adapted for all test idxs)
    val_summary = (
        val_loop.groupby(["test_idx", "win_size", "bin_size"])[["train_loss", "val_loss"]]
        .mean()
        .reset_index()
    )
    val_summary = val_summary.rename(
        columns={"train_loss": "Training", "val_loss": "Validation"}
    )
    val_summary_long = val_summary.melt(
        id_vars=["test_idx", "win_size", "bin_size"], var_name="Dataset split", value_name="loss"
    )

    collage_stats = (
        val_summary_long[val_summary_long["Dataset split"] == "Validation"]
        .drop(columns=["test_idx", "Dataset split"])
        .loc[lambda df: df.groupby(["win_size", "bin_size"])["loss"].idxmin()]
    )
    np.array(collage_stats["bin_size"]).reshape(4, 4)
    np.array(collage_stats["win_size"]).reshape(4, 4)
    np.array(collage_stats["loss"]).reshape(4, 4)
    plt.figure()
    sns.heatmap(
        np.array(collage_stats["loss"]).reshape(4, 4),
        xticklabels=np.unique(collage_stats["bin_size"]),
        yticklabels=np.unique(collage_stats["win_size"]),
        # annot=np.array(collage_stats["loss"]).reshape(4, 4),
        # annot=np.array(collage_stats["lambda_i"]).reshape(4, 4),
        cmap="viridis_r",
        cbar_kws={'label': 'Mean validation loss'},
        annot=True,
        fmt=".2f"
    )
    plt.xlabel("CoLlAGe bin size (v)")
    plt.ylabel("CoLlAGe neighborhood size (M)")
    plt.show()
    plt.close()

y_probs = np.vstack(
    [np.fromstring(x.strip("[]"), sep=" ") for x in test_loop["y_probs"].values]
)
y_true = test_loop["y_true"].values
if N_CLASSES == 3:
    metrics = plot_multiclass_results(
        y_probs,
        y_true,
        ["MI", "IE", "HM"],
        PREDICTION_TASK,
        plot=True,
        save_dir=OUTPUT_DIR,
    )
else:
    metrics = plot_binary_results(
        y_probs,
        y_true,
        CLASS_IDS,
        prediction_task=PREDICTION_TASK,
        plot=True,
        save_dir=OUTPUT_DIR,
    )

print("Formatted test metrics:")
for n in [
    "AUC" if PREDICTION_TASK != "MethylationSubgroup" else "Macro AUC",
    "Balanced Accuracy",
    (
        "Binary Recall (Sensitivity)"
        if PREDICTION_TASK != "MethylationSubgroup"
        else "Macro Recall (Sensitivity)"
    ),
    "Specificity" if PREDICTION_TASK != "MethylationSubgroup" else "Macro Specificity",
    (
        "Binary Precision"
        if PREDICTION_TASK != "MethylationSubgroup"
        else "Macro Precision"
    ),
    "Binary F1" if PREDICTION_TASK != "MethylationSubgroup" else "Macro F1",
    "MCC"
]:
    print(f"\t{n}: {metrics[n]}")

youden_names = ["Youden's J"] if PREDICTION_TASK != "MethylationSubgroup" else ["Youden's J (MI)", "Youden's J (IE)", "Youden's J (HM)"]
for n in youden_names:
    print(f"\t{n}: {metrics[n]}")

you = 0
for n in youden_names:
    you += metric_mean(metrics[n])
print("\tYouden's J Average: ", round(you/len(youden_names), 3))

for c in coefs:
    print(f"BEGINNING PLOTS FOR: {c}")
    print("TOTAL NUM FEATS: ", len(coefs[c]))
    print(
        f"Num features needed to explain >= 0.95 var in model coef: ",
        sum((coefs[c]["Cum Var Exp"] < 0.95).values) + 1,
    )
    print(
        "Variance explained by top 10 feats: ",
        round(coefs[c].iloc[:10]["Cum Var Exp"].values[-1], 3),
    )
    most_robust_feats_df = coefs[c].iloc[:5]

    # Heatmap
    plot_heatmap(most_robust_feats_df.filter(like="Test fold"))

    # Boxplots
    protective_label = f"Protective for {c} intact" if c.startswith('C') else f"Predicts NON-{c.replace('Hypermetabolic', 'Hypermitotic')}"
    risk_label = f"Increased risk of {c} lost" if c.startswith('C') else f"Predicts {c.replace('Hypermetabolic', 'Hypermitotic')}"
    plot_coef_boxplot3(
        most_robust_feats_df.filter(like="Test fold").T,
        most_robust_feats_df["Prop Var Exp"],
        figsize=(15, 4),
        alpha=0.01,
        protective_label=protective_label,
        risk_label=risk_label,
        save_path=OUTPUT_DIR / f"{c}_important_features.png",
    )

    # Correlation matrix of top features
    feat_corr = X[most_robust_feats_df["Prop Var Exp"].index].corr()
    plot_corr_matrix(feat_corr, cbar_pos=None)

    # # CoLlAGe Correlation matrix of top features
    # feat_corr = Xs["win-9 bin-64"][
    #     most_robust_feats_df["Prop Var Exp"].index
    # ].corr()
    # plot_corr_matrix(feat_corr)

    # current_coefs_df.drop(columns=[c for c in most_robust_feats_df.columns if not c.startswith('Test fold')]).T.describe().T[["mean", "std", "min", "max"]].sort_values(by="mean", ascending=False)
    # Frequency stability
    (coefs[c].filter(like="Test fold") != 0).T.mean()
# %%
