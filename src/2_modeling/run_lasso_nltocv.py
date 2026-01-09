"""
In the nested leave-two-out cross-validation (NLTOCV) procedure, we want to run the following pseudocode:

# Outer loop constructing independent test split aiming to give unbiased estimate of model generalizability
for test_sample in dataset:
    lambda_performances = []
    # Inner loop over lambdas for model selection step
    for lambda_ in lambdas:
        val_hats = []
        # Innermost loop constructing independent val split
        for val_sample in dataset - test_sample
            train_dataset = dataset - test_sample - val_sample
            model.fit(train_dataset, lambda_)
            val_hats.append(model.predict(val_sample))
        lambda_performances.append(mean(val_hats))
    lambda_optimal = argmax(lambda_performances)
    # Final model construction
    train_dataset = dataset - test_sample
    model.fit(train_dataset, lambda_optimal)
    test_hats.append(model.predict(test_sample))
test_performance = mean(test_hats)

This is highly parallelizable. We can parallelize:
1. All necessary model runs invoked by the inner two loops (validation & model selection); and
2. All necessary models runs invoked by the outermost loop (testing loop)
"""

# %%
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from itertools import product

from datetime import datetime
import logging
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase
import numpy as np
from pathlib import Path
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from tqdm import tqdm

from src.utils import PYRAD_FILE, LABELS_FILE, METADATA_FILE, MODELING_DIR, get_feats
from src.utils.plotting import *

# User defined settings
PREDICTION_TASK = "Chr1p"  # can be one of "MethylationSubgroup", "Chr22q", or "Chr1p"
SCALER = "Standard"  # can be one of "Standard", "MinMax", or None
LOW_VAR_THRESH = None  # or 0.2?
MAX_WORKERS = 16
LAMBDAS = np.linspace(0.05, 0.35, 30)
LR_PARAMS = {
    "penalty": "l1",
    "class_weight": "balanced",
    "solver": "liblinear",
    "random_state": 0,
    "max_iter": 100,
    "verbose": 0,
}

# Output dir and logfile set up
TIMESTAMP = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
OUTPUT_DIR = MODELING_DIR / "pyradiomics" / PREDICTION_TASK / TIMESTAMP
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOGFILE = OUTPUT_DIR / "logfile.txt"
# Setup logfile
logging.basicConfig(
    filename=LOGFILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Read in data
metadata_df = pd.read_csv(METADATA_FILE)
X, y, SUBJECTS = get_feats(
    prediction_task=PREDICTION_TASK,
    features_path=PYRAD_FILE,
    labels_path=LABELS_FILE,
    scaler=SCALER,
    low_var_thresh=LOW_VAR_THRESH,
)

# Log run's metadata
run_metadata_df = pd.DataFrame(
    {
        "PREDICTION_TASK": PREDICTION_TASK,
        "SCALER": SCALER,
        "LOW_VAR_THRESH": LOW_VAR_THRESH,
        "LAMBDAS": [LAMBDAS],
        "PYRAD_FILE": PYRAD_FILE,
        "LABELS_FILE": LABELS_FILE,
        "METADATA_FILE": METADATA_FILE,
        "LR_PARAMS": [LR_PARAMS],
    },
    index=[0],
)
run_metadata_df.to_csv(OUTPUT_DIR / "run_metadata_df.csv", index=False)
logging.info(f"<>" * 40)
logging.info(f"Log file for {Path(__file__).name} run at {TIMESTAMP}")
logging.info(f"Settings used for run stored in: run_metadata_df.csv")

# Variables based on data read in
N = len(X)
N_CLASSES = len(set(y))
CLASS_IDS = ["Intact", "Lost"]
if N_CLASSES == 3:
    CLASS_IDS = ["Merlin Intact", "Immune Enriched", "Hypermetabolic"]


def val_job(test_idx, val_idx, lambda_i):
    # Data split
    train_idx = [k for k in range(N) if k not in (test_idx, val_idx)]
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_val, y_val = X.iloc[[val_idx]], [y[val_idx]]

    # Fit logistic LASSO regression
    model = LogisticRegression(C=lambda_i, **LR_PARAMS)
    model.fit(X_train, y_train)

    # Train metrics
    y_probs_train = model.predict_proba(X_train)
    train_loss = log_loss(y_train, y_probs_train, labels=np.unique(y))

    # Validation loss
    y_probs_val = model.predict_proba(X_val)
    val_loss = log_loss(y_val, y_probs_val, labels=np.unique(y))

    return dict(
        test_idx=test_idx,
        val_idx=val_idx,
        lambda_i=lambda_i,
        train_loss=train_loss,
        val_loss=val_loss,
    )


val_loop = [
    (n, m, lambda_i)
    for n, m, lambda_i in product(range(N), range(N), LAMBDAS)
    if n != m
]

results = Parallel(n_jobs=MAX_WORKERS, backend="loky", verbose=0)(
    delayed(val_job)(test_idx, val_idx, lambda_i)
    for test_idx, val_idx, lambda_i in tqdm(
        val_loop, total=len(val_loop), ncols=120, desc="Step 1/2: Model selection loops"
    )
)

df = pd.DataFrame(results)

df.to_csv(OUTPUT_DIR / "validation_loop.csv", index=False)
logging.info(
    "Step 1/2 complete. Validation loop results stored in: validation_loop.csv"
)

val_summary = (
    df.groupby(["test_idx", "lambda_i"])[["train_loss", "val_loss"]]
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


# --- plot ---
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


# Plot
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

ax.set_xlabel("Inverse regularization strength (1/λ)")
ax.set_ylabel("Log loss")
ax.legend(
    legend_handles,
    splits,
    handler_map={tuple: HandlerLineMarkerPatch()},
    title="NLTOCV split\n±1 std. dev.",
)
plt.show()
plt.close()
# --- end plot ---

best_lambdas = val_summary.loc[
    val_summary.groupby("test_idx")["Validation"].idxmin()
].set_index("test_idx")["lambda_i"]

best_lambdas.to_csv(OUTPUT_DIR / "best_lambdas.csv")
logging.info("\tSaved best_lambdas.csv")
best_lambdas.value_counts().to_csv(OUTPUT_DIR / "best_lambdas_counts.csv")
logging.info("\tSaved best_lambdas_counts.csv")


def test_job(test_idx, test_lambda):
    # Data split
    train_idx = [k for k in range(N) if k != test_idx]
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_test, y_test = X.iloc[[test_idx]], [y[test_idx]]

    # Fit logistic LASSO regression
    model = LogisticRegression(C=test_lambda, **LR_PARAMS)
    model.fit(X_train, y_train)

    # Test loss
    y_probs = model.predict_proba(X_test)
    test_loss = log_loss(y_test, y_probs, labels=np.unique(y))

    # Save coefficients
    coefs = model.coef_

    return dict(
        test_idx=test_idx,
        test_lambda=test_lambda,
        test_loss=test_loss,
        y_true=y_test[0],
        y_probs=y_probs,
        y_pred=y_probs.argmax(),
        coefs=coefs,
    )


outer_results = Parallel(n_jobs=MAX_WORKERS)(
    delayed(test_job)(i, best_lambdas[i])
    for i in tqdm(range(N), total=N, ncols=120, desc="Step 2/2: Test loop")
)
outer_df = pd.DataFrame(outer_results)
outer_df["Subject Number"] = SUBJECTS
outer_df.to_csv(OUTPUT_DIR / "testing_loop.csv", index=False)
logging.info("Step 2/2 complete. Testing loop results stored in: testing_loop.csv")
test_coefs = np.stack(outer_df.coefs)

# %%
if N_CLASSES == 3:
    metrics = plot_multiclass_results(
        outer_df["y_probs"], outer_df["y_true"], ["MI", "IE", "HM"], PREDICTION_TASK
    )
else:
    metrics = plot_binary_results(outer_df["y_probs"], outer_df["y_true"], CLASS_IDS)

pd.DataFrame(metrics, index=[0]).to_csv(OUTPUT_DIR / "testing_metrics.csv", index=False)
logging.info("\tTesting metrics saved to: testing_metrics.csv")
print(metrics)
# %%
test_coefs = test_coefs.squeeze()
if len(test_coefs.shape) == 3:
    for c in range(test_coefs.shape[1]):
        current_model = test_coefs[:, c, :]
        nonzero_feats_idxs = np.nonzero(np.sum(current_model, axis=0))[0]
        current_coefs = current_model[:, nonzero_feats_idxs]
        current_coefs_df = pd.DataFrame(
            current_coefs, columns=X.columns[nonzero_feats_idxs]
        ).T
        current_coefs_df.columns = [f"Test fold {i + 1}" for i in range(N)]
        current_coefs_df["Absolute Sum"] = current_coefs_df.abs().sum(axis=1)
        current_coefs_df = current_coefs_df.sort_values(
            by="Absolute Sum", ascending=False
        )
        current_coefs_df["Prop Var Exp"] = (
            current_coefs_df["Absolute Sum"] / current_coefs_df["Absolute Sum"].sum()
        )
        current_coefs_df["Cum Var Exp"] = current_coefs_df["Prop Var Exp"].cumsum()
        # most_robust_feats_df = current_coefs_df[current_coefs_df["Cum Var Exp"] < 0.95]
        most_robust_feats_df = current_coefs_df.iloc[:5]

        # Heatmap
        plot_heatmap(most_robust_feats_df.filter(like="Test fold"))

        # Boxplots
        plot_coef_boxplot2(
            most_robust_feats_df.filter(like="Test fold").T,
            most_robust_feats_df["Prop Var Exp"],
            figsize=(15, 4),
        )
        # plot_coef_boxplot2(most_robust_feats_df.filter(like="Test fold").T)
        # plot_var_exp(most_robust_feats_df["Prop Var Exp"])

        # Correlation matrix of top features
        feat_corr = X[most_robust_feats_df["Prop Var Exp"].index].corr()
        plot_corr_matrix(feat_corr)

        # current_coefs_df.drop(columns=[c for c in most_robust_feats_df.columns if not c.startswith('Test fold')]).T.describe().T[["mean", "std", "min", "max"]].sort_values(by="mean", ascending=False)
        # Frequency stability
        (current_coefs_df.filter(like="Test fold") != 0).T.mean()

        current_coefs_df["Feature"] = current_coefs_df.index
        current_coefs_df["Prediction task"] = CLASS_IDS[c]
        output_dir = MODELING_DIR / "pyradiomics"
        output_dir.mkdir(parents=True, exist_ok=True)
        current_coefs_df.to_csv(OUTPUT_DIR / f"{CLASS_IDS[c]}_coefs.csv", index=False)
        logging.info(f"\tSaved {CLASS_IDS[c]}_coefs.csv")
else:
    nonzero_feats_idxs = np.nonzero(np.sum(test_coefs, axis=0))[0]
    current_coefs = test_coefs[:, nonzero_feats_idxs]
    current_coefs_df = pd.DataFrame(
        current_coefs, columns=X.columns[nonzero_feats_idxs]
    ).T
    current_coefs_df.columns = [f"Test fold {i + 1}" for i in range(N)]
    current_coefs_df["Absolute Sum"] = current_coefs_df.abs().sum(axis=1)
    current_coefs_df = current_coefs_df.sort_values(by="Absolute Sum", ascending=False)
    current_coefs_df["Prop Var Exp"] = (
        current_coefs_df["Absolute Sum"] / current_coefs_df["Absolute Sum"].sum()
    )
    current_coefs_df["Cum Var Exp"] = current_coefs_df["Prop Var Exp"].cumsum()
    # most_robust_feats_df = current_coefs_df[current_coefs_df["Cum Var Exp"] < 0.95]
    most_robust_feats_df = current_coefs_df.iloc[:5]

    # Heatmap
    plot_heatmap(most_robust_feats_df.filter(like="Test fold"))

    # Boxplots
    plot_coef_boxplot2(
        most_robust_feats_df.filter(like="Test fold").T,
        most_robust_feats_df["Prop Var Exp"],
        figsize=(15, 4),
    )
    # plot_coef_boxplot(most_robust_feats_df.filter(like="Test fold").T)
    # plot_var_exp2(most_robust_feats_df["Prop Var Exp"])

    # Correlation matrix of top features
    feat_corr = X[most_robust_feats_df["Prop Var Exp"].index].corr()
    plot_corr_matrix(feat_corr)

    # current_coefs_df.drop(columns=[c for c in most_robust_feats_df.columns if not c.startswith('Test fold')]).T.describe().T[["mean", "std", "min", "max"]].sort_values(by="mean", ascending=False)
    # Frequency stability
    (current_coefs_df.filter(like="Test fold") != 0).T.mean()

    current_coefs_df["Feature"] = current_coefs_df.index
    current_coefs_df["Prediction task"] = PREDICTION_TASK
    output_dir = MODELING_DIR / "pyradiomics"
    output_dir.mkdir(parents=True, exist_ok=True)
    current_coefs_df.to_csv(OUTPUT_DIR / f"coefs.csv", index=False)
    logging.info("\tSaved coefs.csv")

logging.info(f"Finished running {Path(__file__).name}")
logging.info(f"<>" * 40)
# %%
