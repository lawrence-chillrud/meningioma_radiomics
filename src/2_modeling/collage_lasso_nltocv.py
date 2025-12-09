"""
In the nested leave-two-out cross-validation (NLTOCV) procedure, we want to run the following pseudocode:

# Outer loop constructing independent test split aiming to give unbiased estimate of model generalizability
for test_sample in dataset:
    lambda_performances = []
    # Inner loop over lambdas for model selection step
    for lambda_i in lambdas:
        val_hats = []
        # Innermost loop constructing independent val split
        for val_sample in dataset - test_sample
            train_dataset = dataset - test_sample - val_sample
            model.fit(train_dataset, lambda_i)
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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from tqdm import tqdm

from src.utils import COLLAGE_DIR, MODELING_DIR, get_feats
from src.utils.plotting import *

MAX_WORKERS = 16
FEATURES_PATHS = [f for f in COLLAGE_DIR.rglob("*wide-features*.csv")]
HARALICK_WINDOW_SIZES = [3, 5, 7, 9]
BIN_SIZES = [16, 32, 48, 64]
PREDICTION_TASK = (
    "MethylationSubgroup"  # can be one of "MethylationSubgroup", "Chr22q", or "Chr1p"
)
SCALER = "Standard"  # can be one of "Standard", "MinMax", or None
LAMBDAS = np.linspace(0.05, 0.35, 10).round(2)
LR_PARAMS = {
    "penalty": "l1",
    "class_weight": "balanced",
    "solver": "liblinear",
    "random_state": 0,
    "max_iter": 100,
    "verbose": 0,
}

Xs, ys, SUBJECTS = {}, {}, {}
for f in tqdm(
    FEATURES_PATHS,
    total=len(FEATURES_PATHS),
    desc="Reading in all collage files...",
    ncols=120,
):
    fname = " ".join(f.name.replace(".csv", "").split("_")[1:])  # e.g. 'win-7 bin-48'
    Xs[fname], ys[fname], SUBJECTS[fname] = get_feats(
        prediction_task=PREDICTION_TASK, features_path=f, scaler=SCALER
    )

N = len(
    Xs["win-7 bin-48"]
)  # doesn't actually matter which one, across all Xs, N is the same
N_CLASSES = len(set(ys["win-7 bin-48"]))
CLASS_IDS = ["Intact", "Lost"]
if N_CLASSES == 3:
    CLASS_IDS = ["Merlin Intact", "Immune Enriched", "Hypermetabolic"]


def val_job(test_idx, val_idx, lambda_i, win_size, bin_size):
    X = Xs[f"win-{win_size} bin-{bin_size}"]
    y = ys[f"win-{win_size} bin-{bin_size}"]

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
        win_size=win_size,
        bin_size=bin_size,
        train_loss=train_loss,
        val_loss=val_loss,
    )


val_loop = [
    (n, m, lambda_i, win_size, bin_size)
    for n, m, lambda_i, win_size, bin_size in product(
        range(N), range(N), LAMBDAS, HARALICK_WINDOW_SIZES, BIN_SIZES
    )
    if n != m
]

# %%
results = Parallel(n_jobs=MAX_WORKERS, backend="loky", verbose=0)(
    delayed(val_job)(test_idx, val_idx, lambda_i, win_size, bin_size)
    for test_idx, val_idx, lambda_i, win_size, bin_size in tqdm(
        val_loop, total=len(val_loop), ncols=120, desc="Step 1/2: Model selection loops"
    )
)

df = pd.DataFrame(results)
df.to_csv(f"{PREDICTION_TASK}_collage_gridsearch.csv", index=False)

# %%
df = pd.read_csv(f"{PREDICTION_TASK}_collage_gridsearch.csv")

# %%
val_summary = (
    df.groupby(["test_idx", "lambda_i", "win_size", "bin_size"])[
        ["train_loss", "val_loss"]
    ]
    .mean()
    .reset_index()
)
val_summary = val_summary.rename(
    columns={"train_loss": "Training", "val_loss": "Validation"}
)
val_summary_long = val_summary.melt(
    id_vars=["test_idx", "lambda_i", "win_size", "bin_size"],
    var_name="Dataset split",
    value_name="loss",
)

if val_summary["test_idx"].nunique() > 1:
    best_hyperparams = (
        val_summary_long[val_summary_long["Dataset split"] == "Validation"]
        .loc[lambda df: df.groupby(["test_idx"])["loss"].idxmin()]
        .reset_index(drop=True)
    )
else:
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
        annot=np.array(collage_stats["lambda_i"]).reshape(4, 4),
    )
    plt.xlabel("Bin size")
    plt.ylabel("Window size")
    plt.show()
    plt.close()

# sns.relplot(val_summary_long, kind="line", x="lambda_i", y="loss", style="Dataset split", col="win_size", row="bin_size")


# %%
def test_job(test_idx):
    test_lambda = best_hyperparams["lambda_i"][test_idx]
    test_win_size = best_hyperparams["win_size"][test_idx]
    test_bin_size = best_hyperparams["bin_size"][test_idx]

    X = Xs[f"win-{test_win_size} bin-{test_bin_size}"]
    y = ys[f"win-{test_win_size} bin-{test_bin_size}"]

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
    delayed(test_job)(i)
    for i in tqdm(range(N), total=N, ncols=120, desc="Step 2/2: Test loop")
)
outer_df = pd.DataFrame(outer_results)
test_coefs = np.stack(outer_df.coefs)

# %%
if N_CLASSES == 3:
    plot_multiclass_results(
        outer_df["y_probs"], outer_df["y_true"], CLASS_IDS, PREDICTION_TASK
    )
else:
    plot_binary_results(outer_df["y_probs"], outer_df["y_true"], CLASS_IDS)

# %%
test_coefs = test_coefs.squeeze()
if len(test_coefs.shape) == 3:
    for c in range(test_coefs.shape[1]):
        current_model = test_coefs[:, c, :]
        nonzero_feats_idxs = np.nonzero(np.sum(current_model, axis=0))[0]
        current_coefs = current_model[:, nonzero_feats_idxs]
        current_coefs_df = pd.DataFrame(
            current_coefs, columns=Xs["win-9 bin-64"].columns[nonzero_feats_idxs]
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
        most_robust_feats_df = current_coefs_df[current_coefs_df["Cum Var Exp"] < 0.95]

        # Heatmap
        plot_heatmap(most_robust_feats_df.filter(like="Test fold"))

        # Boxplots
        plot_coef_boxplot(most_robust_feats_df.filter(like="Test fold").T)

        # Var explained
        plot_var_exp(most_robust_feats_df["Prop Var Exp"])

        # Correlation matrix of top features
        feat_corr = Xs["win-9 bin-64"][
            most_robust_feats_df["Prop Var Exp"].index
        ].corr()
        plot_corr_matrix(feat_corr)

        # current_coefs_df.drop(columns=[c for c in most_robust_feats_df.columns if not c.startswith('Test fold')]).T.describe().T[["mean", "std", "min", "max"]].sort_values(by="mean", ascending=False)
        # Frequency stability
        (current_coefs_df.filter(like="Test fold") != 0).T.mean()

        current_coefs_df["Feature"] = current_coefs_df.index
        current_coefs_df["Prediction task"] = CLASS_IDS[c]
        output_dir = MODELING_DIR / "pyradiomics"
        output_dir.mkdir(parents=True, exist_ok=True)
        current_coefs_df.to_csv(
            MODELING_DIR / "pyradiomics" / f"{CLASS_IDS[c]}_coefs.csv", index=False
        )

else:
    nonzero_feats_idxs = np.nonzero(np.sum(test_coefs, axis=0))[0]
    current_coefs = test_coefs[:, nonzero_feats_idxs]
    current_coefs_df = pd.DataFrame(
        current_coefs, columns=Xs["win-9 bin-64"].columns[nonzero_feats_idxs]
    ).T
    current_coefs_df.columns = [f"Test fold {i + 1}" for i in range(N)]
    current_coefs_df["Absolute Sum"] = current_coefs_df.abs().sum(axis=1)
    current_coefs_df = current_coefs_df.sort_values(by="Absolute Sum", ascending=False)
    current_coefs_df["Prop Var Exp"] = (
        current_coefs_df["Absolute Sum"] / current_coefs_df["Absolute Sum"].sum()
    )
    current_coefs_df["Cum Var Exp"] = current_coefs_df["Prop Var Exp"].cumsum()
    most_robust_feats_df = current_coefs_df[current_coefs_df["Cum Var Exp"] < 0.95]

    # Heatmap
    plot_heatmap(most_robust_feats_df.filter(like="Test fold"))

    # Boxplots
    plot_coef_boxplot(most_robust_feats_df.filter(like="Test fold").T)

    # Var explained
    plot_var_exp(most_robust_feats_df["Prop Var Exp"])

    # Correlation matrix of top features
    feat_corr = Xs["win-9 bin-64"][most_robust_feats_df["Prop Var Exp"].index].corr()
    plot_corr_matrix(feat_corr)

    # current_coefs_df.drop(columns=[c for c in most_robust_feats_df.columns if not c.startswith('Test fold')]).T.describe().T[["mean", "std", "min", "max"]].sort_values(by="mean", ascending=False)
    # Frequency stability
    (current_coefs_df.filter(like="Test fold") != 0).T.mean()

    current_coefs_df["Feature"] = current_coefs_df.index
    current_coefs_df["Prediction task"] = PREDICTION_TASK
    output_dir = MODELING_DIR / "pyradiomics"
    output_dir.mkdir(parents=True, exist_ok=True)
    current_coefs_df.to_csv(
        MODELING_DIR / "pyradiomics" / f"{PREDICTION_TASK}_coefs.csv", index=False
    )

# %%
from src.utils import MRIS_DIR

incorrect_subjects = SUBJECTS[
    f"win-{best_hyperparams[['win_size', 'bin_size']].value_counts().index[0][0]} bin-{best_hyperparams[['win_size', 'bin_size']].value_counts().index[0][1]}"
][outer_df[outer_df["y_true"] != outer_df["y_pred"]].test_idx.to_list()]
mri_filepaths = MRIS_DIR.rglob("*Presurgical*")

sessions = {}
for subject in MRIS_DIR.iterdir():
    if subject.is_dir():
        for session in (MRIS_DIR / subject).iterdir():
            if session.is_dir():
                sessions[int(subject.name)] = " ".join(session.name.split("_")[1:])

incorrect_sessions = []
for s in incorrect_subjects:
    incorrect_sessions.append(sessions[s])
print("INCORRECT SESSIONS:\n", pd.Series(incorrect_sessions).value_counts())

print("TOTAL SESSIONS:\n", pd.Series(sessions.values()).value_counts())

# %%
