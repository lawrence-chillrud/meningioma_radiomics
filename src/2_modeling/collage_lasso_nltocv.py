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

# %% Imports
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from itertools import product

from datetime import datetime
import logging
import numpy as np
from pathlib import Path
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from tqdm import tqdm

from src.utils import (
    LABELS_FILE,
    METADATA_FILE,
    COLLAGE_DIR,
    MODELING_DIR,
    get_feats,
)
from src.utils.plotting import *

# User defined settings
PREDICTION_TASK = "Chr22q"  # can be one of "MethylationSubgroup", "Chr22q", or "Chr1p"
SCALER = "Standard"  # can be one of "Standard", "MinMax", or None
LOW_VAR_THRESH = None  # or 0.2?
MAX_WORKERS = 16
FEATURES_PATHS = [f for f in COLLAGE_DIR.rglob("*wide-features*.csv")]
HARALICK_WINDOW_SIZES = [3, 5, 7, 9]
BIN_SIZES = [16, 32, 48, 64]
LAMBDAS = np.linspace(0.05, 0.35, 10)
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
OUTPUT_DIR = MODELING_DIR / "collage" / PREDICTION_TASK / TIMESTAMP
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
Xs, ys, SUBJECTS = {}, {}, {}
for f in tqdm(
    FEATURES_PATHS,
    total=len(FEATURES_PATHS),
    desc="Reading in all collage files...",
    ncols=120,
):
    fname = " ".join(f.name.replace(".csv", "").split("_")[1:])  # e.g. 'win-7 bin-48'
    Xs[fname], ys[fname], SUBJECTS[fname] = get_feats(
        prediction_task=PREDICTION_TASK,
        features_path=f,
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
        "COLLAGE_DIR": COLLAGE_DIR,
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
N = len(
    Xs["win-7 bin-48"]
)  # doesn't actually matter which one, across all Xs, N is the same
N_CLASSES = len(set(ys["win-7 bin-48"]))
CLASS_IDS = ["Intact", "Lost"]
if N_CLASSES == 3:
    CLASS_IDS = ["Merlin Intact", "Immune Enriched", "Hypermetabolic"]


# Run validation loop, save results
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

results = Parallel(n_jobs=MAX_WORKERS, backend="loky", verbose=0)(
    delayed(val_job)(test_idx, val_idx, lambda_i, win_size, bin_size)
    for test_idx, val_idx, lambda_i, win_size, bin_size in tqdm(
        val_loop, total=len(val_loop), ncols=120, desc="Step 1/2: Model selection loops"
    )
)

df = pd.DataFrame(results)
df.to_csv(OUTPUT_DIR / "validation_loop.csv", index=False)
logging.info(
    "Step 1/2 complete. Validation loop results stored in: validation_loop.csv"
)

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

best_hyperparams = (
    val_summary_long[val_summary_long["Dataset split"] == "Validation"]
    .loc[lambda df: df.groupby(["test_idx"])["loss"].idxmin()]
    .reset_index(drop=True)
)
best_hyperparams.to_csv(OUTPUT_DIR / "best_lambdas.csv")
logging.info("\tSaved best_lambdas.csv")
best_hyperparams.value_counts().to_csv(OUTPUT_DIR / "best_lambdas_counts.csv")
logging.info("\tSaved best_lambdas_counts.csv")


# Run testing loop, save reults
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
outer_df["Subject Number"] = SUBJECTS
outer_df.to_csv(OUTPUT_DIR / "testing_loop.csv", index=False)
logging.info("Step 2/2 complete. Testing loop results stored in: testing_loop.csv")
test_coefs = np.stack(outer_df.coefs)

# Get test set performance metrics, save
if N_CLASSES == 3:
    metrics = plot_multiclass_results(
        outer_df["y_probs"], outer_df["y_true"], ["MI", "IE", "HM"], PREDICTION_TASK
    )
else:
    metrics = plot_binary_results(outer_df["y_probs"], outer_df["y_true"], CLASS_IDS)

pd.DataFrame(metrics, index=[0]).to_csv(OUTPUT_DIR / "testing_metrics.csv", index=False)
logging.info("\tTesting metrics saved to: testing_metrics.csv")

# Get coefs, save
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
        current_coefs_df["Feature"] = current_coefs_df.index
        current_coefs_df["Prediction task"] = CLASS_IDS[c]
        current_coefs_df.to_csv(OUTPUT_DIR / f"{CLASS_IDS[c]}_coefs.csv")
        logging.info(f"\tSaved {CLASS_IDS[c]}_coefs.csv")

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
    current_coefs_df["Feature"] = current_coefs_df.index
    current_coefs_df["Prediction task"] = PREDICTION_TASK
    current_coefs_df.to_csv(OUTPUT_DIR / "coefs.csv")
    logging.info("\tSaved coefs.csv")

logging.info(f"Finished running {Path(__file__).name}")
logging.info(f"<>" * 40)
