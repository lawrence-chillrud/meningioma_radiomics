# %% Imports
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
import matplotlib.pyplot as plt

from src.utils.plotting import *
from src.utils import MODELING_DIR

# User defined variables to select proper experiment
PREDICTION_TASK = "Chr1p"  # can be one of "MethylationSubgroup", "Chr22q", or "Chr1p"
FEATURE_SET = "pyradiomics"  # can be one of "pyradiomics" or "collage"
EXP_DIRS = sorted(
    [d for d in (MODELING_DIR / FEATURE_SET / PREDICTION_TASK).iterdir() if d.is_dir()]
)


def get_val_loss(i):
    EXP_DIR = EXP_DIRS[i]

    # Read in all experiment metadata and results
    run_metadata_df = pd.read_csv(EXP_DIR / "run_metadata_df.csv")
    CORRELATED_FEATS_THRESH = (
        run_metadata_df["CORRELATED_FEATS_THRESH"].values[0]
        if not run_metadata_df["CORRELATED_FEATS_THRESH"].isna().values[0]
        else None
    )
    val_loop = pd.read_csv(EXP_DIR / "validation_loop.csv")
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

    min_val_loss = df_val_summary_agg[
        df_val_summary_agg["Dataset split"] == "Validation"
    ].mean_loss.min()

    return {
        "corr_feat_thresh": CORRELATED_FEATS_THRESH,
        "val_loss": min_val_loss,
        "i": i,
    }


val_losses = []
for i in range(-8, 0):
    val_losses.append(get_val_loss(i))

df = (
    pd.DataFrame(val_losses).sort_values(by=["corr_feat_thresh"]).reset_index(drop=True)
)

fig, ax = plt.subplots()
df.plot(kind="line", x="corr_feat_thresh", y="val_loss", ax=ax)
df.plot(kind="scatter", x="corr_feat_thresh", y="val_loss", ax=ax)
fig.suptitle(PREDICTION_TASK)
plt.show()
plt.close()

df
# %%
