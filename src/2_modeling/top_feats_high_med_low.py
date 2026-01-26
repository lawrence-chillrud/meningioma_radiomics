# %%
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ants import image_read
import matplotlib as mpl
from tqdm import tqdm
from itertools import product
import logging
from datetime import datetime
from pathlib import Path

from src.utils import (
    PYRAD_FILE,
    METADATA_FILE,
    MODELING_DIR,
    get_mris,
    get_segs,
    get_feats,
    translate_feat_names,
    clean_feature_names,
)

RESULTS_DIR = MODELING_DIR / "pyradiomics"
NUM_TOP_FEATS = 5

# Output dir and logfile set up
TIMESTAMP = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
OUTPUT_DIR = RESULTS_DIR / "MRI_Visualizations_Key" / f"Top_{NUM_TOP_FEATS}_Features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOGFILE = OUTPUT_DIR / "logfile.txt"

# Setup logfile
logging.basicConfig(
    filename=LOGFILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

coef_files = [f for f in RESULTS_DIR.rglob("*coefs.csv")]
testing_loop_files = [f for f in RESULTS_DIR.rglob("*testing_loop.csv")]
metadata_df = pd.read_csv(METADATA_FILE, index_col=[0])

logging.info(f"<>" * 40)
logging.info(f"Log file for {Path(__file__).name} run at {TIMESTAMP}")
logging.info(f"Settings used:")
logging.info(f"\tRESULTS_DIR: {RESULTS_DIR}")
logging.info(f"\tcoef_files: {coef_files}")
logging.info(f"\ttesting_loop_files: {testing_loop_files}")
logging.info(f"\tMETADATA_FILE: {METADATA_FILE}")

# Read in coefs files
coefs = {}
for f in coef_files:
    prediction_task = str(f).split("/")[-3]
    if prediction_task == "MethylationSubgroup":
        subtask = str(f).split("/")[-1].replace("_coefs.csv", "")
        coefs[subtask] = pd.read_csv(f, index_col=[0])
    else:
        coefs[prediction_task] = pd.read_csv(f, index_col=[0])

# %%
top_betas = (
    coefs["Chr22q"].filter(regex=r"^Test fold").median(axis=1).round(3).iloc[:5].values
)
top_feats = coefs["Chr22q"]["Feature"].iloc[:5].values
top_feats_clean = translate_feat_names(clean_feature_names(top_feats))

X, y, SUBJECTS = get_feats(
    prediction_task="Chr22q",
    features_path=PYRAD_FILE,
    scaler="None",
)
viable_idxs = X[top_feats[0]].loc[X[top_feats[0]] != 0].index
X, y, SUBJECTS = get_feats(
    prediction_task="Chr22q",
    features_path=PYRAD_FILE,
    scaler="Standard",
)
x_vals = X.iloc[viable_idxs][top_feats[0]].round(3).values
subs = SUBJECTS[viable_idxs]
cur_df = (
    pd.DataFrame(
        {"feat": top_feats_clean[0], "beta": top_betas[0], "x": x_vals, "subject": subs}
    )
    .sort_values(by="x", ascending=False)
    .reset_index(drop=True)
)
to_plot = cur_df.iloc[[0, len(cur_df) // 2, -1]]
to_plot


# %%
def rescale_linear(array: np.ndarray, new_min: int, new_max: int):
    """Rescale an array linearly."""
    minimum, maximum = np.min(array), np.max(array)
    m = (new_max - new_min) / (maximum - minimum)
    b = new_min - m * minimum
    return m * array + b


pulse_key = {"T1": "T1_POST", "DWI": "DIFFUSION", "ADC": "ADC", "FLAIR": "FLAIR"}


def make_thumbnail(
    feat,
    ax,
    reorient="IAR",
    contour_thickness=1,
):
    # parse subject we need to plot
    subject = feat.subject

    # parse the pulse and roi needed
    pulse, roi, _ = feat.feat.split("-")
    pulse = pulse_key[pulse]
    roi = int(roi)

    # get the subject's segmentation and best slice for plotting the feature
    seg = get_segs(subject, rois=roi, reorient=reorient)[roi]
    cancerous_pixels_per_slice = np.sum(seg, axis=(1, 2))
    cslice = np.argmax(cancerous_pixels_per_slice)
    seg_slice = seg[cslice, :, :]

    # get the subject's corresponding mri slice for plotting
    mri_slice = image_read(
        str(get_mris(subject, pulses=pulse)[pulse]), reorient=reorient
    ).numpy()[cslice, :, :]

    # rescale
    mri_rescaled = rescale_linear(mri_slice, 0, 1)
    seg_rescaled = rescale_linear(seg_slice, 0, 1).astype(np.uint8)

    arr_rgb = cv2.cvtColor(mri_rescaled, cv2.COLOR_GRAY2RGB)
    contours, _ = cv2.findContours(seg_rescaled, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    arr_with_contours = cv2.drawContours(
        arr_rgb, contours, -1, (0, 1, 0), contour_thickness
    )

    title = f"Subject {feat.subject}\n{feat.feat_clean} = {feat.x}"

    ax.imshow(arr_with_contours)
    ax.axis("off")
    ax.text(0.5, -0.02, title, transform=ax.transAxes, ha="center", va="top")


def visualize_feature_key(task="Chr22q", dpi=300, save=False):
    if task not in ["Chr22q", "Chr1p"]:
        pt = "MethylationSubgroup"
    else:
        pt = task

    output_fp = OUTPUT_DIR / f"{task}.png"
    if save and output_fp.exists():
        return

    top_betas = (
        coefs[task].filter(regex=r"^Test fold").median(axis=1).round(3).iloc[:5].values
    )
    top_feats = coefs[task]["Feature"].iloc[:5].values
    top_feats_clean = translate_feat_names(clean_feature_names(top_feats))

    X, _, _ = get_feats(
        prediction_task=pt,
        features_path=PYRAD_FILE,
        scaler="None",
    )

    fig, axes = plt.subplots(3, 5, figsize=(20 * 5, 20 * 3), dpi=dpi, squeeze=False)

    for i in range(len(top_feats)):
        viable_idxs = X[top_feats[i]].loc[X[top_feats[i]] != 0].index
        X_i, _, SUBJECTS = get_feats(
            prediction_task=pt,
            features_path=PYRAD_FILE,
            scaler="Standard",
        )
        x_vals = X_i.iloc[viable_idxs][top_feats[i]].round(3).values
        subs = SUBJECTS[viable_idxs]
        cur_df = (
            pd.DataFrame(
                {
                    "feat": top_feats[i],
                    "feat_clean": top_feats_clean[i],
                    "beta": top_betas[i],
                    "x": x_vals,
                    "subject": subs,
                }
            )
            .sort_values(by="x", ascending=False)
            .reset_index(drop=True)
        )
        to_plot = cur_df.iloc[[0, len(cur_df) // 2, -1]].reset_index(drop=True)

        for j in range(len(to_plot)):
            make_thumbnail(to_plot.iloc[j], axes[j, i])

    fig.suptitle(f"{task} Top 5 most important features", y=0.98)
    if save:
        plt.savefig(output_fp, dpi=dpi, bbox_inches="tight")
        plt.close()
    else:
        # plt.tight_layout()
        plt.show()
        plt.close()


FONT_SIZE = 32
mpl.rcParams.update(
    {
        "font.size": FONT_SIZE + 32,
        # "axes.titlesize": FONT_SIZE + 16,
        # "axes.labelsize": FONT_SIZE + 16,
        # "xtick.labelsize": FONT_SIZE + 16,
        "figure.titlesize": FONT_SIZE + 64,
    }
)

for k in coefs.keys():
    visualize_feature_key(task=k)
# %%

logging.info(f"Saved results to: {OUTPUT_DIR}")
logging.info(f"Finished running {Path(__file__).name}")
logging.info(f"<>" * 40)
print(f"✅ Finished running. Saved results to: {OUTPUT_DIR}")
# %%
