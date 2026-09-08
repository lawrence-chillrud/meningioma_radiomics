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
NUM_TOP_FEATS = 4
NUM_CANDIDATES = 20

# Output dir and logfile set up
TIMESTAMP = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
OUTPUT_DIR = RESULTS_DIR / "MRI_Visualizations" / f"Top_{NUM_TOP_FEATS}_Features"
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
coefs = {
    "MethylationSubgroup": {
        "Merlin Intact": {},
        "Immune Enriched": {},
        "Hypermetabolic": {},
    },
    "Chr1p": None,
    "Chr22q": None,
}
for f in coef_files:
    prediction_task = str(f).split("/")[-3]
    if prediction_task == "MethylationSubgroup":
        subtask = str(f).split("/")[-1].replace("_coefs.csv", "")
        coefs[prediction_task][subtask] = pd.read_csv(f, index_col=[0])
    else:
        coefs[prediction_task] = pd.read_csv(f, index_col=[0])

# Read in test loop files
test_loops = {}
for f in testing_loop_files:
    prediction_task = str(f).split("/")[-3]
    df = pd.read_csv(f)
    y_probs = np.vstack(
        [np.fromstring(x.strip("[]"), sep=" ") for x in df["y_probs"].values]
    )
    df["y_probs_1"] = y_probs[:, 1]
    df["TP"] = (df["y_true"] == df["y_pred"]) & (df["y_true"] == 1)
    df["TN"] = (df["y_true"] == df["y_pred"]) & (df["y_true"] == 0)
    df["FP"] = (df["y_true"] != df["y_pred"]) & (df["y_pred"] == 1)
    df["FN"] = (df["y_true"] != df["y_pred"]) & (df["y_pred"] == 0)
    test_loops[prediction_task] = df

# Find top TP, TN, FP, FN candidate subjects for Chr22q, Chr1p
top_candidates = {}
for k in ["Chr22q", "Chr1p"]:
    df = test_loops[k]
    top_candidates[k] = {}
    for p in ["TP", "TN", "FP", "FN"]:
        top_candidates[k][p] = {}
        top_candidates[k][p] = (
            df[df[p] == True]
            .sort_values(by="y_probs_1", ascending=("N" in p))
            .head(NUM_CANDIDATES)[
                ["Subject Number", "test_idx", "y_probs_1", "y_pred", "y_true"]
            ]
            .reset_index(drop=True)
        )

# Find top correct, incorrect candidate subjects for MethylationSubgroup sub tasks
top_candidates["MethylationSubgroup"] = {}
for y, subtask in enumerate(["Merlin Intact", "Immune Enriched", "Hypermetabolic"]):
    top_candidates["MethylationSubgroup"][subtask] = {}
    df = test_loops["MethylationSubgroup"]
    df_filter = df[df["y_true"] == y].sort_values(by="test_loss", ascending=True)
    top_candidates["MethylationSubgroup"][subtask]["correct"] = (
        df_filter[df_filter["y_pred"] == y]
        .head(NUM_CANDIDATES)[
            ["Subject Number", "test_idx", "y_probs", "y_pred", "y_true", "test_loss"]
        ]
        .reset_index(drop=True)
    )
    top_candidates["MethylationSubgroup"][subtask]["incorrect"] = (
        df_filter[df_filter["y_pred"] != y]
        .tail(NUM_CANDIDATES)
        .sort_values(by="test_loss", ascending=False)[
            ["Subject Number", "test_idx", "y_probs", "y_pred", "y_true", "test_loss"]
        ]
        .reset_index(drop=True)
    )


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

    beta = feat.beta.round(2)
    x = feat.x.round(2)
    product = feat.betax.round(2)
    cont = feat.contribution.round(2)
    fancyx = r"$x$"
    title = f"{feat.feat_clean}\nβ={beta}, {fancyx}={x}, β{fancyx}={product}\nInfluence={cont}"

    ax.imshow(arr_with_contours)
    ax.axis("off")
    ax.text(0.5, -0.02, title, transform=ax.transAxes, ha="center", va="top")


def visualize_features(
    task="Chr22q", pred_type="TP", num_feats=NUM_TOP_FEATS, dpi=600, save=True
):
    class_ids = ["Intact", "Lost"]

    for i in tqdm(
        range(len(top_candidates[task][pred_type])),
        total=len(top_candidates[task][pred_type]),
        desc="Inner loop",
        position=1,
        leave=False,
        ncols=120,
    ):
        try:
            output_fp = OUTPUT_DIR / f"{task}_{pred_type}-{i}.png"
            if save and output_fp.exists():
                continue
            c_tc = top_candidates[task][pred_type].iloc[i]
            c_subject = c_tc["Subject Number"].astype(int)
            c_test_fold_num = c_tc["test_idx"].astype(int).item() + 1
            c_y_pred = c_tc["y_pred"].astype(int)
            c_y_true = c_tc["y_true"].astype(int)
            c_coefs = coefs[task][f"Test fold {c_test_fold_num}"]
            c_feats = clean_feature_names(c_coefs.index.values)
            c_betas = c_coefs.values
            X, _, sub_nos = get_feats(
                prediction_task=task,
                features_path=PYRAD_FILE,
            )
            X["subject"] = sub_nos
            X = X.sort_values(by="subject")
            X.index = X["subject"]
            c_image_feat_vals = X.loc[c_subject][c_feats]
            c_df = pd.DataFrame(
                {
                    "subject": c_subject,
                    "feat": c_feats,
                    "beta": c_betas,
                    "x": c_image_feat_vals,
                }
            )
            c_df["betax"] = c_df["beta"] * c_df["x"]
            c_df["contribution"] = c_df["betax"].abs() / c_df["betax"].abs().sum()
            c_df["feat_clean"] = translate_feat_names(c_df["feat"])
            c_top_feats = c_df.head(num_feats)

            fig, axes = plt.subplots(
                1, num_feats, figsize=(20 * num_feats, 20), dpi=dpi, squeeze=False
            )

            for j, ax in enumerate(axes[0]):
                make_thumbnail(c_top_feats.iloc[j], ax)

            subject_details = metadata_df.loc[c_subject]
            sex = subject_details.Sex
            age = subject_details.Age
            ethnicity = subject_details.Ethnicity
            eth = f", {ethnicity}" if isinstance(ethnicity, str) else ""
            deets_str = f"Subject {c_subject} [{sex}{age}{eth}], {task}: {class_ids[c_y_true]} ({float(c_y_true)}), Predicted: {class_ids[c_y_pred]} ({float(c_y_pred)})"
            fig.suptitle(deets_str, y=0.98)
            if save:
                plt.savefig(output_fp, dpi=dpi, bbox_inches="tight")
                plt.close()
            else:
                # plt.tight_layout()
                plt.show()
                plt.close()
        except KeyError:
            logging.exception(
                f"Error for task={task}, pred_type={pred_type}, subject={c_subject}"
            )


def visualize_methylation_features(
    pred_type="correct", num_feats=NUM_TOP_FEATS, dpi=600, save=True
):
    top_candidates_df = top_candidates["MethylationSubgroup"]
    coefs_df = coefs["MethylationSubgroup"]
    class_ids = ["Merlin Intact", "Immune Enriched", "Hypermetabolic"]
    for c in tqdm(
        class_ids, total=3, desc="Subtype loop", position=1, leave=False, ncols=120
    ):
        for i in tqdm(
            range(len(top_candidates_df[c][pred_type])),
            total=len(top_candidates_df[c][pred_type]),
            desc="Inner loop",
            position=2,
            leave=False,
            ncols=120,
        ):
            try:
                output_fp = OUTPUT_DIR / f"{c}_{pred_type}-{i}.png"
                if save and output_fp.exists():
                    continue
                c_tc = top_candidates_df[c][pred_type].iloc[i]
                c_subject = c_tc["Subject Number"].astype(int)
                c_test_fold_num = c_tc["test_idx"].astype(int).item() + 1
                c_y_pred = c_tc["y_pred"].astype(int)
                c_y_true = c_tc["y_true"].astype(int)
                c_coefs = coefs_df[c][f"Test fold {c_test_fold_num}"]
                c_feats = clean_feature_names(c_coefs.index.values)
                c_betas = c_coefs.values
                X, _, sub_nos = get_feats(
                    prediction_task="MethylationSubgroup",
                    features_path=PYRAD_FILE,
                )
                X["subject"] = sub_nos
                X.index = X["subject"]
                c_image_feat_vals = X.loc[c_subject][c_feats]
                c_df = pd.DataFrame(
                    {
                        "subject": c_subject,
                        "feat": c_feats,
                        "beta": c_betas,
                        "x": c_image_feat_vals,
                    }
                )
                c_df["betax"] = c_df["beta"] * c_df["x"]
                c_df["contribution"] = c_df["betax"].abs() / c_df["betax"].abs().sum()
                c_df["feat_clean"] = translate_feat_names(c_df["feat"])
                c_top_feats = c_df.head(num_feats)

                fig, axes = plt.subplots(
                    1, num_feats, figsize=(20 * num_feats, 20), dpi=dpi, squeeze=False
                )

                for j, ax in enumerate(axes[0]):
                    make_thumbnail(c_top_feats.iloc[j], ax)

                subject_details = metadata_df.loc[c_subject]
                sex = subject_details.Sex
                age = subject_details.Age
                ethnicity = subject_details.Ethnicity
                eth = f", {ethnicity}" if isinstance(ethnicity, str) else ""
                deets_str = f"Subject {c_subject} [{sex}{age}{eth}], Methylation Subgroup: {class_ids[c_y_true]} ({float(c_y_true)}), Predicted: {class_ids[c_y_pred]} ({float(c_y_pred)})"
                fig.suptitle(deets_str, y=0.98)
                if save:
                    plt.savefig(output_fp, dpi=dpi, bbox_inches="tight")
                    plt.close()
                else:
                    # plt.tight_layout()
                    plt.show()
                    plt.close()
            except KeyError:
                logging.exception(
                    f"Error for task={c}, pred_type={pred_type}, subject={c_subject}"
                )


# %%
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

# MethylationSubgroup images
pred_types = ["correct", "incorrect"]
for p in tqdm(
    pred_types,
    total=len(pred_types),
    desc="Outer loop",
    position=0,
    ncols=120,
):
    visualize_methylation_features(p)


# tasks = ["Chr22q", "Chr1p"]
# pred_types = ["TP", "TN", "FP", "FN"]
# for t, p in tqdm(
#     product(tasks, pred_types),
#     total=(len(tasks) * len(pred_types)),
#     desc="Outer loop",
#     position=0,
#     ncols=120,
# ):
#     visualize_features(t, p)

logging.info(f"Saved results to: {OUTPUT_DIR}")
logging.info(f"Finished running {Path(__file__).name}")
logging.info(f"<>" * 40)
print(f"✅ Finished running. Saved results to: {OUTPUT_DIR}")
# %%
