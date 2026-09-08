# %%
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from ants import image_read
import matplotlib as mpl

from src.utils import (
    PYRAD_FILE,
    METADATA_FILE,
    MODELING_DIR,
    get_mris,
    get_segs,
    get_feats,
    translate_feat_names,
    clean_feature_names,
    get_segs_roi_key
)

FONT_SIZE = 18
mpl.rcParams.update(
    {
        "font.size": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE, # for MethylationSubgroup tasks,
        "figure.titlesize": FONT_SIZE,
    }
)

RESULTS_DIR = MODELING_DIR / "pyradiomics"
OUTPUT_DIR = RESULTS_DIR / "Final_MRI_Figs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PRED_CSVS = {
    "Chr22q": RESULTS_DIR / "Chr22q" / "02-25-2026_17-18-03" / "testing_loop.csv",
    "Chr1p": RESULTS_DIR / "Chr1p" / "02-25-2026_17-32-30" / "testing_loop.csv",
    "MethylationSubgroup": RESULTS_DIR / "MethylationSubgroup" / "02-24-2026_17-27-31" / "testing_loop.csv"
}
TASKS = {
    "Chr22q": ['Intact', 'Lost', 'N/A'], 
    "Chr1p": ['Intact', 'Lost', 'N/A'], 
    "MethylationSubgroup": ['Merlin Intact', 'Immune Enriched', 'Hypermitotic', 'N/A']
}
SEGS_ROI_KEY = get_segs_roi_key()
PULSE_KEY = {"CET1": "T1_POST", "DWI": "DIFFUSION", "ADC": "ADC", "FLAIR": "FLAIR"}

def reverse_translate_feat(feat_name):
    roi, pulse = None, None
    for k, v in SEGS_ROI_KEY.items():
        if v.upper() in feat_name:
            roi = k
    if " on " in feat_name:
        pulse = PULSE_KEY[feat_name.split(" on ")[-1]]
    else:
        pulse = "T1_POST"
    return int(roi), pulse

def get_model_prediction(subject, biomarker):
    df = pd.read_csv(PRED_CSVS[biomarker]).set_index("Subject Number")
    pred = int(df["y_pred"].loc[subject])
    return TASKS[biomarker][pred]

def rescale_linear(array: np.ndarray, new_min: int, new_max: int):
    """Rescale an array linearly."""
    minimum, maximum = np.min(array), np.max(array)
    m = (new_max - new_min) / (maximum - minimum)
    b = new_min - m * minimum
    return m * array + b

def make_thumbnail(
    subject,
    feat,
    slice_num,
    ax,
    reorient="IAR",
    contour_thickness=1,
    thumbnail_label=None,
    tasks_to_care_about=TASKS.keys(),
):
    X, _, sub_nos = get_feats(
        prediction_task=tasks_to_care_about[-1],
        features_path=PYRAD_FILE,
        scaler="None",
        remove_correlated_feats=None,
    )
    X.columns = translate_feat_names(clean_feature_names(X.columns))
    X["Subject Number"] = sub_nos
    X = X.set_index("Subject Number")

    def make_sub_name(row, tasks_to_care_about=tasks_to_care_about):
        sub_id = f"Subject {row['Subject Number']} ({row['Sex']}{row['Age']}, {row['Ethnicity']})"
        for biomarker in TASKS:
            if np.isnan(row[biomarker]):
                row[biomarker] = -1
        sub_profile = []
        padding = []
        for t in tasks_to_care_about:
            if t != "MethylationSubgroup":
                sub_profile.append(f"{t} {TASKS[t][int(row['Chr22q'])]}")
                padding.append(len(t) + 7)
            else:
                sub_profile.append(f"{TASKS[t][int(row['MethylationSubgroup'])]}")
                padding.append(15)
        combined_profile = "Biomarkers: " + ", ".join(t.rjust(pad) for t, pad in zip(sub_profile, padding))
        return f"{sub_id}\n{combined_profile}"

    LABELS_DF = pd.read_csv(METADATA_FILE)
    LABELS_DF['sub_name'] = LABELS_DF.apply(make_sub_name, axis=1)
    LABELS_DF = LABELS_DF.set_index("Subject Number")

    if not thumbnail_label:
        sub_name = LABELS_DF["sub_name"].loc[subject].replace(", nan", "")
        feat_val = X[feat].loc[subject]
        sub_preds = []
        padding = []
        for t in tasks_to_care_about:
            if t == "MethylationSubgroup":
                sub_preds.append(f"{get_model_prediction(subject, t)}")
                padding.append(15)
            else:
                sub_preds.append(f"{t} {get_model_prediction(subject, t)}")
                padding.append(len(t) + 7)
        combined_sub_preds = "Predicted: ".rjust(12) + ", ".join(t.rjust(pad) for t, pad in zip(sub_preds, padding))
        thumbnail_label = f"{sub_name}\n{combined_sub_preds}\n{feat}: {round(feat_val, 2)}"

    # parse the pulse and roi needed
    roi, pulse = reverse_translate_feat(feat)

    # get the subject's segmentation and best slice for plotting the feature
    seg = get_segs(subject, rois=roi, reorient=reorient)[roi]
    if not slice_num:
        cancerous_pixels_per_slice = np.sum(seg, axis=(1, 2))
        slice_num = np.argmax(cancerous_pixels_per_slice)
    
    seg_slice = seg[slice_num, :, :]

    # get the subject's corresponding mri slice for plotting
    mri_slice = image_read(
        str(get_mris(subject, pulses=pulse)[pulse]), reorient=reorient
    ).numpy()[slice_num, :, :]

    # rescale
    mri_rescaled = rescale_linear(mri_slice, 0, 1)
    seg_rescaled = rescale_linear(seg_slice, 0, 1).astype(np.uint8)

    arr_rgb = cv2.cvtColor(mri_rescaled, cv2.COLOR_GRAY2RGB)
    contours, _ = cv2.findContours(seg_rescaled, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    arr_with_contours = cv2.drawContours(
        arr_rgb, contours, -1, (0, 1, 0), contour_thickness
    )

    ax.imshow(arr_with_contours)
    ax.axis("off")
    # ax.set_xlabel(thumbnail_label) 
    ax.text(0.5, -0.02, thumbnail_label, transform=ax.transAxes, ha="center", va="top", fontfamily="monospace")
    print(thumbnail_label)
    print()

def make_plot(subjects, feats, slices=None, labels=None, tasks_to_care_about=TASKS.keys(), figsize=(20, 12), output_fp=None):
    if not labels:
        labels = [None] * len(subjects)
    if len(feats) != len(subjects):
        feats = feats * len(subjects)
    if not slices:
        slices = [None] * len(subjects)
    
    fig, axes = plt.subplots(nrows=1, ncols=len(subjects), figsize=figsize, sharey=True)
    for i, (subject, feat, s, label) in enumerate(zip(subjects, feats, slices, labels)):
        make_thumbnail(subject, feat, s, axes[i], thumbnail_label=label, tasks_to_care_about=tasks_to_care_about)
    
    plt.tight_layout()
    if output_fp:
        fig.savefig(output_fp, dpi=600, bbox_inches='tight')
    else:
        plt.show()
        plt.close()

# %%
current_feat = "ENH Skewness on CET1"
tasks_oi = ["Chr22q", "Chr1p", "MethylationSubgroup"]

output_fp = OUTPUT_DIR / "correct" / f"{current_feat}.png"
output_fp.parent.mkdir(parents=True, exist_ok=True)
make_plot(subjects=[28, 76], feats=[current_feat], slices=[129, 37], output_fp=output_fp, tasks_to_care_about=tasks_oi)

output_fp = OUTPUT_DIR / "erroneous" / f"{current_feat}.png"
output_fp.parent.mkdir(parents=True, exist_ok=True)
make_plot(subjects=[37, 47], feats=[current_feat], slices=[63, 54], output_fp=output_fp, tasks_to_care_about=tasks_oi)

# %%
current_feat = "ENH Kurtosis on DWI"
tasks_oi = ["Chr22q", "Chr1p", "MethylationSubgroup"]

output_fp = OUTPUT_DIR / "correct" / f"{current_feat}.png"
make_plot(subjects=[58, 31], feats=[current_feat], slices=[54, 117], output_fp=output_fp, tasks_to_care_about=tasks_oi)

output_fp = OUTPUT_DIR / "erroneous" / f"{current_feat}.png"
make_plot(subjects=[34, 52], feats=[current_feat], slices=[None, 65], output_fp=output_fp, tasks_to_care_about=tasks_oi)

# %%
current_feat = "TMR ZoneEntropy on ADC"
tasks_oi = ["Chr1p"]

output_fp = OUTPUT_DIR / "correct" / f"{current_feat}.png"
make_plot(subjects=[83, 41], feats=[current_feat], slices=[127, 73], output_fp=output_fp, tasks_to_care_about=tasks_oi)

output_fp = OUTPUT_DIR / "erroneous" / f"{current_feat}.png"
make_plot(subjects=[77, 32], feats=[current_feat], slices=[88, 104], output_fp=output_fp, tasks_to_care_about=tasks_oi)

# %%
current_feat = "ENH & RSTDFF Min on ADC"
tasks_oi = ["MethylationSubgroup"]

output_fp = OUTPUT_DIR / "correct" / f"{current_feat}.png"
make_plot(subjects=[72, 16], feats=[current_feat], slices=[None, 82], output_fp=output_fp, tasks_to_care_about=tasks_oi)

output_fp = OUTPUT_DIR / "erroneous" / f"{current_feat}.png"
make_plot(subjects=[24, 74], feats=[current_feat], slices=[None, 144], output_fp=output_fp, tasks_to_care_about=tasks_oi)

# %%
current_feat = "ENH LeastAxisLength"
tasks_oi = ["Chr22q", "MethylationSubgroup"]

output_fp = OUTPUT_DIR / "correct" / f"{current_feat}.png"
make_plot(subjects=[105, 73], feats=[current_feat], output_fp=output_fp, tasks_to_care_about=tasks_oi)

output_fp = OUTPUT_DIR / "erroneous" / f"{current_feat}.png"
make_plot(subjects=[52, 65], feats=[current_feat], output_fp=output_fp, tasks_to_care_about=tasks_oi)

# %%
