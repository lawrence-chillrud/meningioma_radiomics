from pathlib import Path
from typing import Literal, Optional

import pandas as pd

from .paths import LABELS_FILE, MRIS_DIR, SEGS_DIR


def get_cohort(
    labels_file: Path = LABELS_FILE,
    mris_dir: Path = MRIS_DIR,
    segs_dir: Path = SEGS_DIR,
    outcome_of_interest: Optional[
        Literal["MethylationSubgroup", "Chr22q", "Chr1p"]
    ] = None,
):
    """
    Given a labels file, MRI directory, segmentations directory, return:
    * a list of subjects with MRI, biomarker, & segmentation data available;
    * a dataframe with the biomarker labels for the subjects with MRI, biomarker, & segmentation data available;
    * and a list of subjects with only MRI & biomarker data available (for deep learning work)

    Parameters
    ----------
    labels_file : Path
        The path to the labels file.
    mris_dir : Path
        The path to the MRI directory.
    segs_dir : Path
        The path to the segmentations directory.
    outcome_of_interest : str or None
        If None (default), drops subjects who have NaN across all outcomes. If 'MethylationSubgroup', 'Chr22q', or 'Chr1p', drops subjects with missing values in the given outcome.
    """
    # Find which subjects have biomarker labels available
    labels = pd.read_csv(labels_file)
    if outcome_of_interest:
        labels_subs = labels.dropna(subset=[outcome_of_interest])[
            "Subject Number"
        ].to_list()
    else:
        labels_subs = labels.dropna(how="all")["Subject Number"].to_list()
    labels_subs = [str(int(s)) for s in labels_subs]

    # Find which subjects have MRI data available
    mri_subjects = [p.name for p in Path(mris_dir).iterdir() if p.is_dir()]

    # Find which subjects have segmentation masks available
    segmentations = [
        p.name for p in Path(segs_dir).iterdir() if p.name.startswith("Segmentation")
    ]
    seg_subs = list(
        set(
            [
                f.split("Segmentation ")[-1].split(" ")[0].split(".nii")[0]
                for f in segmentations
            ]
        )
    )

    # Get set of subjects with MRI & label data
    mris_w_labels = list(set(mri_subjects) & set(labels_subs))

    # Get set of subjects with MRI, label, and segmentation data
    mris_w_labels_w_segs = list(set(mris_w_labels) & set(seg_subs))

    # Return labels df of those subjects w/MRI, labels, and segmentations
    have = [int(e) for e in mris_w_labels_w_segs]
    have_df = labels[labels["Subject Number"].isin(have)]

    return sorted(mris_w_labels_w_segs), have_df, sorted(mris_w_labels)
