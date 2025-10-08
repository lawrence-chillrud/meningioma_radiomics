from pathlib import Path
from typing import List, Union

import numpy as np
from ants import image_read

from .paths import ROIS, SEGS_DIR


def get_segs_roi_key():
    """Returns a dict mapping int keys to corresponding segmentation rois."""
    return {
        1: "Enhancing tumor",
        2: "Other tumor",
        3: "Necrotic tumor",
        4: "Edema",
        5: "Susceptibility",
        6: "Resitricted diffusion",
        7: "Normal-appearing white matter (NAWM)",
        13: "Enhancing tumor + Necrotic tumor",
        15: "Enhancing tumor + Susceptibility",
        16: "Enhancing tumor + Resitricted diffusion",
        156: "Enhancing tumor + Susceptibility + Resitricted diffusion",
        22: "Whole tumor mask",
    }


def get_segs(
    subject: Union[str, int],
    segs_dir: Path = SEGS_DIR,
    rois: Union[int, List[int]] = ROIS,
):
    """
    Given a subject ID number, returns volumetric segmentation masks for the specified region(s) of interest (rois).

    Parameters:
    -----------
    subject (str or int): The subject ID number.
    segs_dir (Path): The directory containing the segmentation masks.
    rois (List[int]): The regions of interest (rois) to extract from the available segmentation mask(s).

    Returns:
    --------
    (dict or None): A dictionary containing the volumetric segmentation masks for the available specified region(s) of interest (roi),
    or None if the subject has no segmentations available. Returned masks are always binary, with 1s indicating the
    presence of the roi and 0s elsewhere.

    ROI Key:
    --------
    1: Enhancing tumor*
    2: Other tumor
    3: Necrotic tumor*
    4: Edema*
    5: Susceptibility*
    6: Resitricted diffusion*
    7: Normal-appearing white matter (NAWM)
    13: Enhancing tumor + Necrotic tumor
    15: Enhancing tumor + Susceptibility
    16: Enhancing tumor + Resitricted diffusion
    156: Enhancing tumor + Susceptibility + Resitricted diffusion
    22: Whole tumor mask
    """
    all_seg_paths = [
        f
        for f in Path(segs_dir).iterdir()
        if (
            f.name.startswith(f"Segmentation {subject}.nii")
            or f.name.startswith(f"Segmentation {subject} ")
        )
    ]
    if len(all_seg_paths) == 0:
        return None
    all_seg_arrays = []
    all_seg_labels = []
    for f in all_seg_paths:
        seg_arr = image_read(str(f), reorient="IAL").numpy()
        all_seg_arrays.append(seg_arr)
        all_seg_labels.extend([int(v) for v in np.unique(seg_arr) if v != 0])

    all_seg_labels = sorted(list(set(all_seg_labels)))

    # Check to see if subject has enhancing and [(necrotic=3), (susceptibility=5), (resitricted diffusion=6)] segmentations, if so, add appropriate labels (13/15/16) to the list
    if 1 in all_seg_labels:
        if 3 in all_seg_labels:
            all_seg_labels.append(13)
        if 5 in all_seg_labels:
            all_seg_labels.append(15)
        if 6 in all_seg_labels:
            all_seg_labels.append(16)
            if 5 in all_seg_labels:
                all_seg_labels.append(156)

    all_seg_labels = sorted(list(set(all_seg_labels)))
    all_seg_labels.append(22)  # Add the whole tumor mask label

    # Create list of masks, one for each segmentation label
    masks = []
    for lab in all_seg_labels:
        mask = np.zeros_like(all_seg_arrays[0])
        for seg_arr in all_seg_arrays:
            if lab == 22:
                mask = np.logical_or(
                    mask > 0, np.logical_and(seg_arr > 0, seg_arr != 7)
                )  # we want to exclude the NAWM label = 7
                mask = mask.astype(int) * 22
            elif lab == 13:
                mask = np.logical_or(mask == 13, seg_arr == 1)
                mask = mask.astype(int) * 13
                mask = np.logical_or(mask == 13, seg_arr == 3)
                mask = mask.astype(int) * 13
            elif lab == 15:
                mask = np.logical_or(mask == 15, seg_arr == 1)
                mask = mask.astype(int) * 15
                mask = np.logical_or(mask == 15, seg_arr == 5)
                mask = mask.astype(int) * 15
            elif lab == 16:
                mask = np.logical_or(mask == 16, seg_arr == 1)
                mask = mask.astype(int) * 16
                mask = np.logical_or(mask == 16, seg_arr == 6)
                mask = mask.astype(int) * 16
            elif lab == 156:
                mask = np.logical_or(mask == 156, seg_arr == 1)
                mask = mask.astype(int) * 156
                mask = np.logical_or(mask == 156, seg_arr == 5)
                mask = mask.astype(int) * 156
                mask = np.logical_or(mask == 156, seg_arr == 6)
                mask = mask.astype(int) * 156
            else:
                mask = np.logical_or(mask == lab, seg_arr == lab)
                mask = mask.astype(int) * lab

        masks.append(mask)

    # grab those masks that correspond to the requested rois
    if isinstance(rois, int):
        rois = [rois]

    final_masks = {}
    for roi in rois:
        if roi in all_seg_labels:
            roi_idx = all_seg_labels.index(roi)
            mask_oi = masks[roi_idx]
            mask_oi = mask_oi > 0
            final_masks[roi] = mask_oi.astype(int)

    return final_masks
