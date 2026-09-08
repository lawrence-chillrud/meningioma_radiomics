# %%
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from ants import image_read
import matplotlib as mpl
from ipywidgets import interact
from skimage.measure import label

from src.utils import (
    get_mris,
    get_segs,
    get_segs_roi_key
)

FONT_SIZE = 14
mpl.rcParams.update(
    {
        "font.size": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE,
        "figure.titlesize": FONT_SIZE,
    }
)

def rescale_linear(array: np.ndarray, new_min: int, new_max: int):
    """Rescale an array linearly."""
    minimum, maximum = np.min(array), np.max(array)
    m = (new_max - new_min) / (maximum - minimum)
    b = new_min - m * minimum
    return m * array + b

def pyradiomics_bin(arr: np.ndarray, mask: np.ndarray, bin_width: float = 25) -> np.ndarray:
    """Discretize arr within mask using fixed bin width, matching PyRadiomics."""
    masked_vals = arr[mask.astype(bool)]
    low = masked_vals.min() - (masked_vals.min() % bin_width)
    high = masked_vals.max() + bin_width
    bin_edges = np.arange(low, high, bin_width)
    quantized = np.zeros_like(arr, dtype=int)
    quantized[mask.astype(bool)] = np.digitize(masked_vals, bin_edges)
    return quantized

def glszm(arr: np.ndarray, mask: np.ndarray, bin_width: float = 25) -> np.ndarray:
    """
    Build the Gray Level Size Zone Matrix for a 2D array and binary mask.

    Rows = gray levels (1..Ng), Cols = zone sizes (1..Ns)
    """
    assert arr.shape == mask.shape

    quantized = pyradiomics_bin(arr, mask, bin_width)

    Ng = quantized[mask.astype(bool)].max()  # actual number of gray levels after binning
    Ns = mask.sum()

    matrix = np.zeros((Ng, Ns), dtype=int)

    for g in range(1, Ng + 1):
        level_map = (quantized == g) & mask.astype(bool)
        if not level_map.any():
            continue

        labeled = label(level_map, connectivity=1)
        _, zone_sizes = np.unique(labeled[labeled > 0], return_counts=True)

        for size in zone_sizes:
            if size <= Ns:
                matrix[g - 1, size - 1] += 1

    matrix = matrix[:, :matrix.any(axis=0).nonzero()[0][-1] + 1]
    return matrix

def zone_entropy(arr: np.ndarray, mask: np.ndarray) -> float:
    """
    Calculates GLSZM Zone Entropy, matching PyRadiomics' implementation.
    """
    eps = np.spacing(1)

    P = glszm(arr, mask)
    Nz = P.sum()               # total number of zones
    p = P / Nz                 # normalize

    ze = -np.sum(p * np.log2(p + eps))
    return ze

def skewness(arr: np.ndarray) -> float:
    """
    Calculates the Fisher-Pearson standardized moment coefficient (skewness).
    Equivalent to scipy.stats.skew(arr, bias=True).
    """
    n = arr.size
    mean = arr.mean()
    std = arr.std()
    return (1/n) * np.sum(((arr - mean) / std) ** 3)

def kurtosis(arr: np.ndarray, excess: bool = False) -> float:
    """
    Calculates kurtosis.
    
    Parameters
    ----------
    excess : bool
        If True (default), returns excess kurtosis (Fisher's definition),
        where a normal distribution has kurtosis=0.
        If False, returns Pearson's kurtosis, where a normal distribution
        has kurtosis=3.
    
    Equivalent to scipy.stats.kurtosis(arr, bias=True, fisher=excess).
    """
    n = arr.size
    mean = arr.mean()
    std = arr.std()
    k = (1/n) * np.sum(((arr - mean) / std) ** 4)
    return k - 3 if excess else k

SEGS_ROI_KEY = get_segs_roi_key()

def explore_3D_array_with_mask_contour(subject, pulse, roi, reorient="IAR", thickness: int = 1):
    arr = image_read(
        str(get_mris(subject, pulses=pulse)[pulse]), reorient=reorient
    ).numpy()
    mask = get_segs(subject, rois=roi, reorient=reorient)[roi]
    assert arr.shape == mask.shape

    mins = []
    skewnesses = []
    kurtoses = []
    zes = []
    for s in range(arr.shape[0]):
        if mask[s, :, :].sum() > 0:
            zes.append(zone_entropy(arr[s, :, :], mask[s, :, :]))
        else:
            zes.append(np.nan)
        
        arr_flat = arr[s, :, :].flatten()
        mask_flat = mask[s, :, :].flatten()
        cur_vals = arr_flat[mask_flat == 1]
        if len(cur_vals):
            mins.append(round(cur_vals.min().round(2), 2))
            skewnesses.append(skewness(cur_vals).round(2))
            kurtoses.append(kurtosis(cur_vals).round(2))
        else:
            mins.append(np.nan)
            skewnesses.append(np.nan)
            kurtoses.append(np.nan)

    _arr = rescale_linear(arr, 0, 1)
    _mask = rescale_linear(mask, 0, 1)
    _mask = _mask.astype(np.uint8)

    arr_min, arr_max = arr.min(), arr.max()  # grab original min/max before rescaling

    def fn(SLICE):
        title1 = f"Subject {subject} on {pulse}"
        title2 = f"Contours identify {SEGS_ROI_KEY[roi].upper()}"
        if mins[SLICE] != None:
            title3 = f"ROI Min={mins[SLICE]:.2f}, Skewness={skewnesses[SLICE]}, Kurtosis={kurtoses[SLICE]}, Zone Entropy={zes[SLICE]:.2f}".ljust(50)
        else:
            title3 = f"ROI Min={mins[SLICE]}, Skewness={skewnesses[SLICE]}, Kurtosis={kurtoses[SLICE]}, Zone Entropy={zes[SLICE]}".ljust(50)
        title = f"{title1}\n{title2}\n{title3}"

        arr_rgb = cv2.cvtColor(_arr[SLICE, :, :], cv2.COLOR_GRAY2RGB)
        contours, _ = cv2.findContours(_mask[SLICE, :, :], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        arr_with_contours = cv2.drawContours(arr_rgb, contours, -1, (0, 1, 0), thickness)

        fig, ax = plt.subplots(figsize=(7, 7))
        im = ax.imshow(arr_with_contours, vmin=0, vmax=1, cmap='gray')
        ax.axis('off')
        ax.set_title(title, fontfamily="monospace")

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels([f"{arr_min:.2f}", f"{arr_max:.2f}"])

        plt.show()

    interact(fn, SLICE=(0, arr.shape[0] - 1))
    return arr, mask, mins, skewnesses, kurtoses, zes
# %%
mri, seg, mins, skewnesses, kurtoses, zone_entropies = explore_3D_array_with_mask_contour(83, "ADC", 22)
print("MIN VAL:")
print(f"\tMax at slice #: {np.nanargmax(mins)}")
print(f"\tMin at slice #: {np.nanargmin(mins)}")
print()
print("SKEWNESS:")
print(f"\tMax at slice #: {np.nanargmax(skewnesses)}")
print(f"\tMin at slice #: {np.nanargmin(skewnesses)}")
print()
print("KURTOSIS:")
print(f"\tMax at slice #: {np.nanargmax(kurtoses)}")
print(f"\tMin at slice #: {np.nanargmin(kurtoses)}")
print()
print("ZONE ENTROPY:")
print(f"\tMax at slice #: {np.nanargmax(zone_entropies)}")
print(f"\tMin at slice #: {np.nanargmin(zone_entropies)}")

# %%

