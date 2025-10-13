import traceback

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numba import njit
from scipy.stats import entropy, kurtosis, skew
from tqdm import tqdm

from src.utils import COLLAGE_DIR, read_ndarray

MAX_WORKERS = 4
A_DIR = COLLAGE_DIR / "a_raw_collage"
RAW_COLLAGE_DIR = sorted([d for d in A_DIR.iterdir() if d.is_dir()])[0]
AGGREGATED_COLLAGE_DIR = COLLAGE_DIR / "b_aggregated_collage" / RAW_COLLAGE_DIR.name
AGGREGATED_COLLAGE_DIR.mkdir(parents=True, exist_ok=True)

COLLAGE_FEAT_NAMES = [
    "AngularSecondMoment",
    "Contrast",
    "Correlation",
    "SumOfSquareVariance",
    "SumAverage",
    "SumVariance",
    "SumEntropy",
    "Entropy",
    "DifferenceVariance",
    "DifferenceEntropy",
    "InformationMeasureOfCorrelation1",
    "InformationMeasureOfCorrelation2",
    "MaximalCorrelationCoefficient",
]


def get_metadata_from_filename(filename):
    metadata_list = (
        filename.name.replace("T1_POST", "T1POST").split(".joblib")[0].split("_")
    )
    metadata_dict = {}
    for item in metadata_list:
        item_name, item_val = item.split("-")
        metadata_dict[item_name] = item_val

    return metadata_dict


@njit
def _robust_mean_abs_dev(arr, p10, p90):
    total = 0.0
    count = 0
    mean_val = 0.0
    # first compute mean of robust slice
    for val in arr:
        if p10 <= val <= p90:
            mean_val += val
            count += 1
    if count == 0:
        return np.nan
    mean_val /= count
    # compute mean absolute deviation
    for val in arr:
        if p10 <= val <= p90:
            total += abs(val - mean_val)
    return total / count


def calculate_summary_stats(arr):
    """Numba-accelerated calculation of summary statistics for a 1D or ND array."""
    arr = arr.ravel()
    mask = ~np.isnan(arr)
    arr = arr[mask]
    n = arr.size
    if n == 0:
        # empty array; return NaNs
        nan_dict = {
            k: np.nan
            for k in [
                "mean",
                "std",
                "var",
                "min",
                "max",
                "median",
                "iqr",
                "range",
                "10perc",
                "90perc",
                "energy",
                "totalenergy",
                "MAD",
                "rMAD",
                "rms",
                "uniformity",
                "skewness",
                "kurtosis",
                "entropy",
            ]
        }
        return nan_dict

    # precompute basic stats
    mean_val = np.mean(arr)
    std_val = np.std(arr)
    var_val = std_val**2
    min_val = np.min(arr)
    max_val = np.max(arr)
    median_val = np.median(arr)
    q25, q75 = np.percentile(arr, [25, 75])
    arr10, arr90 = np.percentile(arr, [10, 90])
    iqr_val = q75 - q25
    range_val = max_val - min_val
    energy = np.sum(arr**2)
    totalenergy = energy * n
    MAD = np.mean(np.abs(arr - mean_val))
    rMAD = _robust_mean_abs_dev(arr, arr10, arr90)
    rms = np.sqrt(energy / n)

    hist, bin_edges = np.histogram(arr, bins=10, density=True)
    uniformity = np.sum((hist**2) * np.diff(bin_edges))

    if std_val < 1e-12:
        skewness_val = 0.0
        kurtosis_val = 0.0
    else:
        skewness_val = skew(arr, nan_policy="omit")
        kurtosis_val = kurtosis(arr, nan_policy="omit")

    probabilities = hist * np.diff(bin_edges)
    entropy_val = entropy(probabilities)

    return {
        "mean": mean_val,
        "std": std_val,
        "var": var_val,
        "min": min_val,
        "max": max_val,
        "median": median_val,
        "iqr": iqr_val,
        "range": range_val,
        "10perc": arr10,
        "90perc": arr90,
        "energy": energy,
        "totalenergy": totalenergy,
        "MAD": MAD,
        "rMAD": rMAD,
        "rms": rms,
        "uniformity": uniformity,
        "skewness": skewness_val,
        "kurtosis": kurtosis_val,
        "entropy": entropy_val,
    }


def post_process_collage(filename):
    metadata = get_metadata_from_filename(filename)
    collage = read_ndarray(filename)

    rows = []
    for angle in range(collage.shape[-1]):
        for feat in range(collage.shape[3]):
            row = dict(metadata)
            row["collage_feat"] = COLLAGE_FEAT_NAMES[feat]
            row["collage_angle"] = angle

            c_collage = collage[:, :, :, feat, angle]
            current_stats = calculate_summary_stats(c_collage)
            row.update(current_stats)

            rows.append(row)

    return rows


def safe_post_process(filename):
    """Wrapper around post_process_collage that catches exceptions."""
    try:
        return post_process_collage(filename)
    except Exception as e:
        print(f"[WARNING] Failed processing {filename}: {e}")
        traceback.print_exc()
        return []  # return empty list so pipeline continues


def main():
    collage_paths = [p for p in RAW_COLLAGE_DIR.iterdir() if p.name.endswith(".joblib")]

    results = Parallel(n_jobs=MAX_WORKERS, backend="loky", verbose=0)(
        delayed(safe_post_process)(p)
        for p in tqdm(collage_paths, ncols=120, desc="Post-processing collage")
    )

    all_rows = [row for file_rows in results for row in file_rows]

    results_df = pd.DataFrame(all_rows)
    for (win_val, bin_val), group_df in results_df.groupby(["win", "bin"]):
        filename = f"features-collage_win-{win_val}_bin-{bin_val}.csv"
        filepath = AGGREGATED_COLLAGE_DIR / filename
        group_df.to_csv(filepath, index=False)


if __name__ == "__main__":
    main()
