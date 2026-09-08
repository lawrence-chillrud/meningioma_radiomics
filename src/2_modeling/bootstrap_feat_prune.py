# %%
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from tqdm import tqdm
from src.utils import PYRAD_FILE, LABELS_FILE, MODELING_DIR, clean_feature_names, get_feats
from src.utils.get_feats import remove_correlated_features
from src.utils.plotting import translate_feat_names

from statsmodels.stats.outliers_influence import variance_inflation_factor
from pathlib import Path

PREDICTION_TASK = "MethylationSubgroup"  # ["Chr22q", "Chr1p", "MethylationSubgroup"]
CORRELATED_FEATS_THRESH = 0.99  # [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
RESULTS_DIR = MODELING_DIR / "pyradiomics" / Path(__file__).name.replace(".py", "")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FONT_SIZE = 16

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

def load_feats(
    prediction_task=PREDICTION_TASK,
    features_path=PYRAD_FILE,
    labels_path=LABELS_FILE,
    drop_extra_shape_feats=True,
):
    # Read in features, labels, & merge
    features = pd.read_csv(features_path)
    features = features.rename(columns={"subject": "Subject Number"})
    labels = pd.read_csv(labels_path)
    labels = labels.dropna(subset=[prediction_task])
    labels = labels[labels["Subject Number"].isin(features["Subject Number"])]
    data = features.merge(labels, on="Subject Number")

    # Clean up column names
    data.columns = clean_feature_names(data.columns)

    # Drop metadata columns to isolate dataset
    data = data.dropna(axis=1, how="all").fillna(0)
    y = data[prediction_task].values.astype(int)
    X = data.drop(
        columns=[
            "Subject Number",
            "MethylationSubgroup",
            "Chr1p",
            "Chr22q",
            "Chr9p",
            "TERT",
        ]
    )
    X = X.drop(columns=[col for col in X.columns if "-7-" in col])  # NAWM feats

    # Drop cols w/constant values
    constant_feats = [col for col in X.columns if X[col].nunique() == 1]
    X = X.drop(columns=constant_feats)

    # Subject IDs
    sub_nos = data["Subject Number"].values

    # Since shape feats do not vary w/pulse seq, drop them for all but T1
    # for easier optimization / to avoid multiple of the same shape feats
    # sharing variance across.
    if drop_extra_shape_feats:
        X = X.drop(columns=[f for f in X.columns if ("shape" in f) and ("T1" not in f)])

    return X, y, sub_nos


def bootstrap_stability(
    X: pd.DataFrame,
    tau: float,
    random_seed: int = 42,
) -> dict:
    """
    Run the pruning procedure on B bootstrap resamples and record
    which features are retained in each iteration.

    Parameters
    ----------
    X             : Full training feature matrix (samples x features).
    tau           : Correlation threshold passed to greedy_prune.
    n_bootstraps  : Number of bootstrap resamples.
    random_seed   : Seed for the random number generator.

    Returns
    -------
    Dictionary with keys:
      'retention_freq'  : pd.Series, retention frequency per feature (0-1).
      'set_sizes'       : list[int], number of features retained per resample.
      'retained_matrix' : pd.DataFrame, binary (features x bootstraps).
    """
    rng = np.random.default_rng(random_seed)
    n_samples = len(X)
    val_test_samples = []
    for i in range(n_samples):
        for j in range(n_samples):
            if i != j:
                val_test_samples.append([i, j])

    all_features = X.columns.tolist()

    retained_matrix = pd.DataFrame(
        0, index=all_features, columns=range(len(val_test_samples)), dtype=np.int8
    )
    set_sizes = []

    for b in tqdm(
        range(len(val_test_samples)), desc="Bootstrap iteration", total=len(val_test_samples)
    ):
        val_idx, test_idx = val_test_samples[b]

        # Draw resample indices with replacement
        train_idx = [k for k in range(n_samples) if (k != val_idx) and (k != test_idx)]
        X_boot = X.iloc[train_idx].reset_index(drop=True)

        vif_path = RESULTS_DIR / "VIF_Tables" / PREDICTION_TASK / f"{b}.csv"
        vif_path.parent.mkdir(parents=True, exist_ok=True)

        kept = remove_correlated_features(X_boot, vif_path=vif_path, threshold=tau)

        retained_matrix.loc[kept.columns, b] = 1
        set_sizes.append(len(kept))
        if b == 999:
            break
    
    retention_freq = retained_matrix.mean(axis=1).sort_values(ascending=False)

    return {
        "retention_freq": retention_freq,
        "set_sizes": set_sizes,
        "retained_matrix": retained_matrix,
    }


def plot_stability_results(
    results: dict,
    stability_threshold: float = 0.9,
    top_n_features: int = 50,
    figsize: tuple = (16, 14),
) -> plt.Figure:
    """
    Three-panel stability report:

      Panel 1 (top-left)  : Histogram of retention frequencies across all
                            features. Bimodality indicates stable selection.
      Panel 2 (top-right) : Histogram of retained set sizes across bootstraps.
                            Tight distribution indicates stable pruning depth.
      Panel 3 (bottom)    : Ranked retention frequency bar chart for the
                            top_n_features most-frequently retained features,
                            with a reference line at stability_threshold.

    Parameters
    ----------
    results              : Output dict from bootstrap_stability().
    stability_threshold  : Reference line drawn on panel 3 (default 0.9).
    top_n_features       : How many features to show in the ranked bar chart.
    figsize              : Overall figure size.
    """
    retention_freq = results["retention_freq"]
    set_sizes = results["set_sizes"]

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    # --- Panel 1: Retention frequency histogram ---
    ax1.hist(retention_freq.values, bins=40, color="steelblue", edgecolor="white")
    ax1.axvline(
        stability_threshold,
        color="crimson",
        linestyle="--",
        linewidth=1.5,
        label=f"Stability threshold = {stability_threshold}",
    )
    n_stable = (retention_freq >= stability_threshold).sum()
    ax1.set_xlabel("Retention frequency", fontsize=11)
    ax1.set_ylabel("Number of features", fontsize=11)
    ax1.set_title(
        "Distribution of retention frequencies\nacross all features", fontsize=12
    )
    ax1.legend(fontsize=9)
    ax1.text(
        0.98,
        0.95,
        f"Stable features: {n_stable}",
        transform=ax1.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="crimson",
    )

    # --- Panel 2: Set size distribution ---
    ax2.hist(set_sizes, bins=30, color="mediumseagreen", edgecolor="white")
    ax2.axvline(
        np.mean(set_sizes),
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean = {np.mean(set_sizes):.1f}",
    )
    ax2.set_xlabel("Number of features retained", fontsize=11)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.set_title(
        f"Retained set size across {len(set_sizes)} bootstrap resamples\n"
        f"(SD = {np.std(set_sizes):.1f})",
        fontsize=12,
    )
    ax2.legend(fontsize=9)

    # --- Panel 3: Ranked bar chart for top N features ---
    # Only show features retained in at least one bootstrap
    nonzero = retention_freq[retention_freq > 0]
    top = nonzero.head(top_n_features)

    bar_colors = [
        "steelblue" if v >= stability_threshold else "lightskyblue" for v in top.values
    ]
    bars = ax3.barh(
        range(len(top)),
        top.values[::-1],
        color=bar_colors[::-1],
        edgecolor="white",
        height=0.7,
    )
    ax3.set_yticks(range(len(top)))
    ax3.set_yticklabels(top.index[::-1], fontsize=8)
    ax3.axvline(
        stability_threshold,
        color="crimson",
        linestyle="--",
        linewidth=1.5,
        label=f"Stability threshold = {stability_threshold}",
    )
    ax3.set_xlabel("Retention frequency", fontsize=11)
    ax3.set_title(
        f"Ranked retention frequency — top {len(top)} most-selected features\n"
        f"(dark blue ≥ {stability_threshold}, light blue < {stability_threshold})",
        fontsize=12,
    )
    ax3.set_xlim(0, 1.05)
    ax3.legend(fontsize=9)

    fig.suptitle("Bootstrap Stability Analysis — Feature Pruning", fontsize=14, y=1.01)
    return fig

def run_stability_analysis(
    X: pd.DataFrame,
    tau: float,
    random_seed: int = 42,
    stability_threshold: float = 0.9,
    top_n_features: int = 50,
    save_path: str = None,
) -> tuple[dict, plt.Figure]:
    """
    End-to-end convenience function.

    Parameters
    ----------
    X                    : Training feature matrix.
    tau                  : Correlation threshold for pruning.
    random_seed          : RNG seed (controls only the bootstrap resampling).
    stability_threshold  : Frequency cutoff for 'stable' label in plots.
    top_n_features       : Features shown in ranked bar chart.
    save_path            : If provided, save the figure to this path.

    Returns
    -------
    (results dict, matplotlib Figure)
    """
    print(f"Running bootstrap stability analysis: tau={tau}")
    results = bootstrap_stability(X, tau=tau, random_seed=random_seed)

    freq = results["retention_freq"]
    stable = freq[freq >= stability_threshold]
    ambiguous = freq[(freq > 0.3) & (freq < 0.7)]

    print(f"\nSummary")
    print(f"  Total features          : {len(freq)}")
    print(f"  Stably retained (≥{stability_threshold}) : {len(stable)}")
    print(f"  Ambiguous (0.3–0.7)     : {len(ambiguous)}")
    print(
        f"  Retained set size       : {np.mean(results['set_sizes']):.1f} ± {np.std(results['set_sizes']):.1f} (mean ± SD)"
    )

    fig = plot_stability_results(
        results,
        stability_threshold=stability_threshold,
        top_n_features=top_n_features,
    )

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=600)
        print(f"\nFigure saved to {save_path}")

    return results, fig

# %%
# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
# X, _, _ = get_feats(
#     prediction_task=PREDICTION_TASK,
#     features_path=PYRAD_FILE,
#     scaler=None,
#     low_var_thresh=None,
#     remove_correlated_feats=None
# )
# X.columns = translate_feat_names(X.columns)
X, _, _ = load_feats()
results, fig = run_stability_analysis(
    X=X,
    tau=CORRELATED_FEATS_THRESH,
    random_seed=42,
    stability_threshold=0.9,
    top_n_features=10,
    save_path=RESULTS_DIR
    / f"{PREDICTION_TASK}_tau{CORRELATED_FEATS_THRESH}_stability_analysis.png",
)
plt.show()
plt.close()
