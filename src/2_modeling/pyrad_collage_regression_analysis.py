# %% Imports
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.linear_model import Lasso
import seaborn as sns
import matplotlib.pyplot as plt
from src.utils import PYRAD_FILE, LABELS_FILE, MODELING_DIR, COLLAGE_DIR, get_feats
import matplotlib as mpl
from datetime import datetime
import logging
from pathlib import Path
from matplotlib.lines import Line2D

sns.set_theme(style="whitegrid")

# User defined settings
PREDICTION_TASK = (
    "MethylationSubgroup"  # can be one of "MethylationSubgroup", "Chr22q", or "Chr1p"
)
SCALER = "Standard"  # can be one of "Standard", "MinMax", or None
LOW_VAR_THRESH = None  # or 0.2?
ALPHA = 0.1
NONZERO_THRESH = 0.99
MAX_ITERS = 20_000
LASSO_TOL = 1e-4
COLLAGE_FEATURES_PATHS = [f for f in COLLAGE_DIR.rglob("*wide-features*.csv")]
COLLAGE_FILE = COLLAGE_FEATURES_PATHS[-1]

# Output dir and logfile set up
TIMESTAMP = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
OUTPUT_DIR = MODELING_DIR / "regression_analysis" / TIMESTAMP
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOGFILE = OUTPUT_DIR / "logfile.txt"

# Setup logfile
logging.basicConfig(
    filename=LOGFILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logging.info(f"<>" * 40)
logging.info(f"Log file for {Path(__file__).name} run at {TIMESTAMP}")
logging.info(f"Settings used:")
logging.info(f"\tPREDICTION_TASK: {PREDICTION_TASK}")
logging.info(f"\tSCALER: {SCALER}")
logging.info(f"\tALPHA: {ALPHA}")
logging.info(f"\tMAX_ITERS: {MAX_ITERS}")
logging.info(f"\tLASSO_TOL: {LASSO_TOL}")
logging.info(f"\tCOLLAGE_FILE: {COLLAGE_FILE}")
logging.info(f"\tPYRAD_FILE: {PYRAD_FILE}")
logging.info(f"\tLABELS_FILE: {LABELS_FILE}")

# Read in collage features, labels, and subject ID numbers
collage_df, y_c, subs_c = get_feats(
    prediction_task=PREDICTION_TASK,
    features_path=COLLAGE_FILE,  # "/Users/lgc2035/Documents/research/Meningioma_project/backups/09-25-2025/Meningioma/data/collage_sparse_small_windows/windowsize-5_binsize-32_summary_22nansfilled.csv", # COLLAGE_FEATURES_PATHS[-3],
    labels_path=LABELS_FILE,
    scaler=SCALER,
    low_var_thresh=LOW_VAR_THRESH,
)
collage_df["Subject Number"] = subs_c
collage_df = collage_df.sort_values(by="Subject Number")

# Read in radiomics features, labels, and subject ID numbers
radiomics_df, y_r, subs_r = get_feats(
    prediction_task=PREDICTION_TASK,
    features_path=PYRAD_FILE,  # "/Users/lgc2035/Documents/research/Meningioma_project/backups/09-25-2025/meningioma_data/radiomics/features8_smoothed/features_wide.csv", # PYRAD_FILE,
    labels_path=LABELS_FILE,
    scaler=SCALER,
    low_var_thresh=LOW_VAR_THRESH,
)
radiomics_df["Subject Number"] = subs_r
radiomics_df = radiomics_df.sort_values(by="Subject Number")

# Get the indices of those subject numbers occurring in both datasets
overlapping_subjects = sorted(
    list(
        set(collage_df["Subject Number"]).intersection(
            set(radiomics_df["Subject Number"])
        )
    )
)
overlapping_cidxs = np.where(
    pd.Series(collage_df["Subject Number"]).isin(overlapping_subjects)
)[0]
overlapping_ridxs = np.where(
    pd.Series(radiomics_df["Subject Number"]).isin(overlapping_subjects)
)[0]

# Filter the datasets to only include the overlapping subjects
collage_df = collage_df.iloc[overlapping_cidxs]
radiomics_df = radiomics_df.iloc[overlapping_ridxs]

# Drop subject numbers now that filtering is done
collage_df = collage_df.drop(columns=["Subject Number"])
radiomics_df = radiomics_df.drop(columns=["Subject Number"])

# Generate synthetic dataset
noise_data = np.random.normal(loc=0, scale=1, size=collage_df.shape)
df_noise = pd.DataFrame(
    noise_data, columns=[f"Col{i+1}" for i in range(collage_df.shape[1])]
)

# Generate theoretical counts (for plot)
theoretical_counts = pd.DataFrame(
    {
        "rsquared": [1.0, 0.0],
        "num_real_nonzeros": [0, 0],
        "counts": [radiomics_df.shape[1], 0],
    }
)


# %%
def regression_analysis(
    predictors_df, outcomes_df, alpha=ALPHA, nonzero_threshold=NONZERO_THRESH
):
    """
    Perform Lasso regression on each outcome in outcomes_df using predictors_df as the predictors.
    Return the R^2 values, the features selected, the coefficients, the number of non-zero coefficients, and the number of non-zero coefficients required to explain 80% of the variance.
    """
    model = Lasso(alpha=alpha, max_iter=MAX_ITERS, tol=LASSO_TOL)

    rsquareds = []
    num_nonzeros = []
    num_real_nonzeros = []
    iters = []

    score = "n/a"
    num_feats = "n/a"
    pbar = tqdm(
        range(outcomes_df.shape[1]),
        total=outcomes_df.shape[1],
        smoothing=0,
        desc=f"score = {score}, # feats (# exp 80% var) = {num_feats} ({num_feats})",
    )
    for i in pbar:
        model.fit(predictors_df, outcomes_df.iloc[:, i])
        score = model.score(predictors_df, outcomes_df.iloc[:, i])
        rsquareds.append(score)
        iters.append(model.n_iter_)
        num_nonzero = np.sum(model.coef_ != 0)
        num_nonzeros.append(num_nonzero)
        coef_sum = np.sum(np.abs(model.coef_))
        sorted_nonzero_coefs = np.sort(np.abs(model.coef_) / coef_sum)[::-1]
        cumsum = np.cumsum(sorted_nonzero_coefs)
        nonzero_real = np.sum(cumsum <= nonzero_threshold)
        num_real_nonzeros.append(nonzero_real)

        pbar.set_description(
            f"score = {round(score, 2)}, iter = {model.n_iter_}, # feats (# exp {round(nonzero_threshold*100)}% var) = {num_nonzero} ({nonzero_real})"
        )

    return rsquareds, num_nonzeros, num_real_nonzeros, iters


# %% proper regression analysis using radx feats as predictors, collage as outcomes
rsquareds, num_nonzeros, num_real_nonzeros, iters = regression_analysis(
    predictors_df=radiomics_df, outcomes_df=collage_df
)
stats_df = pd.DataFrame(
    {
        "rsquared": np.array(rsquareds).round(2),
        "num_nonzeros": num_nonzeros,
        "num_real_nonzeros": num_real_nonzeros,
    }
).sort_values(by=["rsquared", "num_real_nonzeros"], ascending=[False, True])
stats_counts = (
    stats_df.groupby(["rsquared", "num_real_nonzeros"])
    .size()
    .reset_index(name="counts")
)
logging.info(f"Step 1/3: Completed analysis for: CoLlAGe ~ PyRadiomics")

# %% baseline regression analysis using radiomics features as BOTH predictors and outcomes, to see how Lasso behaves
(
    baseline_rsquareds,
    baseline_num_nonzeros,
    baseline_num_real_nonzeros,
    baseline_iters,
) = regression_analysis(predictors_df=radiomics_df, outcomes_df=radiomics_df)
baseline_stats_df = pd.DataFrame(
    {
        "rsquared": np.array(baseline_rsquareds).round(2),
        "num_nonzeros": baseline_num_nonzeros,
        "num_real_nonzeros": baseline_num_real_nonzeros,
    }
).sort_values(by=["rsquared", "num_real_nonzeros"], ascending=[False, True])
baseline_counts = (
    baseline_stats_df.groupby(["rsquared", "num_real_nonzeros"])
    .size()
    .reset_index(name="counts")
)
logging.info(f"Step 2/3: Completed analysis for: PyRadiomics ~ PyRadiomics")

# %% noise regression analysis using radx feats as predictors, noise as outcome
noise_rsquareds, noise_num_nonzeros, noise_num_real_nonzeros, noise_iters = (
    regression_analysis(predictors_df=radiomics_df, outcomes_df=df_noise)
)
noise_stats_df = pd.DataFrame(
    {
        "rsquared": np.array(noise_rsquareds).round(2),
        "num_nonzeros": noise_num_nonzeros,
        "num_real_nonzeros": noise_num_real_nonzeros,
    }
).sort_values(by=["rsquared", "num_real_nonzeros"], ascending=[False, True])
noise_counts = (
    noise_stats_df.groupby(["rsquared", "num_real_nonzeros"])
    .size()
    .reset_index(name="counts")
)
logging.info(f"Step 3/3: Completed analysis for: Random Noise ~ PyRadiomics")

# %%
FONT_SIZE = 32

mpl.rcParams.update(
    {
        "font.size": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE - 4,
        "ytick.labelsize": FONT_SIZE - 4,
        "legend.fontsize": FONT_SIZE - 4,
        "figure.titlesize": FONT_SIZE,
    }
)


def single_relplot(df):
    fig, ax = plt.subplots(figsize=(19.2, 14.4))

    norm = mpl.colors.LogNorm(vmin=1, vmax=df["counts"].max())
    cmap = plt.get_cmap("Greys")
    size_range = (50, 8000)

    sns.scatterplot(
        data=df,
        x="rsquared",
        y="num_real_nonzeros",
        hue="counts",
        size="counts",
        hue_norm=norm,
        palette=cmap,
        sizes=size_range,
        edgecolor=cmap(0.5),
        legend=False,  # disable seaborn legend
        ax=ax,
    )

    # ----- custom combined legend -----
    legend_counts = [1, 25, 50, 100, 200, 400, 800, 2000]

    size_min, size_max = size_range
    count_min, count_max = df["counts"].min(), df["counts"].max()

    def scale_size(c):
        return np.interp(c, [count_min, count_max], [size_min, size_max]) ** 0.5

    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=scale_size(c),
            markerfacecolor=cmap(norm(c)),
            markeredgecolor=cmap(0.5),
            label=f"{c}",
        )
        for c in legend_counts
    ]

    ax.legend(
        handles=handles[::-1],
        title="Num features",
        title_fontsize=FONT_SIZE,
        frameon=True,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        labelspacing=1.75,  # vertical space between entries
        handletextpad=1.0,  # space between marker and text
        borderaxespad=0.0,
        borderpad=0.5,
    )

    # ----- formatting -----
    ax.invert_xaxis()
    ax.invert_yaxis()

    ax.xaxis.grid(True, which="minor", linewidth=0.25)
    ax.yaxis.grid(True, which="minor", linewidth=0.25)

    sns.despine(ax=ax, left=True, bottom=True)

    ax.set_yticks([1, 5, 10, 15, 20, 25, 30, 35, 40])
    ax.set_ylim(41, -1.5)
    ax.set_xlabel(r"$R^2$ of reconstruction")
    ax.set_ylabel(r"Num PyRadiomics predictors used in reconstruction")


def combined_relplot(df, max_counts):
    fig, ax = plt.subplots(figsize=(19.2, 14.4))

    legend_handles = []
    for p in df["palette"].unique():
        cur_df = df[df["palette"] == p]
        cur_name = cur_df.name.unique()[0]
        cur_alpha = cur_df.alpha.unique()[0]
        cur_max = cur_df.max_marker_size.unique()[0]
        sns.scatterplot(
            data=cur_df,
            x="rsquared",
            y="num_real_nonzeros",
            hue="counts",
            hue_norm=mpl.colors.LogNorm(vmin=1, vmax=max_counts),
            size="counts",
            edgecolor=plt.get_cmap(p)(0.5),
            alpha=cur_alpha,
            palette=p,
            sizes=(50, cur_max),  # 8000
            legend=False,
            ax=ax,
        )
        color1 = plt.get_cmap(p)(0.75)
        color2 = plt.get_cmap(p)(0.5)
        proxy = plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=30,
            markeredgewidth=2,
            markerfacecolor=color1,
            markeredgecolor=color2,
            alpha=cur_alpha,
            label=cur_name,
        )
        legend_handles.append(proxy)

    ax.legend(
        handles=[legend_handles[1], legend_handles[0], legend_handles[2]],
        title="Dependent feature set",
        title_fontsize=FONT_SIZE,
        loc="best",
    )

    ax.invert_xaxis()
    ax.invert_yaxis()

    ax.xaxis.grid(True, which="minor", linewidth=0.25)
    ax.yaxis.grid(True, which="minor", linewidth=0.25)

    sns.despine(ax=ax, left=True, bottom=True)

    ax.set_yticks([1, 5, 10, 15, 20, 25, 30, 35, 40])
    ax.set_ylim(41, -1.5)
    ax.set_xlabel(r"$R^2$ of reconstruction")
    ax.set_ylabel(r"Num PyRadiomics predictors used in reconstruction")


all_counts = [
    stats_counts.counts.max(),
    baseline_counts.counts.max(),
    noise_counts.counts.max(),
]

noise_counts["palette"] = "Greens"
noise_counts["name"] = "Random noise"
noise_counts["alpha"] = 0.5
noise_counts["max_marker_size"] = all_counts[2] / max(all_counts) * 8_000

baseline_counts["palette"] = "Blues"
baseline_counts["name"] = "PyRadiomics"
baseline_counts["alpha"] = 0.5
baseline_counts["max_marker_size"] = all_counts[1] / max(all_counts) * 8_000

stats_counts["palette"] = "Oranges"
stats_counts["name"] = "CoLlAGe"
stats_counts["alpha"] = 1.0
stats_counts["max_marker_size"] = all_counts[0] / max(all_counts) * 8_000

combined_df = pd.concat([stats_counts, baseline_counts, noise_counts])
combined_df["num_real_nonzeros"] = combined_df["num_real_nonzeros"] + 1
# combined_relplot(combined_df, max_counts=max(all_counts)) # 'Regressing different datasets on PyRadiomics features'
# single_relplot(combined_df)

# %%
combined_df.to_csv(OUTPUT_DIR / "combined_analysis.csv")
logging.info(f"Saved results to: combined_analysis.csv")
logging.info(f"Finished running {Path(__file__).name}")
logging.info(f"<>" * 40)

# %%
