# %%
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import cm
from matplotlib.colors import Normalize
import pandas as pd
import numpy as np
import seaborn as sns

from src.utils import PYRAD_FILE

# Read in features
df = pd.read_csv(PYRAD_FILE)

# Collect list of columns to drop (NAWM, constant feats, metadata) and drop
nawm_feats = [col for col in df.columns if "-7-" in col]
constant_feats = [col for col in df.columns if df[col].nunique() == 1]
feats_to_drop = ["Subject Number"] + nawm_feats + constant_feats
df_for_corr = df.drop(columns=feats_to_drop).dropna(axis=1, how="all")

# Get possible groupings of features
feat_categories = list(
    set([c.split("Feat-original_")[-1].split("_")[0] for c in df_for_corr.columns])
)
pulse_categories = list(
    set([c.replace("Mod-", "").split("-")[0] for c in df_for_corr.columns])
)
seg_categories = list(
    set(
        [
            c.replace("Mod-", "").split("-SegLab-")[-1].split("-Feat-")[0]
            for c in df_for_corr.columns
        ]
    )
)

# Choose grouping and colour accordingly
unique_groups = feat_categories
palette = sns.color_palette("colorblind", len(unique_groups))
group_colors = {grp: palette[i] for i, grp in enumerate(unique_groups)}
group_map = {}
for c in df_for_corr.columns:
    for g in unique_groups:
        if g in c:
            group_map[c] = g
col_colors = pd.Series(group_map).map(group_colors)

ordered_cols = []
for g in unique_groups:
    grouped_cols = [c for c in df_for_corr.columns if g in c]
    ordered_cols.extend(grouped_cols)
df_for_corr = df_for_corr[ordered_cols]

# Calculate correlation matrix
corr = df_for_corr.corr()

# Plot
sns.clustermap(
    corr,
    row_cluster=False,
    col_cluster=False,
    row_colors=col_colors,
    col_colors=col_colors,
    cmap="viridis",
    cbar_pos=None,
    center=0,
    xticklabels=False,
    yticklabels=False,
    figsize=(10, 10),
)

plt.show()

corr_for_hist = corr.copy().to_numpy()
corr_for_hist[np.diag_indices(corr.shape[0])] = np.nan
data = corr_for_hist.flatten()

# Plot histogram of corr mat
sns.histplot(
    data,
    bins=20
)

# Set x-axis ticks every 0.2
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(0.2))

# Set axis labels
plt.xlabel("Correlation")
plt.ylabel("Count")

# Add commas to y-axis ticks
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{int(x):,}"))

plt.show()

# %%
plt.figure(figsize=(10, 10))
plt.rcParams.update({'font.size': 18})

# Create histogram and capture bar containers and bin edges
counts, bins, patches = plt.hist(data, bins=20)

# Normalize bin centers to range [-1, 1] for mapping to colormap
norm = Normalize(vmin=-1, vmax=1)
cmap = cm.get_cmap('viridis')

# Apply colors to each bar based on bin center
for bin_center, patch in zip((bins[:-1] + bins[1:]) / 2, patches):
    patch.set_facecolor(cmap(norm(bin_center)))

# Set x-axis ticks every 0.2
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(0.2))

# Set axis labels
plt.xlabel("Correlation")
plt.ylabel("Count")

# Format y-axis with commas
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{int(x):,}"))

plt.show()

# %%
