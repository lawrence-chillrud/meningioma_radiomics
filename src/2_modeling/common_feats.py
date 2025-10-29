# %%
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.utils import MODELING_DIR, translate_feat_names

# %%
coef_files = [f for f in MODELING_DIR.rglob("*_coefs.csv")]

dfs = []

for f in coef_files:
    dfs.append(pd.read_csv(f)[["Prediction task", "Feature", "Prop Var Exp"]])

df = pd.concat(dfs, axis=0)

# %%
feature_counts = df["Feature"].value_counts()
common_features = feature_counts[feature_counts > 2]

# %%
feat_order = (
    df[df["Feature"].isin(common_features.index)]
    .groupby("Feature")["Prop Var Exp"]
    .sum()
    .sort_values(ascending=False)
    .index
)

task_order = ["Chr22q", "Chr1p", "Merlin Intact", "Immune Enriched", "Hypermetabolic"]

plt.figure()
sns.barplot(
    df[df["Feature"].isin(common_features.index)],
    order=feat_order,
    x="Feature",
    y="Prop Var Exp",
    hue="Prediction task",
    hue_order=task_order,
)
plt.xticks(
    ticks=np.arange(len(feat_order)),
    labels=translate_feat_names(feat_order),
    rotation=45,
    ha="right",
)
plt.xlabel(None)
plt.ylabel("Proportion of variance explained")
plt.show()
plt.close()
# %%
