# %%
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.utils import (
    PYRAD_FILE,
    LABELS_FILE,
    translate_feat_names,
    clean_feature_names,
)


for feat_of_interest in ["ENH Skewness on T1", "ENH Kurtosis on DWI", "TMR ZoneEntropy on ADC", "ENH & RSTDFF Min on ADC", "ENH LeastAxisLength"]:
    features = pd.read_csv(PYRAD_FILE)
    features = features.rename(columns={"subject": "Subject Number"})
    features = features.drop(columns=[f for f in features.columns if ("shape" in f) and ("T1" not in f)])
    features.columns = ["Subject Number"] + translate_feat_names(clean_feature_names(features.columns[1:]))
    features = features[["Subject Number", feat_of_interest]]
    scaler_obj = StandardScaler()

    X = pd.DataFrame(scaler_obj.fit_transform(features[[feat_of_interest]]), columns=[f"STANDARDIZED {feat_of_interest}"])

    final_df = pd.concat([features, X], axis=1)

    labels = pd.read_csv(LABELS_FILE)
    labels = labels[labels["Subject Number"].isin(features["Subject Number"])]
    final_df2 = final_df.merge(labels, on="Subject Number")
    final_df2["MethylationSubgroup"] = final_df2["MethylationSubgroup"].map({
        2.0: "Hypermitotic",
        1.0: "Immune Enriched",
        0.0: "Merlin Intact"
    })
    final_df2["Chr22q"] = final_df2["Chr22q"].map({
        1.0: "Lost",
        0.0: "Intact"
    })
    final_df2["Chr1p"] = final_df2["Chr1p"].map({
        1.0: "Lost",
        0.0: "Intact"
    })

    final_df2[[
        "Subject Number", feat_of_interest, 
        f"STANDARDIZED {feat_of_interest}",
        "MethylationSubgroup", "Chr1p", "Chr22q"
    ]].dropna(subset=[feat_of_interest]).sort_values(by=[feat_of_interest], ascending=False).to_csv(f"{feat_of_interest}.csv", index=False)

# %%
# Checking on correlations
feats_to_get_corr = ["ENH LeastAxisLength", "ENH Sphericity"]
features = pd.read_csv(PYRAD_FILE)
features = features.rename(columns={"subject": "Subject Number"})
features = features.drop(columns=[f for f in features.columns if ("shape" in f) and ("T1" not in f)])
features.columns = ["Subject Number"] + translate_feat_names(clean_feature_names(features.columns[1:]))
features[feats_to_get_corr].corr()
# %%
