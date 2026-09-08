import pandas as pd
from .paths import PYRAD_FILE
from .get_feats import clean_feature_names
from .plotting import translate_feat_names

def check_correlation(feats=[]):
    features = pd.read_csv(PYRAD_FILE)
    features = features.rename(columns={"subject": "Subject Number"})
    features = features.drop(columns=[f for f in features.columns if ("shape" in f) and ("T1" not in f)])
    features.columns = ["Subject Number"] + translate_feat_names(clean_feature_names(features.columns[1:]))
    return features[feats].corr()
