import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .paths import LABELS_FILE


def clean_feature_names(strings):
    """Tidies up the feature names by removing hard-coded substrings."""
    replacements = [
        "Mod-AX_3D_",
        "Mod-SAG_3D_",
        "_POST",
        "POST",
        "Mod-AX_",
        "SegLab-",
        "Feat-original_",
    ]

    # Replace specified substrings with ""
    replaced_strings = []
    for string in strings:
        for r in replacements:
            string = string.replace(r, "")
        # Replace "DIFFUSION" with "DWI"
        string = string.replace("DIFFUSION", "DWI")
        replaced_strings.append(string)

    return replaced_strings


def get_feats(
    prediction_task,
    features_path,
    labels_path=LABELS_FILE,
    scaler="Standard",
    low_var_thresh=None,
):
    """
    If low_var_thresh=None, no low-variance columns are dropped before applying scaler.
    Otherwise, columns with variance < low_var_thresh * variances.median() are dropped.
    Recommend trying low_var_thresh=0.1, 0.01, 0.001
    """
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

    # Drop cols w/low variance
    if low_var_thresh:
        variances = X.var()
        low_thresh = low_var_thresh * variances.median()
        low_variance_cols = variances[variances < low_thresh].index
        X = X.drop(columns=low_variance_cols)

    # Scale feats
    if scaler == "Standard":
        scaler_obj = StandardScaler()
    elif scaler == "MinMax":
        scaler_obj = MinMaxScaler()
    else:
        scaler_obj = None

    if scaler_obj is not None:
        X = pd.DataFrame(scaler_obj.fit_transform(X), columns=X.columns)

    # Subject IDs
    sub_nos = data["Subject Number"].values

    return X, y, sub_nos
