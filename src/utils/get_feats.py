import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm
from .paths import LABELS_FILE


def remove_correlated_features(
    df: pd.DataFrame, threshold: float = 0.95, verbose: bool = False
) -> pd.DataFrame:
    """
    Remove features with an absolute Pearson correlation greater than `threshold`
    to any earlier feature in the DataFrame.

    When a correlated pair is found, the *second* feature (the one appearing
    later in the column order) is dropped — the first one is kept as the
    "reference" feature.

    Parameters
    ----------
    df        : Input DataFrame (numeric columns only are evaluated).
    threshold : Absolute correlation cutoff (exclusive upper bound kept).
    verbose   : Whether to print which cols were dropped and why.

    Returns
    -------
    A new DataFrame with redundant columns removed, plus a printed summary.
    """
    if not (0 < threshold <= 1):
        raise ValueError("threshold must be in the range (0, 1].")

    # Work only with numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # Compute the absolute correlation matrix
    corr_matrix = numeric_df.corr().abs()

    # Use the upper triangle to avoid double-counting pairs
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Identify columns to drop: any column that has a correlation > threshold
    # with at least one earlier column
    cols_to_drop = [
        col
        for col in tqdm(
            upper_tri.columns,
            total=len(upper_tri.columns),
            desc="Dropping closely correlated vars...",
            ncols=120,
            leave=False,
        )
        if any(upper_tri[col] > threshold)
    ]

    # Report which pairs triggered the removal
    print(f"Correlation threshold : |r| > {threshold}")
    if verbose:
        print(f"Columns dropped       : {cols_to_drop}\n")
        print("Correlated pairs that caused removal:")
        for col in cols_to_drop:
            correlated_with = upper_tri.index[upper_tri[col] > threshold].tolist()
            for ref in correlated_with:
                print(f"  '{ref}' <--> '{col}'  |r| = {corr_matrix.loc[ref, col]:.4f}")
        print()

    # Reconstruct the DataFrame: keep non-numeric + surviving numeric columns
    surviving_numeric = [c for c in numeric_df.columns if c not in cols_to_drop]
    result = pd.concat([df[non_numeric], df[surviving_numeric]], axis=1)

    print(f"Original shape : {df.shape}")
    print(f"Reduced shape  : {result.shape}")
    return result


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
    drop_extra_shape_feats=True,
    remove_correlated_feats=0.6,
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

    # Since shape feats do not vary w/pulse seq, drop them for all but T1
    # for easier optimization / to avoid multiple of the same shape feats
    # sharing variance across.
    if drop_extra_shape_feats:
        X = X.drop(columns=[f for f in X.columns if ("shape" in f) and ("T1" not in f)])

    if remove_correlated_feats:
        if "collage" in str(features_path).lower():
            X = X[sorted(X.columns.to_list())]
        X = remove_correlated_features(X, threshold=remove_correlated_feats)

    return X, y, sub_nos
