# %%
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from pathlib import Path
from src.utils import PYRAD_FILE, LABELS_FILE, get_feats
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion
from tqdm import tqdm

coef_paths = {
    'Chr22q': Path('/Users/lgc2035/Documents/research/Meningioma_project/meningioma_radiomics/data/2_modeling/pyradiomics/Chr22q/02-25-2026_17-18-03'),
    'Chr1p': Path('/Users/lgc2035/Documents/research/Meningioma_project/meningioma_radiomics/data/2_modeling/pyradiomics/Chr1p/02-25-2026_17-32-30'),
    'MethylationSubgroup': Path('/Users/lgc2035/Documents/research/Meningioma_project/meningioma_radiomics/data/2_modeling/pyradiomics/MethylationSubgroup/02-24-2026_17-27-31')
}

def get_coefs(pred_task, exp_dir):
    coefs = {}
    if pred_task != "MethylationSubgroup":
        coefs[pred_task] = pd.read_csv(exp_dir / "coefs.csv", index_col=0)
    else:
        coef_files = [f for f in exp_dir.iterdir() if f.name.endswith("coefs.csv")]
        for f in coef_files:
            coef_name = f.name.replace("_coefs.csv", "")
            coefs[coef_name] = pd.read_csv(f, index_col=0)
    return coefs

coefs = {}
feat_lists = {}
for pt, ed in coef_paths.items():
    coefs[pt] = get_coefs(pt, ed)
    feat_lists[pt] = []
    for class_label in coefs[pt]:
        feat_lists[pt].extend(list(coefs[pt][class_label].index))
    feat_lists[pt] = sorted(list(set(feat_lists[pt])))

all_feats = []
for fl in feat_lists:
    all_feats.extend(feat_lists[fl])
all_feats = sorted(list(set(all_feats)))

PREDICTION_TASK = 'Chr22q'
selected_feats = feat_lists[PREDICTION_TASK]
X, y, sub_nos = get_feats(
    prediction_task=PREDICTION_TASK,
    features_path=PYRAD_FILE,
    labels_path=LABELS_FILE,
    scaler=None,
    low_var_thresh=None,
    drop_extra_shape_feats=False,
    remove_correlated_feats=None,
    drop_constant_feats=False
)
X = X[selected_feats]
X.shape

skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
results = {m: {"auc": [], "acc": [], "bal_acc": [], "f1": []} for m in ["xgb", "tabpfn"]}
# results = {m: {"auc": [], "acc": [], "bal_acc": [], "f1": []} for m in ["xgb"]}
folds = list(skf.split(X, y))

# %%
tabpfn_clf = TabPFNClassifier()

for fold, (train_idx, val_idx) in enumerate(tqdm(folds, desc="Folds", unit="fold", total=4)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)

    # tabpfn_clf = TabPFNClassifier.create_default_for_version(ModelVersion.V2)
    fold_models = {
        "xgb":    (XGBClassifier(
                       n_estimators=100, max_depth=3, learning_rate=0.1,
                       subsample=0.8, eval_metric="logloss", random_state=42
                   ), X_train, X_val),
        "tabpfn": (tabpfn_clf, X_train.to_numpy(), X_val.to_numpy()),
    }

    for name, (model, Xtr, Xvl) in tqdm(fold_models.items(), desc=f"  Fold {fold+1} models", 
                                          unit="model", leave=False):
        print("Model fitting...")
        model.fit(Xtr, y_train)
        print("Model predicting...")
        probs = model.predict_proba(Xvl)[:, 1]
        preds = (probs >= 0.5).astype(int)

        results[name]["auc"].append(roc_auc_score(y_val, probs))
        results[name]["acc"].append(accuracy_score(y_val, preds))
        results[name]["bal_acc"].append(balanced_accuracy_score(y_val, preds))
        results[name]["f1"].append(f1_score(y_val, preds))

metrics_to_print = ["auc", "acc", "bal_acc", "f1"]
header = f"{'Model':<10}" + "".join(f"{m:>18}" for m in metrics_to_print)
print(header)
print("-" * (10 + 18 * len(metrics_to_print)))
for name, metrics in results.items():
    row = f"{name:<10}"
    for m in metrics_to_print:
        row += f"  {np.mean(metrics[m]):.3f} ± {np.std(metrics[m]):.3f}"
    print(row)

metrics
# %%
