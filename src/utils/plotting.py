from itertools import cycle

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (RocCurveDisplay, accuracy_score, auc,
                             balanced_accuracy_score, confusion_matrix,
                             f1_score, jaccard_score, matthews_corrcoef,
                             precision_score, recall_score, roc_curve)
from sklearn.preprocessing import label_binarize

from .get_segs import get_segs_roi_key

FONT_SIZE = 14

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


def translate_feat_names(names):
    new_names = []
    for n in names:
        pulse, annotation, feat = n.split("-")  # e.g. T1, 1, firstorder_Kurtosis
        feat_type, feat_name = feat.split("_")  # e.g. firstorder, Kurtosis
        readable_annotation = get_segs_roi_key()[
            int(annotation)
        ]  # e.g. Enhancing tumor
        new_names.append(f"{readable_annotation}'s {feat_name} on {pulse}")

    return new_names


def plot_confusion_matrix(y_true, y_pred, class_ids):
    """Plots a confusion matrix given y_true labels and y_pred predictions and returns the matrix."""
    conf_matrix = confusion_matrix(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="g",
        cmap="viridis",
        cbar=False,
        xticklabels=class_ids,
        yticklabels=class_ids,
    )
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title(f"Balanced Accuracy = {balanced_accuracy*100:.2f}%")
    plt.show()
    plt.close()

    return conf_matrix, balanced_accuracy


def plot_binary_results(probs, y_true, class_ids):
    """Plot ROC curve, confusion matrix, and metrics table for binary classification tasks. Returns the ROC AUC score."""
    probs = np.stack(probs).squeeze()

    fpr, tpr, _ = roc_curve(y_true, probs[:, 1])
    roc_auc = auc(fpr, tpr)

    # Plot 1/3: ROC curve:
    plt.plot(fpr, tpr, color="tab:orange", lw=2, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], color="tab:blue", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve")
    plt.legend(loc="lower right")

    plt.tight_layout()
    plt.show()
    plt.close()

    # Plot 2/3: Confusion matrix
    y_pred = np.argmax(probs, axis=1)
    conf_matrix, balanced_accuracy = plot_confusion_matrix(y_true, y_pred, class_ids)

    tn, fp, fn, tp = conf_matrix.ravel()

    # Plot 3/3: Metrics table
    metrics = {
        "AUC": roc_auc,
        "Binary F1": f1_score(y_true, y_pred, average="binary"),
        "Weighted F1": f1_score(y_true, y_pred, average="weighted"),
        "Binary Precision": precision_score(y_true, y_pred, average="binary"),
        "Weighted Precision": precision_score(y_true, y_pred, average="weighted"),
        "Binary Recall (Sensitivity)": recall_score(y_true, y_pred, average="binary"),
        "Weighted Recall (Sensitivity)": recall_score(
            y_true, y_pred, average="weighted"
        ),
        "Specificity": tn / (tn + fp),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy,
        "MCC": matthews_corrcoef(y_true, y_pred),
        "Binary Jaccard": jaccard_score(y_true, y_pred, average="binary"),
        "Weighted Jaccard": jaccard_score(y_true, y_pred, average="weighted"),
    }

    return metrics


def plot_multiclass_results(probs, y_true, class_ids, prediction_task):
    """Plot ROC curve, confusion matrix, and metrics table for multiclass classification tasks. Expects y_true to be one-hot encoded."""
    n_classes = len(np.unique(y_true))
    probs = np.stack(probs).squeeze()
    y_true = label_binarize(y_true, classes=np.unique(y_true))

    fpr, tpr, roc_auc = dict(), dict(), dict()

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr_grid = np.linspace(0.0, 1.0, 1000)

    # Interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(fpr_grid)

    for i in range(n_classes):
        mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation

    # Average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = fpr_grid
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot 1/3: ROC curve
    _, ax = plt.subplots(figsize=(9, 9))

    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label=f"micro-average ROC curve (AUC = {roc_auc['micro']:.2f})",
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label=f"macro-average ROC curve (AUC = {roc_auc['macro']:.2f})",
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    colors = cycle(sns.color_palette())
    for i, color, class_id in zip(range(n_classes), colors, class_ids):
        RocCurveDisplay.from_predictions(
            y_true[:, i],
            probs[:, i],
            name=f"ROC curve for {class_id}",
            color=color,
            ax=ax,
            plot_chance_level=(i == n_classes - 1),
        )

    _ = ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=f"{prediction_task}: One-vs-Rest ROC Curves",
    )

    plt.show()

    plt.close()

    # Plot 2/3: Confusion matrix
    y_true = np.argmax(y_true, axis=1)
    y_pred = np.argmax(probs, axis=1)
    plot_confusion_matrix(y_true, y_pred, class_ids)

    # Plot 3/3: Metrics table
    metrics = {
        "Macro AUC": roc_auc["macro"],
        "Micro AUC": roc_auc["micro"],
        "Macro F1": f1_score(y_true, y_pred, average="macro"),
        "Micro F1": f1_score(y_true, y_pred, average="micro"),
        "Weighted F1": f1_score(y_true, y_pred, average="weighted"),
        "Macro Precision": precision_score(y_true, y_pred, average="macro"),
        "Micro Precision": precision_score(y_true, y_pred, average="micro"),
        "Weighted Precision": precision_score(y_true, y_pred, average="weighted"),
        "Macro Recall (Sensitivity)": recall_score(y_true, y_pred, average="macro"),
        "Micro Recall (Sensitivity)": recall_score(y_true, y_pred, average="micro"),
        "Weighted Recall (Sensitivity)": recall_score(
            y_true, y_pred, average="weighted"
        ),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "Macro Jaccard": jaccard_score(y_true, y_pred, average="macro"),
        "Micro Jaccard": jaccard_score(y_true, y_pred, average="micro"),
        "Weighted Jaccard": jaccard_score(y_true, y_pred, average="weighted"),
    }

    return metrics


def plot_heatmap(data):
    plt.figure()
    sns.heatmap(data, cmap="vlag", cbar_kws={"label": "Coefficient"}, center=0)
    plt.xlabel("NLTOCV testing fold")
    plt.xticks([])
    plt.yticks(
        ticks=np.arange(len(data.index)) + 0.5, labels=translate_feat_names(data.index)
    )
    plt.show()
    plt.close()


def plot_coef_boxplot(data):
    means = data.mean()
    norm = plt.Normalize(vmin=-max(abs(means)), vmax=max(abs(means)))
    cmap = sns.color_palette("vlag", as_cmap=True)
    colors = cmap(norm(means.values))

    plt.figure()
    sns.boxplot(data=data, orient="h", palette=colors)
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.yticks(
        ticks=np.arange(len(data.columns)),
        labels=translate_feat_names(data.columns),
        rotation=0,
        ha="right",
    )
    plt.xlabel("Feature coefficient\n(across all NLTOCV testing folds)")
    plt.show()
    plt.close()


def plot_var_exp(data):
    plt.figure()
    sns.barplot(data)
    plt.xticks(
        ticks=np.arange(len(data)),
        labels=translate_feat_names(data.index),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )
    plt.ylabel("Proportion of variance explained")
    plt.title(
        f"Features needed to cumulatively explain ≤ 0.95 of\nvariance in model coefficients (n={len(data)})"
    )
    plt.show()
    plt.close()


def plot_corr_matrix(
    data, custom_labels=True, annot=False, cbar_pos=(0.02, 0.52, 0.05, 0.18)
):
    plt.figure()
    cg = sns.clustermap(
        data,
        row_cluster=False,
        col_cluster=False,
        cmap="vlag",
        cbar_pos=cbar_pos,
        figsize=(10, 10),
        annot=annot,
        vmin=-1,
        vmax=1,
    )
    if custom_labels:
        cg.ax_heatmap.set_xticklabels(
            translate_feat_names(data.index), rotation=45, ha="right"
        )
        cg.ax_heatmap.set_yticklabels(
            translate_feat_names(data.index), rotation=-45, ha="left"
        )
    plt.show()
    plt.close()
