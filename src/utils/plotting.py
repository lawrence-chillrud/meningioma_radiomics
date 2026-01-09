from itertools import cycle
from matplotlib.patches import Patch
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    auc,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    jaccard_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_curve,
)
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


def shorten_feat_name(orig_name):
    short_name = orig_name
    d = {
        "Maximum": "Max",
        "Minimum": "Min",
        "InterquartileRange": "IQR",
        "2D": "",
        "3D": "",
        "Diameter": "Diam",
        "Small": "Sm",
        "Large": "Lg",
        "High": "Hi",
        "Low": "Lo",
        "Level": "Lvl",
        "Variance": "Var",
        "10Percentile": "P10",
        "90Percentile": "P90",
        "Normalized": "Norm",
    }
    for k, v in d.items():
        short_name = short_name.replace(k, v)
    return short_name


def translate_feat_names(names):
    new_names = []
    for n in names:
        if len(n.split("_")) == 5:
            pulse, annotation, haralick, angle, feat = n.split("_")
            feat = shorten_feat_name(feat)
            readable_annotation = get_segs_roi_key()[
                int(annotation)
            ]  # e.g. Enhancing tumor
            new_names.append(
                f"{readable_annotation} {haralick} {feat} (ang {angle}) on {pulse}"
            )
        else:
            pulse, annotation, feat = n.split("-")  # e.g. T1, 1, firstorder_Kurtosis
            feat_type, feat_name = feat.split("_")  # e.g. firstorder, Kurtosis
            feat_name = shorten_feat_name(feat_name)
            readable_annotation = get_segs_roi_key()[
                int(annotation)
            ]  # e.g. Enhancing tumor
            new_names.append(f"{readable_annotation} {feat_name} on {pulse}")

    return new_names


def plot_confusion_matrix(y_true, y_pred, class_ids):
    conf_matrix = confusion_matrix(y_true, y_pred)
    conf_matrix_norm = conf_matrix / conf_matrix.sum(axis=1, keepdims=True)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    plt.figure(figsize=(4, 4))
    ax = sns.heatmap(
        conf_matrix_norm,
        annot=False,
        cmap="viridis",
        cbar=False,
        xticklabels=class_ids,
        yticklabels=class_ids,
        vmax=1,
        vmin=0,
    )

    # Overlay custom annotations
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            val_pct = conf_matrix_norm[i, j] * 100
            val_count = conf_matrix[i, j]
            ax.text(
                j + 0.5,
                i + 0.5,
                f"{val_pct:.2f}%\n(n={val_count})",
                ha="center",
                va="center",
                color="white",
            )

    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    # plt.title(f"Balanced Accuracy = {balanced_acc*100:.2f}%")
    plt.show()
    plt.close()

    return conf_matrix, balanced_acc


def plot_bootstrap_roc(
    y_true, y_score, n_boot=10_000, stratified=True, random_state=None
):
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    # Base ROC
    fpr_base, tpr_base, _ = roc_curve(y_true, y_score)
    auc_base = auc(fpr_base, tpr_base)

    # Prepare indices
    if stratified:
        pos_idx = np.where(y_true == 1)[0]
        neg_idx = np.where(y_true == 0)[0]
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            stratified = False

    # Common FPR grid (to aggregate bootstrap ROC curves)
    fpr_grid = np.linspace(0, 1, 1000)
    tprs = []
    aucs = []

    for _ in range(n_boot):
        if stratified:
            samp_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
            samp_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
            samp_idx = np.concatenate([samp_pos, samp_neg])
        else:
            samp_idx = rng.choice(len(y_true), size=len(y_true), replace=True)

        fpr_b, tpr_b, _ = roc_curve(y_true[samp_idx], y_score[samp_idx])
        aucs.append(auc(fpr_b, tpr_b))

        # Interpolate TPR onto common FPR grid
        tpr_interp = np.interp(fpr_grid, fpr_b, tpr_b)
        tprs.append(tpr_interp)

    tprs = np.array(tprs)
    aucs = np.array(aucs)

    # CI bands for ROC curve
    # lower_band = np.percentile(tprs, 2.5, axis=0)
    # upper_band = np.percentile(tprs, 97.5, axis=0)

    # ± 1 stddev bands
    tpr_mean = tprs.mean(axis=0)
    tpr_std = tprs.std(axis=0)

    lower_band = tpr_mean - tpr_std
    upper_band = tpr_mean + tpr_std

    auc_mean = np.mean(aucs)
    auc_std = np.std(aucs)
    auc_label = f"AUC = {auc_base:.3f} ± {auc_std:.3f}"

    plt.figure(figsize=(4.8, 4.8))
    plt.plot(fpr_base, tpr_base, lw=2, label=auc_label, color="black")
    plt.plot([0, 1], [0, 1], linestyle="--", color="black")

    # Confidence band
    plt.fill_between(
        fpr_grid, lower_band, upper_band, alpha=0.25, color="grey", label="±1 std. dev."
    )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    # plt.title("ROC Curve with Bootstrap Confidence Bands")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()
    plt.close()

    return auc_base, auc_mean, fpr_base, tpr_base


def plot_binary_results(probs, y_true, class_ids):
    """Plot ROC curve, confusion matrix, and metrics table for binary classification tasks. Returns the ROC AUC score."""
    probs = np.stack(probs).squeeze()

    # Plot 1/3: ROC curve
    roc_auc, auc_mean, _, _ = plot_bootstrap_roc(y_true, probs[:, 1])

    # Plot 2/3: Confusion matrix
    y_pred = np.argmax(probs, axis=1)
    conf_matrix, balanced_accuracy = plot_confusion_matrix(y_true, y_pred, class_ids)

    tn, fp, fn, tp = conf_matrix.ravel()

    # Plot 3/3: Metrics table
    metrics = {
        "AUC": roc_auc,
        "AUC (mean)": auc_mean,
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


def plot_multiclass_bootstrap_roc(
    y_true, probs, class_ids, n_boot=10_000, stratified=True, random_state=None
):
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)

    n_classes = y_true.shape[1]
    fpr_grid = np.linspace(0, 1, 1000)

    fig, ax = plt.subplots(figsize=(6.4, 6.4))

    for c in range(n_classes):
        fpr_c, tpr_c, _ = roc_curve(y_true[:, c], probs[:, c])
        auc_c = auc(fpr_c, tpr_c)

        # Stratified indices
        pos_idx = np.where(y_true == 1)[0]
        neg_idx = np.where(y_true == 0)[0]
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            stratified = False

        boot_tprs = []
        boot_aucs = []

        for _ in range(n_boot):
            if stratified:
                samp_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
                samp_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
                samp_idx = np.concatenate([samp_pos, samp_neg])
            else:
                samp_idx = rng.choice(len(y_true), size=len(y_true), replace=True)
            fpr_b, tpr_b, _ = roc_curve(y_true[samp_idx, c], probs[samp_idx, c])
            tpr_interp = np.interp(fpr_grid, fpr_b, tpr_b)
            boot_tprs.append(tpr_interp)
            boot_aucs.append(auc(fpr_b, tpr_b))

        boot_tprs = np.array(boot_tprs)
        boot_aucs = np.array(boot_aucs)
        # lower = np.percentile(boot_tprs, 2.5, axis=0)
        # upper = np.percentile(boot_tprs, 97.5, axis=0)
        tpr_mean = boot_tprs.mean(axis=0)
        tpr_std = boot_tprs.std(axis=0)

        lower_band = tpr_mean - tpr_std
        upper_band = tpr_mean + tpr_std

        auc_mean = np.mean(boot_aucs)
        auc_std = np.std(boot_aucs)
        auc_label = f"{class_ids[c]} AUC = {auc_c:.3f} ± {auc_std:.3f}"

        plt.plot(fpr_c, tpr_c, lw=2, label=auc_label)

        # Confidence band
        plt.fill_between(
            fpr_grid, lower_band, upper_band, alpha=0.25
        )

    ax.plot([0, 1], [0, 1], linestyle="--", color="black")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    # ax.set_title("One-vs-Rest ROC Curves with Bootstrap CIs")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_multiclass_results(probs, y_true, class_ids, prediction_task):
    """Plot ROC curve, confusion matrix, and metrics table for multiclass classification tasks. Expects y_true to be one-hot encoded."""
    n_classes = len(np.unique(y_true))
    probs = np.stack(probs).squeeze()
    y_true = label_binarize(y_true, classes=np.unique(y_true))

    plot_multiclass_bootstrap_roc(y_true, probs, class_ids)

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

    # plt.plot(
    #     fpr["micro"],
    #     tpr["micro"],
    #     label=f"micro-average ROC curve (AUC = {roc_auc['micro']:.2f})",
    #     color="deeppink",
    #     linestyle=":",
    #     linewidth=4,
    # )

    # plt.plot(
    #     fpr["macro"],
    #     tpr["macro"],
    #     label=f"macro-average ROC curve (AUC = {roc_auc['macro']:.2f})",
    #     color="navy",
    #     linestyle=":",
    #     linewidth=4,
    # )

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
    # plt.xlabel("Feature coefficient\n(across all NLTOCV testing folds)")
    plt.xlabel("Coefficient")
    plt.show()
    plt.close()


def plot_var_exp(data):
    plt.figure()
    sns.barplot(data * 100, orient="h")
    plt.yticks(
        ticks=np.arange(len(data)),
        labels=translate_feat_names(data.index),
        rotation=0,
        ha="right",
        rotation_mode="anchor",
    )
    plt.xlabel("% variance in model coefficients explained")
    plt.show()
    plt.close()


def plot_var_exp2(data):
    data_orig = data.values[::-1] * 100
    data_cum = data.cumsum().values[::-1] * 100
    y = np.arange(len(data))

    plt.figure()
    plt.barh(y, data_cum, color="white", edgecolor="black", label="Cumulative")
    plt.barh(y, data_orig, color="gray", edgecolor="black", label="Individual")
    plt.yticks(y, translate_feat_names(data.index[::-1]), rotation=0, ha="right")
    plt.xlabel("% variance in model coefficients explained")
    plt.legend()
    plt.show()
    plt.close()


def plot_coef_boxplot2(
    data_coef, data_var, adaptive_legend=False, print_nums=True, figsize=(15, 6)
):
    data_orig = data_var.values * 100
    data_cum = data_var.cumsum().values * 100
    y = np.arange(len(data_var))

    means = data_coef.mean()
    norm = plt.Normalize(vmin=-max(abs(means)), vmax=max(abs(means)))
    cmap = sns.color_palette("vlag", as_cmap=True)
    colors = cmap(norm(means.values)).tolist()

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize, sharey=True)

    sns.boxplot(data=data_coef, orient="h", palette=colors, ax=axes[0])
    axes[0].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[0].set_yticks(np.arange(len(data_coef.columns)))
    axes[0].set_yticklabels(
        translate_feat_names(data_coef.columns), rotation=0, ha="right"
    )
    axes[0].set_xlabel("Coefficient")

    axes[1].barh(y, data_cum, color="white", edgecolor="black", label="Cumulative")
    axes[1].barh(y, data_orig, color=colors, edgecolor="black", label="Individual")
    axes[1].set_xlabel("% variance in model coefficients explained")

    if print_nums:
        for i, (ind, cum) in enumerate(zip(data_orig, data_cum)):
            if i == 0:
                axes[1].text(ind + 1, i, f"{ind:.1f}", va="center", color=colors[i])
            else:
                axes[1].text(ind + 1, i, f"{ind:.1f}", va="center", color=colors[i])
                axes[1].text(cum + 1, i, f"{cum:.1f}", va="center")
        axes[1].margins(x=0.25)

    if adaptive_legend:
        axes[1].legend()
        handles, labels = axes[1].get_legend_handles_labels()
        axes[1].legend(handles[::-1], labels[::-1])
    else:
        leg_handles = [
            Patch(facecolor="gray", edgecolor="black", label="Individual"),
            Patch(facecolor="white", edgecolor="black", label="Cumulative"),
        ]
        axes[1].legend(handles=leg_handles)

    fig.subplots_adjust(right=0.88)
    plt.tight_layout()
    plt.show()


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
        cg.ax_heatmap.set_yticks(
            ticks=np.arange(len(data)) + 0.5, labels=translate_feat_names(data.index)
        )
        cg.ax_heatmap.set_xticks(
            ticks=np.arange(len(data)) + 0.5,
            labels=translate_feat_names(data.index),
            rotation=-45,
            ha="left",
        )
    plt.show()
    plt.close()
