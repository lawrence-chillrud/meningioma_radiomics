from itertools import cycle
from matplotlib.patches import Patch
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
from sklearn.calibration import calibration_curve
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
from scipy import stats
import matplotlib.patches as mpatches
from imblearn.metrics import specificity_score
from .get_segs import get_segs_roi_key

FONT_SIZE = 14

mpl.rcParams.update(
    {
        "font.size": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE - 2, # for MethylationSubgroup tasks,
        "figure.titlesize": FONT_SIZE,
        "figure.dpi": 600,
        "savefig.dpi": 600,
    }
)


def shorten_feat_name(orig_name):
    short_name = orig_name
    d = {
        "Maximum3DDiameter": "Max3DDiam",
        "Maximum2DDiameterSlice": "MaxSagittalDiam",
        "Maximum2DDiameterColumn": "MaxCoronalDiam",
        "Maximum2DDiameterRow": "MaxAxialDiam",
        "Maximum": "Max",
        "Minimum": "Min",
        "InterquartileRange": "IQR",
        "2D": "",
        # "3D": "",
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
        "SizeZoneNonUniformityNorm": "SZNN",
        "Information": "Info",
        "Correlation": "Corr",
        "Second": "2nd",
        "Maximal": "Max",
        "Coefficient": "Coef",
    }
    for k, v in d.items():
        short_name = short_name.replace(k, v)
    return short_name


def translate_feat_names(names):
    new_names = []
    for n in names:
        if len(n.split("_")) == 5:
            pulse, annotation, haralick, angle, feat = n.split("_")
            pulse = pulse.replace("T1", "CET1")
            feat = shorten_feat_name(feat)
            haralick = shorten_feat_name(haralick)
            readable_annotation = get_segs_roi_key()[
                int(annotation)
            ].upper()  # e.g. Enhancing tumor
            new_names.append(
                f"{readable_annotation} {haralick} {feat} (ang {angle}) on {pulse}"
            )
        else:
            pulse, annotation, feat = n.split("-")  # e.g. T1, 1, firstorder_Kurtosis
            pulse = pulse.replace("T1", "CET1")
            feat_type, feat_name = feat.split("_")  # e.g. firstorder, Kurtosis
            feat_name = shorten_feat_name(feat_name)
            readable_annotation = get_segs_roi_key()[
                int(annotation)
            ].upper()  # e.g. Enhancing tumor
            if feat_type != "shape":
                new_names.append(f"{readable_annotation} {feat_name} on {pulse}")
            else:
                new_names.append(f"{readable_annotation} {feat_name}")

    return new_names


def _bootstrap_stratified_indices(y_true, rng):
    y_true = np.asarray(y_true)
    sampled_idx = []

    for cls in np.unique(y_true):
        cls_idx = np.flatnonzero(y_true == cls)
        sampled_idx.append(rng.choice(cls_idx, size=len(cls_idx), replace=True))

    sampled_idx = np.concatenate(sampled_idx)
    rng.shuffle(sampled_idx)
    return sampled_idx


def _bootstrap_ci(values, alpha=0.05):
    values = np.asarray(values, dtype=float)
    mean = float(np.nanmean(values))
    lower, upper = np.nanpercentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return mean, float(lower), float(upper)


def _format_bootstrap_ci(values, digits=3):
    mean, lower, upper = _bootstrap_ci(values)
    return f"{mean:.{digits}f} (95% CI {lower:.{digits}f}-{upper:.{digits}f})"


def _bootstrap_binary_metric_table(y_true, probs, n_boot=10_000, random_state=0):
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    rng = np.random.default_rng(random_state)

    metric_samples = {
        "AUC": [],
        "AUC (mean)": [],
        "Binary F1": [],
        "Weighted F1": [],
        "Binary Precision": [],
        "Weighted Precision": [],
        "Binary Recall (Sensitivity)": [],
        "Weighted Recall (Sensitivity)": [],
        "Specificity": [],
        "Accuracy": [],
        "Balanced Accuracy": [],
        "MCC": [],
        "Binary Jaccard": [],
        "Weighted Jaccard": [],
        "Youden's J": [],
    }

    for _ in range(n_boot):
        idx = _bootstrap_stratified_indices(y_true, rng)
        y_true_b = y_true[idx]
        probs_b = probs[idx]
        y_pred_b = np.argmax(probs_b, axis=1)

        fpr_b, tpr_b, _ = roc_curve(y_true_b, probs_b[:, 1])
        auc_b = auc(fpr_b, tpr_b)

        metric_samples["AUC"].append(auc_b)
        metric_samples["AUC (mean)"].append(auc_b)
        metric_samples["Binary F1"].append(
            f1_score(y_true_b, y_pred_b, average="binary", zero_division=0)
        )
        metric_samples["Weighted F1"].append(
            f1_score(y_true_b, y_pred_b, average="weighted", zero_division=0)
        )
        metric_samples["Binary Precision"].append(
            precision_score(y_true_b, y_pred_b, average="binary", zero_division=0)
        )
        metric_samples["Weighted Precision"].append(
            precision_score(y_true_b, y_pred_b, average="weighted", zero_division=0)
        )
        metric_samples["Binary Recall (Sensitivity)"].append(
            recall_score(y_true_b, y_pred_b, average="binary", zero_division=0)
        )
        metric_samples["Weighted Recall (Sensitivity)"].append(
            recall_score(y_true_b, y_pred_b, average="weighted", zero_division=0)
        )
        tn, fp, _, _ = confusion_matrix(y_true_b, y_pred_b).ravel()
        metric_samples["Specificity"].append(tn / (tn + fp))
        metric_samples["Accuracy"].append(accuracy_score(y_true_b, y_pred_b))
        metric_samples["Balanced Accuracy"].append(
            balanced_accuracy_score(y_true_b, y_pred_b)
        )
        metric_samples["MCC"].append(matthews_corrcoef(y_true_b, y_pred_b))
        metric_samples["Binary Jaccard"].append(
            jaccard_score(y_true_b, y_pred_b, average="binary", zero_division=0)
        )
        metric_samples["Weighted Jaccard"].append(
            jaccard_score(y_true_b, y_pred_b, average="weighted", zero_division=0)
        )
        metric_samples["Youden's J"].append(np.max(tpr_b - fpr_b))

    return {name: _format_bootstrap_ci(samples) for name, samples in metric_samples.items()}


def _bootstrap_multiclass_metric_table(y_true, probs, class_ids, n_boot=10_000, random_state=0):
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    y_true_bin = label_binarize(y_true, classes=np.arange(probs.shape[1]))
    rng = np.random.default_rng(random_state)

    metric_samples = {
        "Macro AUC": [],
        "Micro AUC": [],
        "Macro F1": [],
        "Micro F1": [],
        "Weighted F1": [],
        "Macro Precision": [],
        "Micro Precision": [],
        "Weighted Precision": [],
        "Macro Recall (Sensitivity)": [],
        "Micro Recall (Sensitivity)": [],
        "Weighted Recall (Sensitivity)": [],
        "Macro Specificity": [],
        "Micro Specificity": [],
        "Weighted Specificity": [],
        "Accuracy": [],
        "Balanced Accuracy": [],
        "MCC": [],
        "Macro Jaccard": [],
        "Micro Jaccard": [],
        "Weighted Jaccard": [],
    }

    for class_id in class_ids:
        metric_samples[f"Youden's J ({class_id})"] = []

    n_classes = probs.shape[1]

    for _ in range(n_boot):
        idx = _bootstrap_stratified_indices(y_true, rng)
        y_true_b = y_true[idx]
        y_true_bin_b = y_true_bin[idx]
        probs_b = probs[idx]
        y_pred_b = np.argmax(probs_b, axis=1)

        class_aucs = []
        for c in range(n_classes):
            fpr_c, tpr_c, _ = roc_curve(y_true_bin_b[:, c], probs_b[:, c])
            class_aucs.append(auc(fpr_c, tpr_c))
            metric_samples[f"Youden's J ({class_ids[c]})"].append(np.max(tpr_c - fpr_c))

        fpr_micro, tpr_micro, _ = roc_curve(y_true_bin_b.ravel(), probs_b.ravel())

        metric_samples["Macro AUC"].append(float(np.mean(class_aucs)))
        metric_samples["Micro AUC"].append(auc(fpr_micro, tpr_micro))
        metric_samples["Macro F1"].append(
            f1_score(y_true_b, y_pred_b, average="macro", zero_division=0)
        )
        metric_samples["Micro F1"].append(
            f1_score(y_true_b, y_pred_b, average="micro", zero_division=0)
        )
        metric_samples["Weighted F1"].append(
            f1_score(y_true_b, y_pred_b, average="weighted", zero_division=0)
        )
        metric_samples["Macro Precision"].append(
            precision_score(y_true_b, y_pred_b, average="macro", zero_division=0)
        )
        metric_samples["Micro Precision"].append(
            precision_score(y_true_b, y_pred_b, average="micro", zero_division=0)
        )
        metric_samples["Weighted Precision"].append(
            precision_score(y_true_b, y_pred_b, average="weighted", zero_division=0)
        )
        metric_samples["Macro Recall (Sensitivity)"].append(
            recall_score(y_true_b, y_pred_b, average="macro", zero_division=0)
        )
        metric_samples["Micro Recall (Sensitivity)"].append(
            recall_score(y_true_b, y_pred_b, average="micro", zero_division=0)
        )
        metric_samples["Weighted Recall (Sensitivity)"].append(
            recall_score(y_true_b, y_pred_b, average="weighted", zero_division=0)
        )
        metric_samples["Macro Specificity"].append(
            specificity_score(y_true_b, y_pred_b, average="macro")
        )
        metric_samples["Micro Specificity"].append(
            specificity_score(y_true_b, y_pred_b, average="micro")
        )
        metric_samples["Weighted Specificity"].append(
            specificity_score(y_true_b, y_pred_b, average="weighted")
        )
        metric_samples["Accuracy"].append(accuracy_score(y_true_b, y_pred_b))
        metric_samples["Balanced Accuracy"].append(
            balanced_accuracy_score(y_true_b, y_pred_b)
        )
        metric_samples["MCC"].append(matthews_corrcoef(y_true_b, y_pred_b))
        metric_samples["Macro Jaccard"].append(
            jaccard_score(y_true_b, y_pred_b, average="macro", zero_division=0)
        )
        metric_samples["Micro Jaccard"].append(
            jaccard_score(y_true_b, y_pred_b, average="micro", zero_division=0)
        )
        metric_samples["Weighted Jaccard"].append(
            jaccard_score(y_true_b, y_pred_b, average="weighted", zero_division=0)
        )

    return {name: _format_bootstrap_ci(samples) for name, samples in metric_samples.items()}


def plot_confusion_matrix(y_true, y_pred, class_ids, plot=False, save_path=None):
    conf_matrix = confusion_matrix(y_true, y_pred)
    conf_matrix_norm = conf_matrix / conf_matrix.sum(axis=1, keepdims=True)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    if plot:
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
        fig = ax.figure
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=600)
        plt.show()
        plt.close()

    return conf_matrix, balanced_acc


def plot_bootstrap_roc(
    y_true,
    y_score,
    n_boot=10_000,
    stratified=True,
    random_state=None,
    plot=False,
    save_path=None,
):
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    # Base ROC
    fpr_base, tpr_base, thresholds = roc_curve(y_true, y_score)
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
        tpr_interp = np.interp(fpr_grid, fpr_b, tpr_b)
        tprs.append(tpr_interp)

    tprs = np.array(tprs)
    aucs = np.array(aucs)

    # ± 1 stddev bands
    tpr_mean = tprs.mean(axis=0)
    tpr_std = tprs.std(axis=0)
    lower_band = tpr_mean - tpr_std
    upper_band = tpr_mean + tpr_std

    auc_mean = np.mean(aucs)
    auc_std = np.std(aucs)
    auc_label = f"AUC = {auc_base:.3f} ± {auc_std:.3f}"

    # Point estimate: optimal threshold by Youden's J (TPR - FPR)
    j_scores = tpr_base - fpr_base
    youdens_j = max(j_scores)
    opt_idx = np.argmax(j_scores)
    opt_fpr = fpr_base[opt_idx]
    opt_tpr = tpr_base[opt_idx]

    if plot:
        fig, ax = plt.subplots(figsize=(4.8, 4.8))
        ax.plot(fpr_base, tpr_base, lw=2, label=auc_label, color="black")
        ax.plot([0, 1], [0, 1], linestyle="--", color="black")
        ax.fill_between(
            fpr_grid,
            lower_band,
            upper_band,
            alpha=0.25,
            color="grey",
            label="±1 std. dev.",
        )
        # ── star marker at the optimal operating point ──────────────────
        ax.scatter(
            opt_fpr, opt_tpr,
            marker="*",
            s=200,
            color="black",
            zorder=5,
            label=f"Op. point",
            # label=f"Op. point (FPR={opt_fpr:.2f}, TPR={opt_tpr:.2f})",
        )
        # ────────────────────────────────────────────────────────────────
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=600)
        plt.show()
        plt.close()

    return auc_base, auc_mean, fpr_base, tpr_base, youdens_j


def plot_calibration_curve(
    y_true,
    y_prob,
    n_bins=10,
    plot=True,
    figsize=(4.8, 4.8),
    model_label="Our model",
    save_path=None,
):
    """Plot a calibration curve against the ideal y=x reference line."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    frac_pos, mean_pred = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="quantile"
    )

    if plot:
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot([0, 1], [0, 1], linestyle="--", color="black", label="Calibrated model")
        ax.plot(
            mean_pred,
            frac_pos,
            color="red",
            linewidth=2,
            label=model_label,
        )
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc="best")
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=600)
        plt.show()
        plt.close()

    return mean_pred, frac_pos


def plot_binary_results(
    probs, y_true, class_ids, prediction_task=None, plot=False, save_dir=None
):
    """Plot ROC curve, confusion matrix, and a bootstrapped metrics table for binary classification tasks."""
    probs = np.stack(probs).squeeze()
    y_true = np.asarray(y_true)
    save_dir = Path(save_dir) if save_dir else None
    prediction_task = prediction_task or "Our model"

    # Plot 1/3: ROC curve
    if plot:
        plot_bootstrap_roc(
            y_true,
            probs[:, 1],
            plot=plot,
            save_path=save_dir / "auroc_bootstrap.png" if save_dir else None,
        )

    # Plot 2/3: Confusion matrix
    y_pred = np.argmax(probs, axis=1)
    if plot:
        plot_confusion_matrix(
            y_true,
            y_pred,
            class_ids,
            plot=plot,
            save_path=save_dir / "confusion_matrix.png" if save_dir else None,
        )

    # Plot 3/3: Calibration curve
    plot_calibration_curve(
        y_true,
        probs[:, 1],
        plot=plot,
        model_label=prediction_task,
        save_path=save_dir / "calibration.png" if save_dir else None,
    )

    # Plot 4/4: Metrics table
    metrics = _bootstrap_binary_metric_table(y_true, probs)

    return metrics


def plot_multiclass_bootstrap_roc(
    y_true,
    probs,
    class_ids,
    n_boot=10_000,
    stratified=True,
    random_state=None,
    save_path=None,
):
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)

    n_classes = y_true.shape[1]
    fpr_grid = np.linspace(0, 1, 1000)

    fig, ax = plt.subplots(figsize=(4.8, 4.8))

    youden_js = []
    op_points = []  # store (opt_fpr, opt_tpr, colour) for each class

    for c in range(n_classes):
        fpr_c, tpr_c, _ = roc_curve(y_true[:, c], probs[:, c])
        auc_c = auc(fpr_c, tpr_c)

        # Stratified indices
        pos_idx = np.where(y_true[:, c] == 1)[0]
        neg_idx = np.where(y_true[:, c] == 0)[0]
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
        tpr_mean = boot_tprs.mean(axis=0)
        tpr_std = boot_tprs.std(axis=0)
        lower_band = tpr_mean - tpr_std
        upper_band = tpr_mean + tpr_std

        auc_mean = np.mean(boot_aucs)
        auc_std = np.std(boot_aucs)
        auc_label = f"{class_ids[c]} AUC = {auc_c:.3f} ± {auc_std:.3f}"

        # Youden's J for this class
        j_scores = tpr_c - fpr_c
        opt_idx = np.argmax(j_scores)
        youden_js.append(j_scores[opt_idx])
        op_points.append((fpr_c[opt_idx], tpr_c[opt_idx]))

        line, = ax.plot(fpr_c, tpr_c, lw=2, label=auc_label)
        colour = line.get_color()
        ax.fill_between(fpr_grid, lower_band, upper_band, alpha=0.25, color=colour)

        # Star at the operating point, coloured to match the curve
        ax.scatter(
            fpr_c[opt_idx], tpr_c[opt_idx],
            marker="*",
            s=200,
            color=colour,
            zorder=5,
        )

    ax.plot([0, 1], [0, 1], linestyle="--", color="black")

    # Single black star legend entry for all operating points
    ax.scatter([], [], marker="*", s=200, color="black", label="Op. point")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=600)
    plt.show()
    plt.close()

    return youden_js, op_points


def plot_multiclass_results(
    probs, y_true, class_ids, prediction_task, plot=False, save_dir=None
):
    """Plot ROC curve, confusion matrix, and a bootstrapped metrics table for multiclass classification tasks."""
    probs = np.stack(probs).squeeze()
    y_true = np.asarray(y_true)
    n_classes = probs.shape[1]
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
    save_dir = Path(save_dir) if save_dir else None

    if plot:
        plot_multiclass_bootstrap_roc(
            y_true_bin,
            probs,
            class_ids,
            save_path=save_dir / "auroc_bootstrap.png" if save_dir else None,
        )

    # Plot 1/3: ROC curve
    if plot:
        fig, ax = plt.subplots(figsize=(9, 9))

        colors = cycle(sns.color_palette())
        for i, color, class_id in zip(range(n_classes), colors, class_ids):
            RocCurveDisplay.from_predictions(
                y_true_bin[:, i],
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

        if save_dir:
            fig.savefig(save_dir / "roc_curves.png", bbox_inches="tight", dpi=600)
        plt.show()
        plt.close()

    # Plot 2/3: Confusion matrix
    y_true = np.argmax(y_true_bin, axis=1)
    y_pred = np.argmax(probs, axis=1)
    if plot:
        plot_confusion_matrix(
            y_true,
            y_pred,
            class_ids,
            plot=plot,
            save_path=save_dir / "confusion_matrix.png" if save_dir else None,
        )

    # Plot 3/3: Calibration curve
    plot_calibration_curve(
        y_true_bin.ravel(),
        probs.ravel(),
        plot=plot,
        model_label=prediction_task,
        save_path=save_dir / "calibration.png" if save_dir else None,
    )

    # Plot 4/4: Metrics table
    metrics = _bootstrap_multiclass_metric_table(y_true, probs, class_ids)

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


def plot_coef_boxplot3(
    data_coef, 
    data_var, 
    adaptive_legend=False, 
    print_nums=True, 
    figsize=(15, 6), 
    alpha=0.05,
    protective_label = "Protective (negative)",
    risk_label = "Increased risk (positive)",
    save_path=None,
):
    data_orig = data_var.values * 100
    data_cum = data_var.cumsum().values * 100
    means = data_coef.mean()

    y = np.arange(len(data_var))

    # Color by sign: blue for negative (protective), red for positive (increased risk)
    colors = ["#A9363B" if m >= 0 else "#2569BD" for m in means.values]

    # Compute confidence intervals
    n = len(data_coef)
    se = data_coef.sem()
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    ci_low = means - t_crit * se
    ci_high = means + t_crit * se

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize, sharey=True)

    # --- axes[0]: Coefficient plot using absolute values, all on right side ---
    feature_names = translate_feat_names(means.index)

    abs_means = means.abs()
    abs_ci_low = abs_means - t_crit * se
    abs_ci_high = abs_means + t_crit * se

    for i, (m, lo, hi, color) in enumerate(zip(abs_means, abs_ci_low, abs_ci_high, colors)):
        axes[0].errorbar(
            x=m, y=i,
            xerr=[[m - lo], [hi - m]],
            fmt="o",
            color=color,
            markersize=12,
            capsize=6,
            capthick=3,
            linewidth=3,
            markeredgecolor="white",
            markeredgewidth=1.5,
            zorder=3,
        )

    axes[0].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(feature_names, rotation=0, ha="right")
    ci_pct = (1 - alpha) * 100
    ci_str = f"{ci_pct:.0f}" if ci_pct == int(ci_pct) else f"{ci_pct:.2f}".rstrip("0")
    p_str = f"{alpha:.4f}".rstrip("0").rstrip(".")
    axes[0].set_xlabel(f"Absolute β Estimate ({ci_str}% CI about the mean, p < {p_str})")
    axes[0].invert_yaxis()

    # Legend for color meaning
    legend_handles = [
        mpatches.Patch(color="#2569BD", label=protective_label),
        mpatches.Patch(color="#A9363B", label=risk_label),
    ]
    axes[0].legend(handles=legend_handles)

    # --- axes[1]: Variance explained bars ---
    axes[1].barh(y, data_cum, color="white", edgecolor="black", label="Cumulative")
    axes[1].barh(y, data_orig, color=colors, edgecolor="black", label="Individual")
    axes[1].set_xlabel("Feature Importance (% var in β explained)")

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
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=600)
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
