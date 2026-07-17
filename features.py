import numpy as np
from sklearn.metrics import (
    classification_report, roc_auc_score,
    average_precision_score, precision_recall_curve,
)


def cost_threshold(y_true, proba, fn_fp_ratio=15):
    """
    Find the decision threshold that minimizes expected misclassification cost.
    fn_fp_ratio: how many times more costly is a missed fraud vs. a false alert.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    churn_rate = y_true.mean()

    fn_rates = 1 - recall[:-1]
    fp_rates = 1 - precision[:-1]
    costs = fn_fp_ratio * fn_rates * churn_rate + fp_rates * (1 - churn_rate)
    return float(thresholds[np.argmin(costs)])


def evaluate(y_true, proba, threshold=None, fn_fp_ratio=15, label="Model"):
    if threshold is None:
        threshold = cost_threshold(y_true, proba, fn_fp_ratio)

    y_pred = (proba >= threshold).astype(int)
    auc = roc_auc_score(y_true, proba)
    ap  = average_precision_score(y_true, proba)

    print(f"\n{'='*50}")
    print(f"{label} Evaluation")
    print(f"{'='*50}")
    print(f"AUC-ROC:            {auc:.4f}")
    print(f"Average Precision:  {ap:.4f}")
    print(f"Decision threshold: {threshold:.3f}  (FN:FP cost={fn_fp_ratio}:1)")
    print()
    print(classification_report(y_true, y_pred, target_names=["legitimate", "fraud"]))
    return {"auc": auc, "ap": ap, "threshold": threshold}
