import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score, mean_squared_error, mean_absolute_error, r2_score
from ..utils.logger import setup_logger

logger = setup_logger("UncertaintyBootstrap")

def compute_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error (SMAPE) bound between 0% and 200%."""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_pred - y_true)
    # Avoid division by zero
    mask = denominator != 0
    return float(np.mean(diff[mask] / denominator[mask]) * 100.0) if np.any(mask) else 0.0

def compute_bootstrap_confidence_intervals(
    y_true_clf: pd.Series,
    y_pred_clf: np.ndarray,
    y_true_reg: pd.Series,
    y_pred_reg: np.ndarray,
    n_iterations: int = 500,
    random_state: int = 42
) -> Dict[str, Any]:
    """Calculate empirical 95% bootstrap confidence intervals across test set predictions."""
    logger.info(f"Computing 95% bootstrap confidence intervals over {n_iterations} iterations...")
    np.random.seed(random_state)
    n_samples = len(y_true_clf)

    y_t_c = np.array(y_true_clf)
    y_p_c = np.array(y_pred_clf)

    y_t_r = np.array(y_true_reg, dtype=float)
    y_p_r = np.array(y_pred_reg, dtype=float)

    clf_metrics = {"accuracy": [], "balanced_acc": [], "macro_f1": [], "high_risk_recall": []}
    reg_metrics = {"rmse": [], "mae": [], "r2": [], "smape": []}

    for _ in range(n_iterations):
        idx = np.random.randint(0, n_samples, size=n_samples)

        # Classification bootstrap
        sub_yt_c = y_t_c[idx]
        sub_yp_c = y_p_c[idx]
        clf_metrics["accuracy"].append(accuracy_score(sub_yt_c, sub_yp_c))
        clf_metrics["balanced_acc"].append(balanced_accuracy_score(sub_yt_c, sub_yp_c))
        clf_metrics["macro_f1"].append(f1_score(sub_yt_c, sub_yp_c, average="macro", zero_division=0))
        rec_arr = recall_score(sub_yt_c, sub_yp_c, average=None, labels=["Eligible", "High_Risk", "Not_Eligible"], zero_division=0)
        clf_metrics["high_risk_recall"].append(rec_arr[1])

        # Regression bootstrap
        sub_yt_r = y_t_r[idx]
        sub_yp_r = y_p_r[idx]
        reg_metrics["rmse"].append(np.sqrt(mean_squared_error(sub_yt_r, sub_yp_r)))
        reg_metrics["mae"].append(mean_absolute_error(sub_yt_r, sub_yp_r))
        reg_metrics["r2"].append(r2_score(sub_yt_r, sub_yp_r))
        reg_metrics["smape"].append(compute_smape(sub_yt_r, sub_yp_r))

    def _ci(arr):
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "ci_lower": float(np.percentile(arr, 2.5)),
            "ci_upper": float(np.percentile(arr, 97.5)),
            "formatted": f"{np.mean(arr):.4f} [95% CI: {np.percentile(arr, 2.5):.4f} – {np.percentile(arr, 97.5):.4f}]"
        }

    def _ci_currency(arr):
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "ci_lower": float(np.percentile(arr, 2.5)),
            "ci_upper": float(np.percentile(arr, 97.5)),
            "formatted": f"₹{np.mean(arr):.2f} [95% CI: ₹{np.percentile(arr, 2.5):.2f} – ₹{np.percentile(arr, 97.5):.2f}]"
        }

    results = {
        "classification_ci": {
            "accuracy": _ci(clf_metrics["accuracy"]),
            "balanced_accuracy": _ci(clf_metrics["balanced_acc"]),
            "macro_f1": _ci(clf_metrics["macro_f1"]),
            "high_risk_recall": _ci(clf_metrics["high_risk_recall"])
        },
        "regression_ci": {
            "rmse": _ci_currency(reg_metrics["rmse"]),
            "mae": _ci_currency(reg_metrics["mae"]),
            "r2": _ci(reg_metrics["r2"]),
            "smape": _ci(reg_metrics["smape"])
        }
    }

    logger.info("Bootstrap uncertainty quantification complete.")
    return results
