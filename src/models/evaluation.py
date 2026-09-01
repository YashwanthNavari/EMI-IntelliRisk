import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score, median_absolute_error,
    explained_variance_score
)
from ..utils.logger import setup_logger

logger = setup_logger("ModelEvaluation")

@dataclass
class ClassificationMetrics:
    model_name: str
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    weighted_f1: float
    macro_precision: float
    macro_recall: float
    high_risk_recall: float
    eligible_recall: float
    not_eligible_recall: float
    roc_auc_ovr: Optional[float]
    confusion_matrix: List[List[int]]
    classes: List[str]
    classification_report: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class RegressionMetrics:
    model_name: str
    mae: float
    rmse: float
    r2: float
    mape: float
    median_ae: float
    max_error: float
    explained_variance: float
    residual_mean: float
    residual_std: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

def evaluate_classification_model(
    model_name: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    classes: Optional[List[str]] = None
) -> ClassificationMetrics:
    """Evaluate multiclass classification performance thoroughly."""
    if classes is None:
        classes = sorted(list(np.unique(y_true)))

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    # Per-class recall
    rec_per_class = recall_score(y_true, y_pred, average=None, labels=classes, zero_division=0)
    rec_dict = {cls: float(r) for cls, r in zip(classes, rec_per_class)}

    high_risk_rec = rec_dict.get("High_Risk", 0.0)
    eligible_rec = rec_dict.get("Eligible", 0.0)
    not_eligible_rec = rec_dict.get("Not_Eligible", 0.0)

    # Multiclass ROC AUC (OVR)
    roc_auc = None
    if y_proba is not None:
        try:
            roc_auc = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
        except Exception as e:
            logger.warning(f"Could not compute ROC-AUC for {model_name}: {e}")

    cm = confusion_matrix(y_true, y_pred, labels=classes).tolist()
    report = classification_report(y_true, y_pred, labels=classes, output_dict=True, zero_division=0)

    logger.info(f"[{model_name}] Accuracy={acc:.4f}, Macro-F1={macro_f1:.4f}, High-Risk Recall={high_risk_rec:.4f}, ROC-AUC={roc_auc if roc_auc else 0:.4f}")

    return ClassificationMetrics(
        model_name=model_name,
        accuracy=round(acc, 4),
        balanced_accuracy=round(bal_acc, 4),
        macro_f1=round(macro_f1, 4),
        weighted_f1=round(weighted_f1, 4),
        macro_precision=round(macro_prec, 4),
        macro_recall=round(macro_rec, 4),
        high_risk_recall=round(high_risk_rec, 4),
        eligible_recall=round(eligible_rec, 4),
        not_eligible_recall=round(not_eligible_rec, 4),
        roc_auc_ovr=round(roc_auc, 4) if roc_auc is not None else None,
        confusion_matrix=cm,
        classes=classes,
        classification_report=report
    )

def evaluate_regression_model(
    model_name: str,
    y_true: pd.Series,
    y_pred: np.ndarray
) -> RegressionMetrics:
    """Evaluate continuous regression performance thoroughly."""
    y_t = np.array(y_true, dtype=float)
    y_p = np.array(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_t, y_p))
    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
    r2 = float(r2_score(y_t, y_p))

    # MAPE with epsilon to prevent div-by-zero
    eps = 1e-5
    mape = float(np.mean(np.abs((y_t - y_p) / (np.abs(y_t) + eps))) * 100.0)
    med_ae = float(median_absolute_error(y_t, y_p))
    max_err = float(np.max(np.abs(y_t - y_p)))
    exp_var = float(explained_variance_score(y_t, y_p))

    residuals = y_t - y_p
    res_mean = float(np.mean(residuals))
    res_std = float(np.std(residuals))

    logger.info(f"[{model_name}] RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.4f}, MAPE={mape:.2f}%")

    return RegressionMetrics(
        model_name=model_name,
        mae=round(mae, 2),
        rmse=round(rmse, 2),
        r2=round(r2, 4),
        mape=round(mape, 2),
        median_ae=round(med_ae, 2),
        max_error=round(max_err, 2),
        explained_variance=round(exp_var, 4),
        residual_mean=round(res_mean, 2),
        residual_std=round(res_std, 2)
    )
