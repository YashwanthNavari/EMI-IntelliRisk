from typing import Dict, Tuple, Any
from sklearn.pipeline import Pipeline
from .evaluation import ClassificationMetrics, RegressionMetrics
from ..utils.logger import setup_logger

logger = setup_logger("ModelSelection")

def select_best_classification_model(
    results: Dict[str, Tuple[Pipeline, ClassificationMetrics]]
) -> Tuple[str, Pipeline, ClassificationMetrics, str]:
    """Select best classification model using weighted multi-criteria scoring."""
    best_name = None
    best_score = -1.0
    best_pipeline = None
    best_metrics = None

    scores = {}
    for name, (pipeline, metrics) in results.items():
        # Composite decision score
        # Macro F1 (40%), High Risk Recall (30%), Balanced Accuracy (20%), ROC-AUC (10%)
        auc_score = metrics.roc_auc_ovr if metrics.roc_auc_ovr is not None else metrics.macro_f1
        composite_score = (
            0.40 * metrics.macro_f1 +
            0.30 * metrics.high_risk_recall +
            0.20 * metrics.balanced_accuracy +
            0.10 * auc_score
        )
        scores[name] = composite_score

        if composite_score > best_score:
            best_score = composite_score
            best_name = name
            best_pipeline = pipeline
            best_metrics = metrics

    rationale = (
        f"Selected '{best_name}' with highest multi-criteria composite score ({best_score:.4f}). "
        f"Key metrics: Macro-F1={best_metrics.macro_f1:.4f}, High-Risk Recall={best_metrics.high_risk_recall:.4f}, "
        f"Balanced Accuracy={best_metrics.balanced_accuracy:.4f}, Overall Accuracy={best_metrics.accuracy:.4f}."
    )
    logger.info(f"Best classification model: {best_name}. Rationale: {rationale}")
    return best_name, best_pipeline, best_metrics, rationale

def select_best_regression_model(
    results: Dict[str, Tuple[Pipeline, RegressionMetrics]]
) -> Tuple[str, Pipeline, RegressionMetrics, str]:
    """Select best regression model based on minimum RMSE and highest R²."""
    best_name = None
    best_score = -float("inf")
    best_pipeline = None
    best_metrics = None

    for name, (pipeline, metrics) in results.items():
        # Score normalized higher is better: R² - (normalized RMSE / 10000)
        score = metrics.r2 - (metrics.rmse / 10000.0)

        if score > best_score:
            best_score = score
            best_name = name
            best_pipeline = pipeline
            best_metrics = metrics

    rationale = (
        f"Selected '{best_name}' with superior predictive accuracy: "
        f"RMSE={best_metrics.rmse:.2f}, MAE={best_metrics.mae:.2f}, R²={best_metrics.r2:.4f}, MAPE={best_metrics.mape:.2f}%."
    )
    logger.info(f"Best regression model: {best_name}. Rationale: {rationale}")
    return best_name, best_pipeline, best_metrics, rationale
