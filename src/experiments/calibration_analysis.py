import numpy as np
import pandas as pd
from typing import Dict, List, Any
from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve
from ..utils.logger import setup_logger

logger = setup_logger("CalibrationAnalysis")

def evaluate_probability_calibration(
    y_true_clf: pd.Series,
    y_proba_clf: np.ndarray,
    classes: List[str] = ["Eligible", "High_Risk", "Not_Eligible"],
    n_bins: int = 10
) -> Dict[str, Any]:
    """Compute multiclass Brier score and per-class calibration curves."""
    logger.info("Evaluating probability calibration curves and Brier score...")
    y_bin = label_binarize(y_true_clf, classes=classes)

    # Multiclass Brier Score: Mean squared error between one-hot true labels and predicted probabilities
    brier_score = float(np.mean(np.sum((y_proba_clf - y_bin) ** 2, axis=1)))

    curves = {}
    for i, cls_name in enumerate(classes):
        prob_true, prob_pred = calibration_curve(y_bin[:, i], y_proba_clf[:, i], n_bins=n_bins, strategy="uniform")
        curves[cls_name] = {
            "fraction_of_positives": [round(float(x), 4) for x in prob_true],
            "mean_predicted_probability": [round(float(x), 4) for x in prob_pred]
        }

    logger.info(f"Calibration analysis complete. Multiclass Brier Score: {brier_score:.4f}")

    return {
        "multiclass_brier_score": round(brier_score, 4),
        "calibration_quality": "Excellent (< 0.10)" if brier_score < 0.10 else "Moderate",
        "curves_by_class": curves
    }
