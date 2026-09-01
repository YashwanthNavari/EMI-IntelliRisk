"""
Scientific experiment suites for Feature Ablation, Leakage Stress Testing, Bootstrap Uncertainty, Fairness Audit, and Model Calibration.
"""

from .ablation_study import run_classification_ablation, run_regression_ablation
from .uncertainty_bootstrap import compute_bootstrap_confidence_intervals
from .fairness_audit import run_demographic_fairness_audit
from .calibration_analysis import evaluate_probability_calibration

__all__ = [
    "run_classification_ablation",
    "run_regression_ablation",
    "compute_bootstrap_confidence_intervals",
    "run_demographic_fairness_audit",
    "evaluate_probability_calibration"
]
