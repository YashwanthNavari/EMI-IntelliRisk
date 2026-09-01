import os
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from src.utils.logger import setup_logger
from src.utils.config import get_project_root, load_config
from src.data.ingestion import load_raw_dataset
from src.data.cleaning import clean_dataset
from src.features.financial_features import compute_financial_features
from src.features.risk_features import compute_risk_features
from src.data.splitting import create_stratified_splits
from src.tracking.model_registry import load_model_artifact, export_mlflow_summary
from src.experiments.ablation_study import run_classification_ablation, run_regression_ablation
from src.experiments.uncertainty_bootstrap import compute_bootstrap_confidence_intervals
from src.experiments.fairness_audit import run_demographic_fairness_audit
from src.experiments.calibration_analysis import evaluate_probability_calibration

logger = setup_logger("AcademicValidation")

def main():
    logger.info("=================================================================")
    logger.info("  EMIPredict AI - FINAL ACADEMIC VALIDATION & STRESS-TEST SUITE")
    logger.info("=================================================================")
    start_time = time.time()
    root = get_project_root()
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ingestion, Cleaning & Feature Engineering
    logger.info("Ingesting and processing 404,800 dataset records...")
    df_raw = load_raw_dataset()
    df_clean = clean_dataset(df_raw)
    df_feat = compute_financial_features(df_clean)
    df_feat = compute_risk_features(df_feat)

    # 2. Stratified 70/15/15 Holdout Split
    train_df, val_df, test_df = create_stratified_splits(df_feat, stratify_col="emi_eligibility", test_size=0.15, val_size=0.15, random_state=42)

    target_class = "emi_eligibility"
    target_reg = "max_monthly_emi"
    feature_cols = [c for c in df_feat.columns if c not in [target_class, target_reg]]

    X_train = train_df[feature_cols]
    y_train_class = train_df[target_class]
    y_train_reg = train_df[target_reg]

    X_test = test_df[feature_cols]
    y_test_class = test_df[target_class]
    y_test_reg = test_df[target_reg]

    # Sample train partition for speed in ablation
    train_sample_idx = X_train.groupby(y_train_class, group_keys=False).apply(
        lambda x: x.sample(int(40000 * len(x) / len(X_train)), random_state=42)
    ).index
    X_train_sub = X_train.loc[train_sample_idx]
    y_train_class_sub = y_train_class.loc[train_sample_idx]
    y_train_reg_sub = y_train_reg.loc[train_sample_idx]

    # 3. Load Champion Models and Evaluate on Final Untouched Holdout Test Set (60,720 rows)
    logger.info("\n--- EVALUATING CHAMPION MODELS ON UNTOUCHED TEST SET (N=60,720) ---")
    clf_model = load_model_artifact("classification")
    reg_model = load_model_artifact("regression")

    # Final holdout predictions
    test_pred_clf = clf_model.predict(X_test)
    test_proba_clf = clf_model.predict_proba(X_test)
    test_pred_reg = np.maximum(500.0, reg_model.predict(X_test))

    # Normalized Confusion Matrix & Detailed Per-Class Metrics
    classes = ["Eligible", "High_Risk", "Not_Eligible"]
    raw_cm = confusion_matrix(y_test_class, test_pred_clf, labels=classes).tolist()
    norm_cm = confusion_matrix(y_test_class, test_pred_clf, labels=classes, normalize="true").tolist()
    clf_report_dict = classification_report(y_test_class, test_pred_clf, labels=classes, output_dict=True, zero_division=0)

    # 4. Feature Ablation Study
    logger.info("\n--- RUNNING FEATURE ABLATION & LEAKAGE STRESS TESTS ---")
    clf_ablation = run_classification_ablation(X_train_sub, y_train_class_sub, X_test, y_test_class)
    reg_ablation = run_regression_ablation(X_train_sub, y_train_reg_sub, X_test, y_test_reg)

    with open(reports_dir / "ablation_study_results.json", "w", encoding="utf-8") as f:
        json.dump({"classification_ablation": clf_ablation, "regression_ablation": reg_ablation}, f, indent=2)

    # 5. Bootstrap Uncertainty & 95% Confidence Intervals
    logger.info("\n--- COMPUTING 95% BOOTSTRAP CONFIDENCE INTERVALS ---")
    bootstrap_results = compute_bootstrap_confidence_intervals(y_test_class, test_pred_clf, y_test_reg, test_pred_reg, n_iterations=500)

    with open(reports_dir / "uncertainty_bootstrap_results.json", "w", encoding="utf-8") as f:
        json.dump(bootstrap_results, f, indent=2)

    # 6. Demographic Fairness & Subgroup Performance Audit
    logger.info("\n--- CONDUCTING DEMOGRAPHIC FAIRNESS & SUBGROUP AUDIT ---")
    fairness_results = run_demographic_fairness_audit(test_df, y_test_class, test_pred_clf, y_test_reg, test_pred_reg)

    with open(reports_dir / "fairness_audit_results.json", "w", encoding="utf-8") as f:
        json.dump(fairness_results, f, indent=2)

    # 7. Probability Calibration & Brier Score
    logger.info("\n--- ASSESSING PROBABILITY CALIBRATION & BRIER SCORE ---")
    calibration_results = evaluate_probability_calibration(y_test_class, test_proba_clf, classes=classes)

    # 8. Update models/mlflow_summary.json with Complete Scientific Suite
    summary_file = root / "models" / "mlflow_summary.json"
    existing_summary = {}
    if summary_file.exists():
        with open(summary_file, "r") as f:
            existing_summary = json.load(f)

    existing_summary.update({
        "final_holdout_test_size": len(test_df),
        "normalized_confusion_matrix": norm_cm,
        "raw_confusion_matrix": raw_cm,
        "per_class_classification_report": clf_report_dict,
        "classification_ablation": clf_ablation,
        "regression_ablation": reg_ablation,
        "bootstrap_confidence_intervals": bootstrap_results,
        "fairness_audit": fairness_results,
        "calibration_analysis": calibration_results,
        "academic_validation_completed_at": time.strftime("%Y-%m-%d %H:%M:%S")
    })

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(existing_summary, f, indent=2)

    # Save comprehensive master report
    with open(reports_dir / "academic_validation_master_report.json", "w", encoding="utf-8") as f:
        json.dump(existing_summary, f, indent=2)

    elapsed = time.time() - start_time
    logger.info(f"\n=================================================================")
    logger.info(f"  ACADEMIC VALIDATION & STRESS TESTS COMPLETED IN {elapsed/60:.2f} MINUTES")
    logger.info(f"  Brier Score:        {calibration_results['multiclass_brier_score']}")
    logger.info(f"  Bootstrap Acc 95% CI: {bootstrap_results['classification_ci']['accuracy']['formatted']}")
    logger.info(f"  Bootstrap MAE 95% CI: {bootstrap_results['regression_ci']['mae']['formatted']}")
    logger.info("=================================================================")

if __name__ == "__main__":
    main()
