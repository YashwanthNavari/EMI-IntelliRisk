import os
import sys
import json
import time
import pandas as pd
import numpy as np
from pathlib import Path

# Ensure project root is in sys.path
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from src.utils.logger import setup_logger
from src.utils.config import load_config, get_project_root
from src.data.ingestion import load_raw_dataset
from src.data.schema_validation import validate_schema
from src.data.data_quality import perform_data_quality_audit
from src.data.cleaning import clean_dataset
from src.data.splitting import create_stratified_splits
from src.features.financial_features import compute_financial_features
from src.features.risk_features import compute_risk_features
from src.models.classification.train import train_classification_models
from src.models.classification.tune import tune_classification_hyperparameters
from src.models.regression.train import train_regression_models
from src.models.regression.tune import tune_regression_hyperparameters
from src.models.selection import select_best_classification_model, select_best_regression_model
from src.tracking.mlflow_tracking import MLflowExperimentTracker
from src.tracking.model_registry import save_model_artifact, export_mlflow_summary
from database.database import init_db
from database.repository import CustomerRepository, PredictionRepository

logger = setup_logger("MasterPipeline")

def run_pipeline():
    logger.info("=================================================================")
    logger.info("  EMIPredict AI - END-TO-END TRAINING & MLOPS ORCHESTRATION")
    logger.info("=================================================================")
    start_time = time.time()
    root = get_project_root()
    config = load_config()

    # Step 1: Ingestion
    logger.info("\n--- PHASE 1: DATA INGESTION ---")
    df_raw = load_raw_dataset()
    logger.info(f"Loaded raw dataset with shape: {df_raw.shape}")

    # Step 2: Schema Validation & Data Quality Audit
    logger.info("\n--- PHASE 2: SCHEMA VALIDATION & DATA QUALITY AUDIT ---")
    schema_res = validate_schema(df_raw)
    dq_report = perform_data_quality_audit(df_raw)

    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "data_quality_report.json", "w", encoding="utf-8") as f:
        json.dump(dq_report.to_dict(), f, indent=2)

    # Step 3: Data Cleaning
    logger.info("\n--- PHASE 3: DATA CLEANING ---")
    df_clean = clean_dataset(df_raw, is_training=True)

    # Save cleaned interim sample
    interim_dir = root / "data" / "interim"
    interim_dir.mkdir(parents=True, exist_ok=True)
    df_clean.head(1000).to_csv(interim_dir / "cleaned_sample.csv", index=False)

    # Step 4: Financial Feature Engineering
    logger.info("\n--- PHASE 4: FINANCIAL FEATURE ENGINEERING ---")
    df_featured = compute_financial_features(df_clean)
    df_featured = compute_risk_features(df_featured)

    # Step 5: Target Leakage Audit
    logger.info("\n--- PHASE 5: TARGET LEAKAGE AUDIT ---")
    target_class = config["targets"]["classification"]["name"]
    target_reg = config["targets"]["regression"]["name"]

    feature_cols = [c for c in df_featured.columns if c not in [target_class, target_reg]]
    logger.info(f"Audit Verified: {len(feature_cols)} predictor features identified. Zero target columns present in feature matrix.")

    # Step 6: Dataset Splitting
    logger.info("\n--- PHASE 6: STRATIFIED DATA SPLITTING ---")
    train_df, val_df, test_df = create_stratified_splits(
        df_featured,
        stratify_col=target_class,
        test_size=0.15,
        val_size=0.15,
        random_state=config.get("project", {}).get("random_seed", 42)
    )

    # Save processed splits
    proc_dir = root / "data" / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)
    train_df.head(1000).to_csv(proc_dir / "train_sample.csv", index=False)
    test_df.head(1000).to_csv(proc_dir / "test_sample.csv", index=False)

    X_train = train_df[feature_cols]
    y_train_class = train_df[target_class]
    y_train_reg = train_df[target_reg]

    X_val = val_df[feature_cols]
    y_val_class = val_df[target_class]
    y_val_reg = val_df[target_reg]

    X_test = test_df[feature_cols]
    y_test_class = test_df[target_class]
    y_test_reg = test_df[target_reg]

    # Step 7: Classification Model Development & MLflow
    logger.info("\n--- PHASE 7: CLASSIFICATION MODEL TRAINING & MLFLOW TRACKING ---")
    clf_tracker = MLflowExperimentTracker("EMI_Eligibility_Classification")

    # Use a solid subset for high-speed multi-model benchmark if data > 60k
    train_sample_size = min(len(X_train), 80000)
    train_sample_idx = X_train.groupby(y_train_class, group_keys=False).apply(
        lambda x: x.sample(int(train_sample_size * len(x) / len(X_train)), random_state=42)
    ).index

    X_train_sub = X_train.loc[train_sample_idx]
    y_train_class_sub = y_train_class.loc[train_sample_idx]
    y_train_reg_sub = y_train_reg.loc[train_sample_idx]

    clf_results = train_classification_models(X_train_sub, y_train_class_sub, X_val, y_val_class)

    mlflow_clf_runs = []
    for model_name, (pipeline, metrics) in clf_results.items():
        run_id = clf_tracker.log_run(
            run_name=f"Classification_{model_name}",
            params={"model_type": model_name, "train_size": len(X_train_sub)},
            metrics={
                "accuracy": metrics.accuracy,
                "balanced_accuracy": metrics.balanced_accuracy,
                "macro_f1": metrics.macro_f1,
                "weighted_f1": metrics.weighted_f1,
                "high_risk_recall": metrics.high_risk_recall,
                "eligible_recall": metrics.eligible_recall,
                "not_eligible_recall": metrics.not_eligible_recall,
                "roc_auc_ovr": metrics.roc_auc_ovr or 0.0
            }
        )
        mlflow_clf_runs.append({
            "run_id": run_id,
            "model_name": model_name,
            "metrics": metrics.to_dict()
        })

    best_clf_name, best_clf_pipe, best_clf_metrics, clf_rationale = select_best_classification_model(clf_results)

    # Step 8: Regression Model Development & MLflow
    logger.info("\n--- PHASE 8: REGRESSION MODEL TRAINING & MLFLOW TRACKING ---")
    reg_tracker = MLflowExperimentTracker("Maximum_EMI_Regression")

    reg_results = train_regression_models(X_train_sub, y_train_reg_sub, X_val, y_val_reg)

    mlflow_reg_runs = []
    for model_name, (pipeline, metrics) in reg_results.items():
        run_id = reg_tracker.log_run(
            run_name=f"Regression_{model_name}",
            params={"model_type": model_name, "train_size": len(X_train_sub)},
            metrics={
                "rmse": metrics.rmse,
                "mae": metrics.mae,
                "r2": metrics.r2,
                "mape": metrics.mape,
                "median_ae": metrics.median_ae
            }
        )
        mlflow_reg_runs.append({
            "run_id": run_id,
            "model_name": model_name,
            "metrics": metrics.to_dict()
        })

    best_reg_name, best_reg_pipe, best_reg_metrics, reg_rationale = select_best_regression_model(reg_results)

    # Step 9: Save Best Models & Metadata
    logger.info("\n--- PHASE 9: MODEL PERSISTENCE & REGISTRY ---")
    save_model_artifact(
        model_pipeline=best_clf_pipe,
        task_type="classification",
        model_name=best_clf_name,
        metadata={
            "model_name": best_clf_name,
            "selection_rationale": clf_rationale,
            "metrics": best_clf_metrics.to_dict(),
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    )

    save_model_artifact(
        model_pipeline=best_reg_pipe,
        task_type="regression",
        model_name=best_reg_name,
        metadata={
            "model_name": best_reg_name,
            "selection_rationale": reg_rationale,
            "metrics": best_reg_metrics.to_dict(),
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    )

    # Step 10: Export MLflow Summary for Streamlit UI
    export_mlflow_summary({
        "classification_runs": mlflow_clf_runs,
        "regression_runs": mlflow_reg_runs,
        "best_classification_model": best_clf_name,
        "best_regression_model": best_reg_name,
        "classification_rationale": clf_rationale,
        "regression_rationale": reg_rationale,
        "exported_at": time.strftime("%Y-%m-%d %H:%M:%S")
    })

    # Step 11: Initialize Application Database
    logger.info("\n--- PHASE 10: DATABASE INITIALIZATION & SAMPLE PROFILES ---")
    init_db()

    # Seed 3 realistic demo customer profiles into database
    sample_customers = [
        {
            "full_name": "Aarav Sharma",
            "email": "aarav.sharma@example.com",
            "phone": "+91 98765 43210",
            "age": 32.0,
            "gender": "Male",
            "marital_status": "Married",
            "education": "Post Graduate",
            "monthly_salary": 95000.0,
            "employment_type": "Private",
            "years_of_employment": 6.5,
            "company_type": "MNC",
            "house_type": "Rented",
            "monthly_rent": 18000.0,
            "family_size": 3,
            "dependents": 1,
            "school_fees": 4000.0,
            "college_fees": 0.0,
            "travel_expenses": 6000.0,
            "groceries_utilities": 16000.0,
            "other_monthly_expenses": 8000.0,
            "existing_loans": "No",
            "current_emi_amount": 0.0,
            "credit_score": 780.0,
            "bank_balance": 450000.0,
            "emergency_fund": 180000.0
        },
        {
            "full_name": "Priya Patel",
            "email": "priya.patel@example.com",
            "phone": "+91 98123 45678",
            "age": 28.0,
            "gender": "Female",
            "marital_status": "Single",
            "education": "Graduate",
            "monthly_salary": 42000.0,
            "employment_type": "Private",
            "years_of_employment": 2.0,
            "company_type": "Mid-size",
            "house_type": "Rented",
            "monthly_rent": 12000.0,
            "family_size": 1,
            "dependents": 0,
            "school_fees": 0.0,
            "college_fees": 0.0,
            "travel_expenses": 3500.0,
            "groceries_utilities": 8500.0,
            "other_monthly_expenses": 6000.0,
            "existing_loans": "Yes",
            "current_emi_amount": 7500.0,
            "credit_score": 670.0,
            "bank_balance": 85000.0,
            "emergency_fund": 25000.0
        },
        {
            "full_name": "Rajesh Kumar",
            "email": "rajesh.kumar@example.com",
            "phone": "+91 97555 12345",
            "age": 45.0,
            "gender": "Male",
            "marital_status": "Married",
            "education": "High School",
            "monthly_salary": 28000.0,
            "employment_type": "Self-employed",
            "years_of_employment": 12.0,
            "company_type": "Small",
            "house_type": "Own",
            "monthly_rent": 0.0,
            "family_size": 5,
            "dependents": 3,
            "school_fees": 6000.0,
            "college_fees": 4000.0,
            "travel_expenses": 3000.0,
            "groceries_utilities": 12000.0,
            "other_monthly_expenses": 4500.0,
            "existing_loans": "Yes",
            "current_emi_amount": 5000.0,
            "credit_score": 610.0,
            "bank_balance": 35000.0,
            "emergency_fund": 10000.0
        }
    ]

    for cust in sample_customers:
        cust_id = CustomerRepository.create_customer(cust)
        # Add sample prediction for each demo customer
        PredictionRepository.save_prediction({
            "customer_id": cust_id,
            "emi_scenario": "Personal Loan EMI",
            "requested_amount": 300000.0,
            "requested_tenure": 24,
            "predicted_eligibility": "Eligible" if cust["credit_score"] > 700 else "High_Risk",
            "confidence_score": 0.88 if cust["credit_score"] > 700 else 0.72,
            "prob_eligible": 0.88 if cust["credit_score"] > 700 else 0.20,
            "prob_high_risk": 0.08 if cust["credit_score"] > 700 else 0.72,
            "prob_not_eligible": 0.04 if cust["credit_score"] > 700 else 0.08,
            "predicted_max_emi": 22500.0 if cust["credit_score"] > 700 else 6500.0,
            "disposable_income": cust["monthly_salary"] - 30000,
            "foir": 0.25 if cust["credit_score"] > 700 else 0.45,
            "model_version": "1.0.0",
            "notes": "Initial system demonstration baseline record."
        })

    elapsed = time.time() - start_time
    logger.info(f"\n=================================================================")
    logger.info(f"  TRAINING PIPELINE COMPLETED IN {elapsed/60:.2f} MINUTES")
    logger.info(f"  Best Classification: {best_clf_name} (Macro-F1: {best_clf_metrics.macro_f1})")
    logger.info(f"  Best Regression:     {best_reg_name} (RMSE: {best_reg_metrics.rmse}, R²: {best_reg_metrics.r2})")
    logger.info("=================================================================")

if __name__ == "__main__":
    run_pipeline()
