import pandas as pd
import numpy as np
from typing import Dict, List, Any
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from lightgbm import LGBMClassifier, LGBMRegressor
from ..models.evaluation import evaluate_classification_model, evaluate_regression_model
from ..utils.logger import setup_logger

logger = setup_logger("AblationStudy")

def _build_simple_pipeline(num_cols: List[str], cat_cols: List[str], is_classifier: bool = True):
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler())
    ])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ], remainder="drop")

    model = LGBMClassifier(n_estimators=120, learning_rate=0.08, max_depth=6, class_weight="balanced", random_state=42, verbose=-1) if is_classifier else \
            LGBMRegressor(n_estimators=120, learning_rate=0.08, max_depth=6, random_state=42, verbose=-1)

    return Pipeline([("preprocessor", preprocessor), ("estimator", model)])

def run_classification_ablation(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> List[Dict[str, Any]]:
    """Execute classification feature ablation experiments."""
    logger.info("Starting Classification Feature Ablation experiments...")
    results = []

    # Model A: Demographics Only
    demo_num = ["age", "family_size", "dependents"]
    demo_cat = ["gender", "marital_status", "education", "house_type"]
    pipe_a = _build_simple_pipeline(demo_num, demo_cat, is_classifier=True)
    pipe_a.fit(X_train, y_train)
    pred_a = pipe_a.predict(X_test)
    m_a = evaluate_classification_model("Model A (Demographics Only)", y_test, pred_a)
    results.append({
        "experiment": "Model A (Demographics Only)",
        "features": f"{len(demo_num)+len(demo_cat)} features (Age, Gender, Marital, Education, Family, House)",
        "accuracy": m_a.accuracy,
        "balanced_accuracy": m_a.balanced_accuracy,
        "macro_f1": m_a.macro_f1,
        "high_risk_recall": m_a.high_risk_recall
    })

    # Model B: Demographics + Raw Financials
    raw_num = demo_num + [
        "monthly_salary", "years_of_employment", "monthly_rent", "school_fees", "college_fees",
        "travel_expenses", "groceries_utilities", "other_monthly_expenses", "current_emi_amount",
        "credit_score", "bank_balance", "emergency_fund", "requested_amount", "requested_tenure"
    ]
    raw_cat = demo_cat + ["employment_type", "company_type", "existing_loans", "emi_scenario"]
    pipe_b = _build_simple_pipeline(raw_num, raw_cat, is_classifier=True)
    pipe_b.fit(X_train, y_train)
    pred_b = pipe_b.predict(X_test)
    m_b = evaluate_classification_model("Model B (Demographics + Raw Financials)", y_test, pred_b)
    results.append({
        "experiment": "Model B (Demographics + Raw Financials)",
        "features": f"{len(raw_num)+len(raw_cat)} features (All raw un-engineered inputs)",
        "accuracy": m_b.accuracy,
        "balanced_accuracy": m_b.balanced_accuracy,
        "macro_f1": m_b.macro_f1,
        "high_risk_recall": m_b.high_risk_recall
    })

    # Model C: Full Feature Set (with Engineered Features)
    all_num = raw_num + [
        "total_expenses", "disposable_income", "expense_to_income_ratio", "foir", "emi_burden_ratio",
        "savings_to_income_ratio", "emergency_fund_buffer_months", "loan_to_income_ratio",
        "requested_monthly_installment_estimate", "per_capita_disposable_income", "financial_health_score"
    ]
    all_cat = raw_cat + ["credit_risk_tier", "employment_stability_score"]
    pipe_c = _build_simple_pipeline(all_num, all_cat, is_classifier=True)
    pipe_c.fit(X_train, y_train)
    pred_c = pipe_c.predict(X_test)
    m_c = evaluate_classification_model("Model C (Full Feature Matrix)", y_test, pred_c)
    results.append({
        "experiment": "Model C (Full Feature Matrix)",
        "features": f"{len(all_num)+len(all_cat)} features (Raw + 12 Engineered FinTech Ratios)",
        "accuracy": m_c.accuracy,
        "balanced_accuracy": m_c.balanced_accuracy,
        "macro_f1": m_c.macro_f1,
        "high_risk_recall": m_c.high_risk_recall
    })

    logger.info("Classification feature ablation completed.")
    return results

def run_regression_ablation(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> List[Dict[str, Any]]:
    """Execute regression ablation & leakage stress-testing experiments."""
    logger.info("Starting Regression Feature Ablation & Leakage Stress-Test...")
    results = []

    # Exp A: Full Features
    full_num = [
        "age", "monthly_salary", "years_of_employment", "monthly_rent", "family_size", "dependents",
        "school_fees", "college_fees", "travel_expenses", "groceries_utilities", "other_monthly_expenses",
        "current_emi_amount", "credit_score", "bank_balance", "emergency_fund", "requested_amount", "requested_tenure",
        "total_expenses", "disposable_income", "expense_to_income_ratio", "foir", "emi_burden_ratio",
        "savings_to_income_ratio", "emergency_fund_buffer_months", "loan_to_income_ratio",
        "requested_monthly_installment_estimate", "per_capita_disposable_income", "financial_health_score"
    ]
    full_cat = ["gender", "marital_status", "education", "employment_type", "company_type", "house_type", "existing_loans", "emi_scenario", "credit_risk_tier", "employment_stability_score"]
    pipe_a = _build_simple_pipeline(full_num, full_cat, is_classifier=False)
    pipe_a.fit(X_train, y_train)
    pred_a = np.maximum(500.0, pipe_a.predict(X_test))
    m_a = evaluate_regression_model("Exp A (Full Feature Set)", y_test, pred_a)
    results.append({
        "experiment": "Exp A (Full Feature Matrix)",
        "feature_group": "Raw + All Engineered FinTech Ratios (38 features)",
        "r2": m_a.r2,
        "rmse": m_a.rmse,
        "mae": m_a.mae,
        "mape": m_a.mape
    })

    # Exp B: Raw Features Only
    raw_num = [
        "age", "monthly_salary", "years_of_employment", "monthly_rent", "family_size", "dependents",
        "school_fees", "college_fees", "travel_expenses", "groceries_utilities", "other_monthly_expenses",
        "current_emi_amount", "credit_score", "bank_balance", "emergency_fund", "requested_amount", "requested_tenure"
    ]
    raw_cat = ["gender", "marital_status", "education", "employment_type", "company_type", "house_type", "existing_loans", "emi_scenario"]
    pipe_b = _build_simple_pipeline(raw_num, raw_cat, is_classifier=False)
    pipe_b.fit(X_train, y_train)
    pred_b = np.maximum(500.0, pipe_b.predict(X_test))
    m_b = evaluate_regression_model("Exp B (Raw Features Only)", y_test, pred_b)
    results.append({
        "experiment": "Exp B (Raw Features Only)",
        "feature_group": "Raw Disclosures without engineered math (25 features)",
        "r2": m_b.r2,
        "rmse": m_b.rmse,
        "mae": m_b.mae,
        "mape": m_b.mape
    })

    # Exp C: Engineered Features Only
    eng_num = [
        "total_expenses", "disposable_income", "expense_to_income_ratio", "foir", "emi_burden_ratio",
        "savings_to_income_ratio", "emergency_fund_buffer_months", "loan_to_income_ratio",
        "requested_monthly_installment_estimate", "per_capita_disposable_income", "financial_health_score"
    ]
    eng_cat = ["credit_risk_tier", "employment_stability_score"]
    pipe_c = _build_simple_pipeline(eng_num, eng_cat, is_classifier=False)
    pipe_c.fit(X_train, y_train)
    pred_c = np.maximum(500.0, pipe_c.predict(X_test))
    m_c = evaluate_regression_model("Exp C (Engineered Features Only)", y_test, pred_c)
    results.append({
        "experiment": "Exp C (Engineered Ratios Only)",
        "feature_group": "Only Derived FinTech Ratios (13 features)",
        "r2": m_c.r2,
        "rmse": m_c.rmse,
        "mae": m_c.mae,
        "mape": m_c.mape
    })

    # Exp D: Restricted Proximal Features (Exclude disposable_income and foir)
    rest_num = [c for c in full_num if c not in ["disposable_income", "foir"]]
    pipe_d = _build_simple_pipeline(rest_num, full_cat, is_classifier=False)
    pipe_d.fit(X_train, y_train)
    pred_d = np.maximum(500.0, pipe_d.predict(X_test))
    m_d = evaluate_regression_model("Exp D (Restricted Proximal Features)", y_test, pred_d)
    results.append({
        "experiment": "Exp D (Restricted Target-Proximal Features)",
        "feature_group": "Excludes Disposable Income & FOIR (36 features)",
        "r2": m_d.r2,
        "rmse": m_d.rmse,
        "mae": m_d.mae,
        "mape": m_d.mape
    })

    logger.info("Regression ablation and leakage stress test completed.")
    return results
