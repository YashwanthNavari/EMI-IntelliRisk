import pytest
import numpy as np
import pandas as pd
from src.features.feature_pipeline import create_preprocessor_pipeline
from src.models.classification.train import build_classification_pipeline
from src.models.regression.train import build_regression_pipeline
from src.models.evaluation import evaluate_classification_model, evaluate_regression_model

def test_feature_pipeline_fit_transform():
    sample_df = pd.DataFrame([{
        "age": 30.0, "gender": "Male", "marital_status": "Married", "education": "Graduate",
        "monthly_salary": 60000.0, "employment_type": "Private", "years_of_employment": 3.0,
        "company_type": "MNC", "house_type": "Rented", "monthly_rent": 15000.0,
        "family_size": 3, "dependents": 1, "school_fees": 2000.0, "college_fees": 0.0,
        "travel_expenses": 4000.0, "groceries_utilities": 12000.0, "other_monthly_expenses": 4000.0,
        "existing_loans": "No", "current_emi_amount": 0.0, "credit_score": 750.0,
        "bank_balance": 200000.0, "emergency_fund": 50000.0, "emi_scenario": "Personal Loan EMI",
        "requested_amount": 200000.0, "requested_tenure": 24
    }])

    pipe = create_preprocessor_pipeline()
    transformed = pipe.fit_transform(sample_df)
    assert transformed is not None
    assert transformed.shape[0] == 1
    assert transformed.shape[1] > 20

def test_classification_pipeline_build():
    pipe = build_classification_pipeline(model_type="logistic_regression")
    assert pipe is not None
    assert "feature_pipeline" in pipe.named_steps
    assert "classifier" in pipe.named_steps

def test_regression_pipeline_build():
    pipe = build_regression_pipeline(model_type="ridge_regression")
    assert pipe is not None
    assert "feature_pipeline" in pipe.named_steps
    assert "regressor" in pipe.named_steps

def test_evaluation_metrics_calculation():
    y_true_clf = pd.Series(["Eligible", "Not_Eligible", "High_Risk", "Eligible"])
    y_pred_clf = np.array(["Eligible", "Not_Eligible", "Eligible", "Eligible"])
    metrics_clf = evaluate_classification_model("TestModel", y_true_clf, y_pred_clf)
    assert metrics_clf.accuracy == 0.75
    assert metrics_clf.macro_f1 > 0

    y_true_reg = pd.Series([10000.0, 20000.0, 30000.0])
    y_pred_reg = np.array([10500.0, 19500.0, 30200.0])
    metrics_reg = evaluate_regression_model("TestReg", y_true_reg, y_pred_reg)
    assert metrics_reg.mae == pytest.approx(400.0, abs=1.0)
    assert metrics_reg.r2 > 0.95
