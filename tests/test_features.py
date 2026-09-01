import pytest
import pandas as pd
import numpy as np
from src.features.financial_features import compute_financial_features
from src.features.risk_features import compute_risk_features
from src.utils.helpers import calculate_financial_ratios

def test_financial_features_math():
    df = pd.DataFrame([{
        "monthly_salary": 100000.0,
        "monthly_rent": 20000.0,
        "school_fees": 5000.0,
        "college_fees": 0.0,
        "travel_expenses": 5000.0,
        "groceries_utilities": 15000.0,
        "other_monthly_expenses": 5000.0,
        "current_emi_amount": 10000.0,
        "bank_balance": 300000.0,
        "emergency_fund": 150000.0,
        "requested_amount": 240000.0,
        "requested_tenure": 24,
        "family_size": 4
    }])

    feat_df = compute_financial_features(df)

    assert feat_df["total_expenses"].iloc[0] == 50000.0
    assert feat_df["disposable_income"].iloc[0] == 40000.0
    assert feat_df["expense_to_income_ratio"].iloc[0] == pytest.approx(0.50, abs=1e-3)
    assert feat_df["foir"].iloc[0] == pytest.approx(0.30, abs=1e-3)
    assert feat_df["emi_burden_ratio"].iloc[0] == pytest.approx(0.10, abs=1e-3)
    assert feat_df["per_capita_disposable_income"].iloc[0] == 10000.0

def test_risk_features_logic():
    df = pd.DataFrame([
        {"credit_score": 780.0, "years_of_employment": 8.0, "monthly_salary": 80000.0, "disposable_income": 30000.0},
        {"credit_score": 620.0, "years_of_employment": 0.5, "monthly_salary": 25000.0, "disposable_income": -5000.0}
    ])

    risk_df = compute_risk_features(df)
    assert risk_df["credit_risk_tier"].iloc[0] == "Excellent"
    assert risk_df["credit_risk_tier"].iloc[1] == "Poor"
    assert risk_df["employment_stability_score"].iloc[0] == "Senior/Veteran"
    assert risk_df["employment_stability_score"].iloc[1] == "Probation/Junior"

def test_calculate_financial_ratios_helper():
    sample = {
        "monthly_salary": 50000.0,
        "monthly_rent": 10000.0,
        "current_emi_amount": 5000.0,
        "emergency_fund": 50000.0
    }
    ratios = calculate_financial_ratios(sample)
    assert ratios["foir"] == pytest.approx(0.30, abs=1e-3)
    assert ratios["financial_health_score"] >= 0.0
    assert ratios["financial_health_score"] <= 100.0
