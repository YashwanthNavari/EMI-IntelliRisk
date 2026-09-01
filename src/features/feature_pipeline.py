import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from .transformations import FinancialFeatureTransformer
from ..utils.logger import setup_logger
from ..utils.config import load_config

logger = setup_logger("FeaturePipeline")

def get_feature_columns() -> Tuple[List[str], List[str]]:
    """Return complete list of numeric and categorical features including engineered ones."""
    numeric_features = [
        "age",
        "monthly_salary",
        "years_of_employment",
        "monthly_rent",
        "family_size",
        "dependents",
        "school_fees",
        "college_fees",
        "travel_expenses",
        "groceries_utilities",
        "other_monthly_expenses",
        "current_emi_amount",
        "credit_score",
        "bank_balance",
        "emergency_fund",
        "requested_amount",
        "requested_tenure",
        # Engineered numerics
        "total_expenses",
        "disposable_income",
        "expense_to_income_ratio",
        "foir",
        "emi_burden_ratio",
        "savings_to_income_ratio",
        "emergency_fund_buffer_months",
        "loan_to_income_ratio",
        "requested_monthly_installment_estimate",
        "per_capita_disposable_income",
        "affordability_index",
        "financial_health_score"
    ]

    categorical_features = [
        "gender",
        "marital_status",
        "education",
        "employment_type",
        "company_type",
        "house_type",
        "existing_loans",
        "emi_scenario",
        # Engineered categoricals
        "credit_risk_tier",
        "employment_stability_score"
    ]

    return numeric_features, categorical_features

def create_preprocessor_pipeline(scale_numerics: bool = True) -> Pipeline:
    """Create reusable ColumnTransformer preprocessor pipeline."""
    num_cols, cat_cols = get_feature_columns()

    # Numeric Pipeline
    num_steps = [
        ("imputer", SimpleImputer(strategy="median"))
    ]
    if scale_numerics:
        num_steps.append(("scaler", RobustScaler()))

    num_pipeline = Pipeline(steps=num_steps)

    # Categorical Pipeline
    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    col_transformer = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols)
        ],
        remainder="drop"
    )

    full_pipeline = Pipeline(steps=[
        ("feature_creator", FinancialFeatureTransformer()),
        ("preprocessor", col_transformer)
    ])

    return full_pipeline

def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """Extract output feature names after ColumnTransformer fitting."""
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        num_cols, cat_cols = get_feature_columns()
        return num_cols + cat_cols
