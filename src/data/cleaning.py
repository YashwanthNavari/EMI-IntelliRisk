import re
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from ..utils.logger import setup_logger
from ..utils.config import load_config

logger = setup_logger("DataCleaning")

def clean_dirty_numeric_series(series: pd.Series) -> pd.Series:
    """Strip repeated '.0' suffixes or unwanted non-numeric characters and parse as float."""
    def _clean_str(val):
        if pd.isna(val):
            return np.nan
        if isinstance(val, (int, float)):
            return float(val)
        val_str = str(val).strip()
        # Remove trailing repeated '.0' occurrences
        val_str = re.sub(r"(\.0)+$", "", val_str)
        # Extract first valid numeric floating or integer pattern
        match = re.search(r"[-+]?\d*\.?\d+", val_str)
        if match:
            try:
                return float(match.group(0))
            except ValueError:
                return np.nan
        return np.nan

    return series.apply(_clean_str).astype(float)

def normalize_gender(series: pd.Series) -> pd.Series:
    """Normalize inconsistent gender casing and abbreviations."""
    mapping = {
        "male": "Male", "m": "Male", "female": "Female", "f": "Female"
    }
    return series.astype(str).str.strip().str.lower().map(mapping).fillna("Male")

def clean_dataset(df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
    """End-to-end robust data cleaning pipeline."""
    logger.info(f"Initiating data cleaning pipeline on {len(df):,} records...")
    df = df.copy()

    # 1. Clean Object Numerics with String Formatting Anomalies
    for num_col in ["age", "monthly_salary", "bank_balance", "monthly_rent", "credit_score", "emergency_fund"]:
        if num_col in df.columns and (df[num_col].dtype == "object" or num_col in ["age", "monthly_salary", "bank_balance"]):
            df[num_col] = clean_dirty_numeric_series(df[num_col])

    # 2. Normalize Categorical Columns
    if "gender" in df.columns:
        df["gender"] = normalize_gender(df["gender"])

    for cat_col in ["marital_status", "education", "employment_type", "company_type", "house_type", "existing_loans", "emi_scenario"]:
        if cat_col in df.columns:
            df[cat_col] = df[cat_col].astype(str).str.strip()
            # Restore NaN where string was 'nan' or empty
            df.loc[df[cat_col].isin(["nan", "None", "", "NaN"]), cat_col] = np.nan

    # 3. Missing Value Handling
    if "education" in df.columns:
        df["education"] = df["education"].fillna("Graduate")

    if "house_type" in df.columns and "monthly_rent" in df.columns:
        # Non-rented houses naturally incur 0 rent
        df.loc[df["house_type"].isin(["Own", "Family"]) & df["monthly_rent"].isna(), "monthly_rent"] = 0.0
        df["monthly_rent"] = df["monthly_rent"].fillna(0.0)

    # Impute remaining numeric medians
    median_cols = ["age", "monthly_salary", "credit_score", "bank_balance", "emergency_fund"]
    for col in median_cols:
        if col in df.columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    # 4. Outlier & Domain Boundary Bounds
    if "credit_score" in df.columns:
        # Clip credit score between 300 and 900 (standard CIBIL/FICO range)
        # For scores <= 0, map to 300 (poor credit history)
        df.loc[df["credit_score"] < 300, "credit_score"] = 300.0
        df.loc[df["credit_score"] > 900, "credit_score"] = 900.0

    if "age" in df.columns:
        df["age"] = df["age"].clip(18.0, 80.0)

    logger.info(f"Data cleaning completed successfully. Resulting shape: {df.shape}")
    return df
