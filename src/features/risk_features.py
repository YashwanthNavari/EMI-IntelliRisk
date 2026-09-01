import numpy as np
import pandas as pd
from ..utils.logger import setup_logger

logger = setup_logger("RiskFeatures")

def compute_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute credit risk tiers, composite burden index, and stability indicators."""
    logger.info("Computing credit risk and stability features...")
    df = df.copy()

    # 1. Credit Score Tiers
    if "credit_score" in df.columns:
        cs = df["credit_score"].fillna(700)
        conditions = [
            (cs < 650),
            (cs >= 650) & (cs < 700),
            (cs >= 700) & (cs < 750),
            (cs >= 750)
        ]
        choices = ["Poor", "Fair", "Good", "Excellent"]
        df["credit_risk_tier"] = np.select(conditions, choices, default="Good")

    # 2. Employment Stability Indicator
    if "years_of_employment" in df.columns:
        yoe = df["years_of_employment"].fillna(2.0)
        df["employment_stability_score"] = np.select(
            [yoe < 1.0, (yoe >= 1.0) & (yoe < 3.0), (yoe >= 3.0) & (yoe < 7.0), yoe >= 7.0],
            ["Probation/Junior", "Moderate", "Established", "Senior/Veteran"],
            default="Moderate"
        )

    # 3. Composite Financial Health Score (0-100)
    # Vectorized calculation
    salary = df["monthly_salary"].replace(0, 1) if "monthly_salary" in df.columns else 50000
    eti = df.get("expense_to_income_ratio", pd.Series(0.6, index=df.index))
    foir = df.get("foir", pd.Series(0.3, index=df.index))
    emi_burden = df.get("emi_burden_ratio", pd.Series(0.1, index=df.index))
    emerg_buf = df.get("emergency_fund_buffer_months", pd.Series(2.0, index=df.index))
    disp_inc = df.get("disposable_income", pd.Series(10000, index=df.index))

    score = pd.Series(100.0, index=df.index)
    score -= np.maximum(0.0, (eti - 0.60) * 80.0)
    score -= np.maximum(0.0, (foir - 0.40) * 70.0)
    score -= np.maximum(0.0, (emi_burden - 0.30) * 60.0)
    score -= np.maximum(0.0, (3.0 - emerg_buf) * 5.0)
    score = np.where(disp_inc < 0, score - 25.0, score)

    df["financial_health_score"] = np.clip(score, 5.0, 100.0)

    logger.info("Added credit risk tier, stability score, and financial health score.")
    return df
