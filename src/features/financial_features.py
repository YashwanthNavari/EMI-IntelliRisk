import numpy as np
import pandas as pd
from ..utils.logger import setup_logger

logger = setup_logger("FinancialFeatures")

def compute_financial_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mathematical domain-specific financial features."""
    logger.info("Computing FinTech domain features...")
    df = df.copy()
    eps = 1e-6

    # 1. Total Living Expenses
    expense_cols = [
        "monthly_rent", "school_fees", "college_fees",
        "travel_expenses", "groceries_utilities", "other_monthly_expenses"
    ]
    available_expense_cols = [c for c in expense_cols if c in df.columns]
    df["total_expenses"] = df[available_expense_cols].fillna(0).sum(axis=1)

    # 2. Disposable Income
    current_emi = df["current_emi_amount"].fillna(0) if "current_emi_amount" in df.columns else 0
    salary = df["monthly_salary"].fillna(df["monthly_salary"].median() if "monthly_salary" in df.columns else 50000)
    df["disposable_income"] = salary - df["total_expenses"] - current_emi

    # 3. Expense-to-Income Ratio (ETI)
    df["expense_to_income_ratio"] = df["total_expenses"] / (salary + eps)

    # 4. Fixed Obligation to Income Ratio (FOIR)
    rent = df["monthly_rent"].fillna(0) if "monthly_rent" in df.columns else 0
    df["foir"] = (rent + current_emi) / (salary + eps)

    # 5. Existing EMI Burden Ratio
    df["emi_burden_ratio"] = current_emi / (salary + eps)

    # 6. Savings and Liquidity Ratios
    bank_bal = df["bank_balance"].fillna(0) if "bank_balance" in df.columns else 0
    emerg_fund = df["emergency_fund"].fillna(0) if "emergency_fund" in df.columns else 0
    total_commitments = df["total_expenses"] + current_emi

    df["savings_to_income_ratio"] = (bank_bal + emerg_fund) / (salary + eps)
    df["emergency_fund_buffer_months"] = emerg_fund / (total_commitments + eps)

    # 7. Loan Pressure and Installment Ratios
    req_amt = df["requested_amount"].fillna(0) if "requested_amount" in df.columns else 0
    req_tenure = df["requested_tenure"].replace(0, 1).fillna(12) if "requested_tenure" in df.columns else 12

    df["loan_to_income_ratio"] = req_amt / (12 * salary + eps)
    df["requested_monthly_installment_estimate"] = req_amt / (req_tenure + eps)

    # 8. Per-Capita Disposable Income
    family_size = df["family_size"].replace(0, 1).fillna(1) if "family_size" in df.columns else 1
    df["per_capita_disposable_income"] = df["disposable_income"] / family_size

    # 9. Affordability Ratio
    df["affordability_index"] = df["disposable_income"] / (df["requested_monthly_installment_estimate"] + eps)

    logger.info(f"Added {10} financial domain features.")
    return df
