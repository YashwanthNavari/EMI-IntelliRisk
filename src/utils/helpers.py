import numpy as np
import pandas as pd
from typing import Dict, Any, Union

def format_currency(amount: Union[float, int], symbol: str = "₹", precision: int = 2) -> str:
    """Format numbers into Indian standard currency notation (e.g. ₹1,25,000.00)."""
    if amount is None or (isinstance(amount, float) and np.isnan(amount)):
        return f"{symbol}0.00"

    is_negative = amount < 0
    amount = abs(amount)
    parts = f"{amount:.{precision}f}".split(".")
    integer_part = parts[0]
    decimal_part = f".{parts[1]}" if len(parts) > 1 and precision > 0 else ""

    if len(integer_part) <= 3:
        formatted = integer_part
    else:
        last3 = integer_part[-3:]
        remaining = integer_part[:-3]
        groups = []
        while len(remaining) > 2:
            groups.insert(0, remaining[-2:])
            remaining = remaining[:-2]
        if remaining:
            groups.insert(0, remaining)
        formatted = ",".join(groups) + "," + last3

    prefix = "-" if is_negative else ""
    return f"{prefix}{symbol}{formatted}{decimal_part}"

def calculate_financial_ratios(data: Dict[str, Any]) -> Dict[str, float]:
    """Calculate key FinTech financial health indicators from user inputs."""
    salary = float(data.get("monthly_salary", 0) or 0)
    eps = 1e-6

    rent = float(data.get("monthly_rent", 0) or 0)
    school = float(data.get("school_fees", 0) or 0)
    college = float(data.get("college_fees", 0) or 0)
    travel = float(data.get("travel_expenses", 0) or 0)
    groceries = float(data.get("groceries_utilities", 0) or 0)
    other = float(data.get("other_monthly_expenses", 0) or 0)
    current_emi = float(data.get("current_emi_amount", 0) or 0)
    bank_bal = float(data.get("bank_balance", 0) or 0)
    emerg_fund = float(data.get("emergency_fund", 0) or 0)
    req_amt = float(data.get("requested_amount", 0) or 0)
    req_tenure = float(data.get("requested_tenure", 1) or 1)
    family_size = max(1, int(data.get("family_size", 1) or 1))

    total_expenses = rent + school + college + travel + groceries + other
    total_commitments = total_expenses + current_emi
    disposable_income = salary - total_commitments

    expense_to_income = total_expenses / (salary + eps)
    foir = (rent + current_emi) / (salary + eps)
    emi_burden = current_emi / (salary + eps)
    savings_ratio = (bank_bal + emerg_fund) / (salary + eps)
    emergency_months = emerg_fund / (total_commitments + eps)
    loan_pressure = req_amt / (12 * salary + eps)
    est_installment = req_amt / (max(1.0, req_tenure) + eps)
    per_capita_disposable = disposable_income / family_size

    # Composite financial health score (0 to 100)
    score = 100.0
    if expense_to_income > 0.60:
        score -= min(30, (expense_to_income - 0.60) * 80)
    if foir > 0.40:
        score -= min(25, (foir - 0.40) * 70)
    if emi_burden > 0.30:
        score -= min(20, (emi_burden - 0.30) * 60)
    if emergency_months < 3.0:
        score -= max(0, (3.0 - emergency_months) * 5)
    if disposable_income < 0:
        score -= 25

    health_score = max(5.0, min(100.0, score))

    return {
        "total_expenses": total_expenses,
        "disposable_income": disposable_income,
        "expense_to_income_ratio": expense_to_income,
        "foir": foir,
        "emi_burden_ratio": emi_burden,
        "savings_to_income_ratio": savings_ratio,
        "emergency_fund_buffer_months": emergency_months,
        "loan_to_income_ratio": loan_pressure,
        "requested_monthly_installment_estimate": est_installment,
        "per_capita_disposable_income": per_capita_disposable,
        "financial_health_score": health_score
    }
