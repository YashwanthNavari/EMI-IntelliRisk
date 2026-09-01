import numpy as np
import pandas as pd
from typing import Dict, List, Any
from ..utils.logger import setup_logger
from ..utils.helpers import calculate_financial_ratios

logger = setup_logger("LocalExplainability")

def generate_financial_factors(user_input: Dict[str, Any], prediction: str) -> List[Dict[str, str]]:
    """Derive transparent, evidence-based driving factors from applicant inputs."""
    ratios = calculate_financial_ratios(user_input)
    factors = []

    salary = float(user_input.get("monthly_salary", 0) or 0)
    credit_score = float(user_input.get("credit_score", 700) or 700)
    current_emi = float(user_input.get("current_emi_amount", 0) or 0)

    # 1. FOIR Assessment
    foir = ratios["foir"]
    if foir > 0.50:
        factors.append({
            "type": "negative",
            "title": "High Fixed Obligation Ratio (FOIR)",
            "detail": f"Fixed obligations consume {foir*100:.1f}% of income (>50% ceiling threshold)."
        })
    elif foir < 0.30:
        factors.append({
            "type": "positive",
            "title": "Healthy Fixed Obligation Ratio",
            "detail": f"Fixed debt/rent commitments consume only {foir*100:.1f}% of monthly income."
        })

    # 2. Disposable Income
    disp = ratios["disposable_income"]
    if disp < 5000:
        factors.append({
            "type": "negative",
            "title": "Restricted Disposable Income",
            "detail": f"Estimated remaining cash flow is ₹{disp:,.2f}, leaving minimal buffer for additional loan servicing."
        })
    else:
        factors.append({
            "type": "positive",
            "title": "Adequate Free Cash Flow",
            "detail": f"Estimated disposable surplus is ₹{disp:,.2f} per month."
        })

    # 3. Credit Score Health
    if credit_score < 650:
        factors.append({
            "type": "negative",
            "title": "Sub-Prime Credit Score",
            "detail": f"Credit score of {credit_score:.0f} indicates elevated historical default risk."
        })
    elif credit_score >= 750:
        factors.append({
            "type": "positive",
            "title": "Prime Credit Score",
            "detail": f"Credit score of {credit_score:.0f} demonstrates strong repayment discipline."
        })

    # 4. Emergency Buffer
    buffer_m = ratios["emergency_fund_buffer_months"]
    if buffer_m < 1.0:
        factors.append({
            "type": "negative",
            "title": "Minimal Liquidity Buffer",
            "detail": f"Liquid emergency reserves cover less than 1 month of recurring expenses ({buffer_m:.1f} months)."
        })
    elif buffer_m >= 3.0:
        factors.append({
            "type": "positive",
            "title": "Robust Emergency Buffer",
            "detail": f"Liquid emergency savings can cover {buffer_m:.1f} months of expenses."
        })

    # 5. Loan Pressure
    loan_pres = ratios["loan_to_income_ratio"]
    if loan_pres > 2.0:
        factors.append({
            "type": "negative",
            "title": "Elevated Loan-to-Income Multiple",
            "detail": f"Requested amount is {loan_pres:.1f}x total annual income."
        })

    # Ensure we return at least 3 factors
    if len(factors) < 3:
        if prediction == "Eligible":
            factors.append({
                "type": "positive",
                "title": "Favorable Debt Burden Profile",
                "detail": f"Current EMI burden is {ratios['emi_burden_ratio']*100:.1f}% of income."
            })
        else:
            factors.append({
                "type": "negative",
                "title": "High Living Expense Burden",
                "detail": f"Living expenses account for {ratios['expense_to_income_ratio']*100:.1f}% of salary."
            })

    return factors

def explain_individual_prediction(pipeline: Any, input_df: pd.DataFrame) -> Dict[str, Any]:
    """Provide local explanation for an individual prediction."""
    try:
        user_dict = input_df.iloc[0].to_dict()
        pred_class = pipeline.predict(input_df)[0]
        factors = generate_financial_factors(user_dict, str(pred_class))
        return {
            "prediction": str(pred_class),
            "driving_factors": factors
        }
    except Exception as e:
        logger.warning(f"Error generating local explanation: {e}")
        return {
            "prediction": "Unknown",
            "driving_factors": []
        }
