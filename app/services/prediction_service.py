import streamlit as st
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import sys

# Ensure root is in path
root_path = Path(__file__).resolve().parents[2]
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from src.tracking.model_registry import load_model_artifact
from src.utils.helpers import calculate_financial_ratios, format_currency
from src.explainability.local_explainability import generate_financial_factors
from src.utils.logger import setup_logger

logger = setup_logger("PredictionService")

@st.cache_resource
def get_cached_models():
    """Load and cache classification and regression models in memory."""
    clf_model = load_model_artifact("classification")
    reg_model = load_model_artifact("regression")
    return clf_model, reg_model

def predict_risk_and_emi(user_input: Dict[str, Any]) -> Dict[str, Any]:
    """Execute dual inference (Eligibility Classification + Max Monthly EMI Regression)."""
    clf_model, reg_model = get_cached_models()

    input_df = pd.DataFrame([user_input])
    ratios = calculate_financial_ratios(user_input)

    # 1. Classification Prediction
    predicted_eligibility = "Not_Eligible"
    confidence = 0.85
    probs = {"Not_Eligible": 0.85, "Eligible": 0.10, "High_Risk": 0.05}

    if clf_model is not None:
        try:
            pred_arr = clf_model.predict(input_df)
            predicted_eligibility = str(pred_arr[0])

            if hasattr(clf_model, "predict_proba"):
                proba_arr = clf_model.predict_proba(input_df)[0]
                classes = clf_model.classes_ if hasattr(clf_model, "classes_") else ["Eligible", "High_Risk", "Not_Eligible"]
                probs = {str(c): float(p) for c, p in zip(classes, proba_arr)}
                confidence = probs.get(predicted_eligibility, float(np.max(proba_arr)))
        except Exception as e:
            logger.error(f"Classification inference error: {e}")

    # 2. Regression Prediction
    predicted_max_emi = 500.0
    if reg_model is not None:
        try:
            reg_arr = reg_model.predict(input_df)
            predicted_max_emi = max(500.0, float(reg_arr[0]))
        except Exception as e:
            logger.error(f"Regression inference error: {e}")
            # Fallback estimation based on disposable income & risk class
            if predicted_eligibility == "Eligible":
                predicted_max_emi = max(1000.0, ratios["disposable_income"] * 0.40)
            elif predicted_eligibility == "High_Risk":
                predicted_max_emi = max(500.0, ratios["disposable_income"] * 0.20)
            else:
                predicted_max_emi = 500.0

    # 3. Driving Factors for Explanation
    driving_factors = generate_financial_factors(user_input, predicted_eligibility)

    return {
        "predicted_eligibility": predicted_eligibility,
        "confidence": confidence,
        "class_probabilities": probs,
        "predicted_max_emi": predicted_max_emi,
        "formatted_max_emi": format_currency(predicted_max_emi),
        "ratios": ratios,
        "driving_factors": driving_factors
    }
