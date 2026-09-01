import streamlit as st
import pandas as pd
from pathlib import Path
import sys

root_path = Path(__file__).resolve().parents[2]
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from app.components.charts import plot_feature_importance_bar
from app.services.prediction_service import get_cached_models
from src.explainability.global_explainability import get_global_feature_importance

st.set_page_config(page_title="Explainable AI (XAI)", page_icon="🧠", layout="wide")

css_file = root_path / "app" / "assets" / "styles.css"
if css_file.exists():
    with open(css_file, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1 class="header-title">🧠 Explainable AI & Financial Attribution</h1>
    <div class="header-subtitle">Global Feature Attributions, Local Decision Decomposition, and Transparent Factor Modeling</div>
</div>
""", unsafe_allow_html=True)

clf_model, reg_model = get_cached_models()

tab1, tab2 = st.tabs(["🌐 Global Feature Importance", "🔬 Explainability Methodology"])

with tab1:
    st.subheader("1. Global Model Drivers Across 404.8K Records")
    st.markdown("Feature importance indicates the relative contribution of each variable to underwriting classifications and EMI amount estimations:")

    col_imp1, col_imp2 = st.columns(2)

    with col_imp1:
        st.markdown("#### Classification Attribution (Eligibility)")
        if clf_model is not None:
            df_imp_clf = get_global_feature_importance(clf_model, top_n=12)
            st.plotly_chart(plot_feature_importance_bar(df_imp_clf, top_n=12), use_container_width=True)
        else:
            st.info("Loading classifier...")

    with col_imp2:
        st.markdown("#### Regression Attribution (Max Monthly EMI)")
        if reg_model is not None:
            df_imp_reg = get_global_feature_importance(reg_model, top_n=12)
            st.plotly_chart(plot_feature_importance_bar(df_imp_reg, top_n=12), use_container_width=True)
        else:
            st.info("Loading regressor...")

with tab2:
    st.subheader("2. Explainability Framework & Theoretical Foundations")
    st.markdown("""
    ### Dual Explainability Architecture:
    1. **TreeSHAP & Tree Feature Attributions (Global)**:
       - Measures the average marginal contribution of each financial metric across all potential coalition subsets of features.
       - Confirms that *Disposable Income*, *FOIR*, and *Credit Score* are the primary drivers of loan approval.

    2. **Evidence-Based Transparent Rule Attribution (Local)**:
       - Every individual underwriting assessment is decomposed into actionable, human-interpretable factors:
         - **Fixed Obligation Ratio (FOIR)**: Checks whether existing commitments exceed the 50% prudent ceiling.
         - **Free Cash Flow Surplus**: Verifies sufficient monthly buffer after living costs.
         - **Credit Bureau Score Tier**: Assesses historical repayment delinquency risks.
         - **Emergency Reserve Coverage**: Evaluates resilience against income disruption.
    """)
