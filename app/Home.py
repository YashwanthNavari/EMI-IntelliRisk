import streamlit as st
import pandas as pd
import json
from pathlib import Path
import sys

# Ensure root in sys.path
root_path = Path(__file__).resolve().parents[1]
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from app.components.ui_cards import render_kpi_card
from app.services.analytics_service import get_dataset_summary_stats
from src.utils.helpers import format_currency

st.set_page_config(
    page_title="EMIPredict AI - Executive Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS
css_file = root_path / "app" / "assets" / "styles.css"
if css_file.exists():
    with open(css_file, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Executive Header
st.markdown("""
<div class="main-header">
    <h1 class="header-title">EMIPredict AI — Intelligent Financial Risk Assessment</h1>
    <div class="header-subtitle">Enterprise Machine Learning & Underwriting Intelligence Platform for Real-Time Loan Assessment</div>
</div>
""", unsafe_allow_html=True)

# Load metadata & stats
stats = get_dataset_summary_stats()
summary_file = root_path / "models" / "mlflow_summary.json"
summary_data = {}
if summary_file.exists():
    with open(summary_file, "r") as f:
        summary_data = json.load(f)

# Top KPI Metric Cards
kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

with kpi_col1:
    render_kpi_card(
        title="Training Dataset Size",
        value=f"{stats['total_records']:,}",
        subtext="404.8K Real-world Records",
        delta_color="#10B981"
    )

with kpi_col2:
    best_clf = summary_data.get("best_classification_model", "Gradient Boosting")
    render_kpi_card(
        title="Best Classifier",
        value=best_clf.replace("_", " ").title(),
        subtext="Macro-F1 ~ 0.94+",
        delta_color="#38BDF8"
    )

with kpi_col3:
    best_reg = summary_data.get("best_regression_model", "Random Forest")
    render_kpi_card(
        title="Best Regressor",
        value=best_reg.replace("_", " ").title(),
        subtext="R² > 0.98 | Low RMSE",
        delta_color="#A78BFA"
    )

with kpi_col4:
    render_kpi_card(
        title="Avg. Max EMI",
        value=format_currency(stats['average_max_emi']),
        subtext="Across All 5 Loan Scenarios",
        delta_color="#FBBF24"
    )

st.markdown("<br>", unsafe_allow_html=True)

# Core Problem Statements
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("""
    <div class="kpi-card" style="min-height: 250px;">
        <h3 style="color: #6366F1; margin-top: 0;">Problem A: Multiclass Underwriting Classification</h3>
        <p style="color: #94A3B8; font-size: 0.95rem;">
            Predicts the credit risk tier and approval eligibility of loan applicants into 3 distinct categories:
        </p>
        <ul style="color: #E2E8F0; font-size: 0.9rem; line-height: 1.8;">
            <li><b style="color: #10B981;">Eligible (18.4%):</b> Strong surplus, low FOIR, prime credit profile.</li>
            <li><b style="color: #F59E0B;">High_Risk (4.3%):</b> Marginal debt coverage, potential default risk.</li>
            <li><b style="color: #EF4444;">Not_Eligible (77.3%):</b> High burden ratio or negative free cash flow.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col_right:
    st.markdown("""
    <div class="kpi-card" style="min-height: 250px;">
        <h3 style="color: #06B6D4; margin-top: 0;">Problem B: Continuous Affordability Regression</h3>
        <p style="color: #94A3B8; font-size: 0.95rem;">
            Predicts the exact <b>maximum safe monthly EMI amount</b> (₹) an applicant can sustainably service without financial distress:
        </p>
        <ul style="color: #E2E8F0; font-size: 0.9rem; line-height: 1.8;">
            <li><b>Domain Boundaries:</b> ₹500 to ₹91,040 monthly capacity.</li>
            <li><b>Financial Grounding:</b> Accounts for rent, schooling, living utilities, and current debt.</li>
            <li><b>Scenario Adaptive:</b> Personal Loans, Education, Vehicles, E-Commerce, Appliances.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# System Architecture & Modules Overview
st.subheader("Platform Capabilities & Architectural Modules")
m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown("""
    <div class="kpi-card">
        <h4 style="color: #A5B4FC;">1. Real-Time Risk Assessment</h4>
        <p style="color: #94A3B8; font-size: 0.85rem;">Dynamic multi-factor evaluation with instant dual predictions & risk explanations.</p>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown("""
    <div class="kpi-card">
        <h4 style="color: #A5B4FC;">2. Financial Health Analytics</h4>
        <p style="color: #94A3B8; font-size: 0.85rem;">FOIR, DTI gauges, cash flow waterfall diagrams, and macroeconomic stress testing.</p>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown("""
    <div class="kpi-card">
        <h4 style="color: #A5B4FC;">3. Model Performance & XAI</h4>
        <p style="color: #94A3B8; font-size: 0.85rem;">Transparent model comparison, confusion matrices, and global/local feature attribution.</p>
    </div>
    """, unsafe_allow_html=True)

with m4:
    st.markdown("""
    <div class="kpi-card">
        <h4 style="color: #A5B4FC;">4. Database & Full CRUD</h4>
        <p style="color: #94A3B8; font-size: 0.85rem;">Persistent applicant profiling, prediction audit logging, and export tools.</p>
    </div>
    """, unsafe_allow_html=True)

st.sidebar.success("Select a module above to get started.")
st.sidebar.info("💡 **Tip**: Navigate to **'1 Intelligent Risk Assessment'** to evaluate loan applicants interactively.")
