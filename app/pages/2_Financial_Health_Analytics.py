import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

root_path = Path(__file__).resolve().parents[2]
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from app.components.ui_cards import render_kpi_card
from app.components.charts import plot_budget_breakdown, plot_gauge_meter
from src.utils.helpers import format_currency, calculate_financial_ratios
from database.repository import CustomerRepository

st.set_page_config(page_title="Financial Health Analytics", page_icon="📈", layout="wide")

css_file = root_path / "app" / "assets" / "styles.css"
if css_file.exists():
    with open(css_file, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1 class="header-title">📈 Financial Health & Cash Flow Analytics</h1>
    <div class="header-subtitle">Deterministic Underwriting Analytics, Stress-Testing Simulator, and Budget Health Diagnostics</div>
</div>
""", unsafe_allow_html=True)

st.info("ℹ️ **Note**: This module computes rule-based financial indicators and macroeconomic stress-tests for decision support, distinct from statistical ML predictions.")

# Input profile
st.subheader("Configure Financial Scenario")
c1, c2, c3, c4 = st.columns(4)

with c1:
    salary = st.number_input("Monthly Salary (₹)", min_value=10000.0, max_value=1000000.0, value=75000.0, step=5000.0)
    rent = st.number_input("Monthly Rent (₹)", min_value=0.0, max_value=150000.0, value=16000.0, step=1000.0)

with c2:
    groceries = st.number_input("Groceries & Utilities (₹)", min_value=0.0, max_value=100000.0, value=15000.0, step=1000.0)
    travel = st.number_input("Travel Expenses (₹)", min_value=0.0, max_value=50000.0, value=5000.0, step=500.0)

with c3:
    school_college = st.number_input("Education Fees (₹)", min_value=0.0, max_value=100000.0, value=4000.0, step=500.0)
    other = st.number_input("Other Living Expenses (₹)", min_value=0.0, max_value=50000.0, value=6000.0, step=500.0)

with c4:
    current_emi = st.number_input("Current Monthly EMI (₹)", min_value=0.0, max_value=200000.0, value=8000.0, step=1000.0)
    emergency_fund = st.number_input("Emergency Reserve (₹)", min_value=0.0, max_value=2000000.0, value=120000.0, step=10000.0)

# Calculate base ratios
expenses_dict = {
    "Rent": rent,
    "Groceries & Utilities": groceries,
    "Travel": travel,
    "Education": school_college,
    "Other Living": other
}
total_expenses = sum(expenses_dict.values())
disposable_income = salary - total_expenses - current_emi
foir = (rent + current_emi) / salary if salary > 0 else 0
eti = total_expenses / salary if salary > 0 else 0
emerg_months = emergency_fund / (total_expenses + current_emi + 1e-6)

st.markdown("---")
st.subheader("1. Monthly Cash Flow & Budget Allocation")

k1, k2, k3, k4 = st.columns(4)
with k1:
    render_kpi_card("Monthly Disposable Surplus", format_currency(disposable_income), f"{(disposable_income/salary)*100:.1f}% of Income", delta_color="#10B981" if disposable_income > 0 else "#EF4444")
with k2:
    render_kpi_card("Fixed Obligations (FOIR)", f"{foir*100:.1f}%", "Ceiling Benchmark: 50.0%", delta_color="#10B981" if foir <= 0.50 else "#EF4444")
with k3:
    render_kpi_card("Living Expense Burden (ETI)", f"{eti*100:.1f}%", f"{format_currency(total_expenses)} total", delta_color="#38BDF8")
with k4:
    render_kpi_card("Emergency Reserve Coverage", f"{emerg_months:.1f} Months", "Recommended: 3+ Months", delta_color="#10B981" if emerg_months >= 3.0 else "#F59E0B")

col_chart1, col_chart2 = st.columns([1.3, 1])

with col_chart1:
    st.plotly_chart(plot_budget_breakdown(salary, expenses_dict, current_emi, disposable_income), use_container_width=True)

with col_chart2:
    st.plotly_chart(plot_gauge_meter(round(foir * 100, 1), "Fixed Obligation Ratio (FOIR %)", max_val=100.0, threshold=50.0), use_container_width=True)

st.markdown("---")
st.subheader("2. Financial Stress-Testing & Resilience Simulator")
st.markdown("Simulate how macroeconomic or personal shocks affect debt affordability:")

sc1, sc2, sc3 = st.columns(3)
with sc1:
    exp_shock = st.slider("Living Expense Inflation (%)", 0, 40, 10, step=5)
with sc2:
    income_shock = st.slider("Income Reduction / Pay Cut (%)", 0, 30, 0, step=5)
with sc3:
    extra_emi = st.slider("Additional Loan Obligation (₹)", 0, 30000, 5000, step=1000)

stressed_salary = salary * (1 - income_shock / 100)
stressed_expenses = total_expenses * (1 + exp_shock / 100)
stressed_commitments = stressed_expenses + current_emi + extra_emi
stressed_surplus = stressed_salary - stressed_commitments
stressed_foir = (rent + current_emi + extra_emi) / stressed_salary if stressed_salary > 0 else 0

sk1, sk2, sk3 = st.columns(3)
with sk1:
    render_kpi_card("Stressed Monthly Surplus", format_currency(stressed_surplus), f"Impact: -{format_currency(disposable_income - stressed_surplus)}", delta_color="#10B981" if stressed_surplus > 0 else "#EF4444")
with sk2:
    render_kpi_card("Stressed FOIR", f"{stressed_foir*100:.1f}%", "Safe threshold: < 50%", delta_color="#10B981" if stressed_foir <= 0.50 else "#EF4444")
with sk3:
    resilience = "HIGH" if (stressed_surplus > 10000 and stressed_foir < 0.50) else ("MODERATE" if stressed_surplus > 0 else "CRITICAL RISK")
    color = "#10B981" if resilience == "HIGH" else ("#F59E0B" if resilience == "MODERATE" else "#EF4444")
    render_kpi_card("Stress Resilience Tier", resilience, "Under Simulated Shocks", delta_color=color)
