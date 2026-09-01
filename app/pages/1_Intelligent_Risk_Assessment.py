import streamlit as st
import pandas as pd
from pathlib import Path
import sys

root_path = Path(__file__).resolve().parents[2]
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from app.services.prediction_service import predict_risk_and_emi
from app.components.ui_cards import render_kpi_card, render_eligibility_badge, render_factor_card
from app.components.charts import plot_probability_distribution, plot_gauge_meter
from database.repository import CustomerRepository, PredictionRepository
from src.utils.helpers import format_currency

st.set_page_config(page_title="Underwriting Workstation", page_icon="⚡", layout="wide")

css_file = root_path / "app" / "assets" / "styles.css"
if css_file.exists():
    with open(css_file, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h1 class="header-title">⚡ Intelligent Underwriting Workstation</h1>
            <div class="header-subtitle">Dual-Model Credit Risk Assessment, Continuous Affordability Engine & Decision Attribution</div>
        </div>
        <div style="background: rgba(16, 185, 129, 0.2); border: 1px solid #10B981; padding: 6px 16px; border-radius: 20px; color: #34D399; font-weight: 600; font-size: 0.85rem;">
            ● UNDERWRITING ENGINE ACTIVE
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Customer pre-loader
customers = CustomerRepository.list_customers(limit=50)
cust_options = {"-- Select Existing Profile or Enter Manually --": None}
for c in customers:
    cust_options[f"#{c['id']}: {c['full_name']} ({c['email'] or 'No Email'})"] = c

top_c1, top_c2 = st.columns([2.5, 1])
with top_c1:
    selected_cust_key = st.selectbox("👤 Quick-Load Customer Financial Profile:", list(cust_options.keys()))
    loaded_cust = cust_options[selected_cust_key]
with top_c2:
    st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
    if loaded_cust:
        st.caption(f"Loaded Profile ID: #{loaded_cust['id']} | Last Updated: {loaded_cust.get('created_at', 'Active')}")

st.markdown("---")

form_col, result_col = st.columns([1.1, 1.2])

with form_col:
    st.markdown("### 📝 Applicant Financial Inputs")
    with st.form("workstation_form"):
        # Section 1: Demographics & Income
        with st.expander("👤 1. Applicant & Income Disclosures", expanded=True):
            f1, f2 = st.columns(2)
            with f1:
                age = st.number_input("Age", 18, 80, int(loaded_cust["age"]) if loaded_cust else 32)
                gender = st.selectbox("Gender", ["Male", "Female"], index=0 if not loaded_cust or loaded_cust["gender"] == "Male" else 1)
                marital_status = st.selectbox("Marital Status", ["Married", "Single"], index=0 if not loaded_cust or loaded_cust["marital_status"] == "Married" else 1)
                education = st.selectbox("Education", ["Graduate", "Post Graduate", "High School", "Professional"], index=0)
            with f2:
                monthly_salary = st.number_input("Monthly Gross Salary (₹)", 5000.0, 1000000.0, float(loaded_cust["monthly_salary"]) if loaded_cust else 65000.0, step=2500.0)
                employment_type = st.selectbox("Employment Type", ["Private", "Government", "Self-employed"], index=0)
                years_of_employment = st.number_input("Employment Experience (Yrs)", 0.0, 45.0, float(loaded_cust["years_of_employment"]) if loaded_cust else 4.5, step=0.5)
                company_type = st.selectbox("Company Type", ["Large Indian", "MNC", "Mid-size", "Startup", "Small"], index=1)

        # Section 2: Residential & Household
        with st.expander("🏠 2. Residential & Living Commitments", expanded=True):
            r1, r2 = st.columns(2)
            with r1:
                house_type = st.selectbox("House Ownership", ["Rented", "Own", "Family"], index=0)
                monthly_rent = st.number_input("Monthly Rent (₹)", 0.0, 200000.0, float(loaded_cust["monthly_rent"]) if loaded_cust else 15000.0, step=1000.0)
                family_size = st.number_input("Household Size", 1, 10, int(loaded_cust["family_size"]) if loaded_cust else 3)
                dependents = st.number_input("Number of Dependents", 0, 8, int(loaded_cust["dependents"]) if loaded_cust else 1)
            with r2:
                groceries_utilities = st.number_input("Groceries & Utilities (₹)", 0.0, 150000.0, float(loaded_cust["groceries_utilities"]) if loaded_cust else 14000.0, step=1000.0)
                travel_expenses = st.number_input("Travel & Commute (₹)", 0.0, 100000.0, float(loaded_cust["travel_expenses"]) if loaded_cust else 5000.0, step=500.0)
                school_fees = st.number_input("School Tuition (₹)", 0.0, 100000.0, float(loaded_cust["school_fees"]) if loaded_cust else 2000.0, step=500.0)
                college_fees = st.number_input("Higher Education / College (₹)", 0.0, 150000.0, float(loaded_cust["college_fees"]) if loaded_cust else 0.0, step=1000.0)
                other_monthly_expenses = st.number_input("Other Discretionary Costs (₹)", 0.0, 100000.0, float(loaded_cust["other_monthly_expenses"]) if loaded_cust else 5000.0, step=500.0)

        # Section 3: Existing Debt & Credit
        with st.expander("💳 3. Existing Obligations & Credit Profile", expanded=True):
            d1, d2 = st.columns(2)
            with d1:
                existing_loans = st.selectbox("Active Existing Loans?", ["No", "Yes"], index=0 if not loaded_cust or loaded_cust["existing_loans"] == "No" else 1)
                current_emi_amount = st.number_input("Current Monthly EMI Debt (₹)", 0.0, 200000.0, float(loaded_cust["current_emi_amount"]) if loaded_cust else 0.0, step=1000.0)
                credit_score = st.slider("Credit Bureau Score (CIBIL)", 300, 900, int(loaded_cust["credit_score"]) if loaded_cust else 740, step=5)
            with d2:
                bank_balance = st.number_input("Savings Bank Balance (₹)", 0.0, 5000000.0, float(loaded_cust["bank_balance"]) if loaded_cust else 250000.0, step=10000.0)
                emergency_fund = st.number_input("Emergency Liquid Reserves (₹)", 0.0, 2000000.0, float(loaded_cust["emergency_fund"]) if loaded_cust else 80000.0, step=5000.0)

        # Section 4: Loan Request
        with st.expander("🎯 4. Loan Application Terms", expanded=True):
            l1, l2 = st.columns(2)
            with l1:
                emi_scenario = st.selectbox("EMI Loan Purpose", [
                    "Personal Loan EMI", "E-commerce Shopping EMI", "Education EMI", "Vehicle EMI", "Home Appliances EMI"
                ], index=0)
                requested_amount = st.number_input("Requested Principal Amount (₹)", 10000.0, 3000000.0, 350000.0, step=25000.0)
            with l2:
                requested_tenure = st.number_input("Requested Tenure (Months)", 3, 84, 24, step=1)
                est_installment = requested_amount / max(1, requested_tenure)
                st.caption(f"Estimated Monthly Installment: **{format_currency(est_installment)}/mo**")

        eval_btn = st.form_submit_button("⚡ Evaluate Applicant Risk & Affordability", use_container_width=True)

with result_col:
    st.markdown("### 📊 Underwriting Decision Dossier")

    user_payload = {
        "age": age, "gender": gender, "marital_status": marital_status, "education": education,
        "monthly_salary": monthly_salary, "employment_type": employment_type, "years_of_employment": years_of_employment,
        "company_type": company_type, "house_type": house_type, "monthly_rent": monthly_rent,
        "family_size": family_size, "dependents": dependents, "school_fees": school_fees, "college_fees": college_fees,
        "travel_expenses": travel_expenses, "groceries_utilities": groceries_utilities, "other_monthly_expenses": other_monthly_expenses,
        "existing_loans": existing_loans, "current_emi_amount": current_emi_amount, "credit_score": float(credit_score),
        "bank_balance": bank_balance, "emergency_fund": emergency_fund, "emi_scenario": emi_scenario,
        "requested_amount": requested_amount, "requested_tenure": requested_tenure
    }

    res = predict_risk_and_emi(user_payload)

    # Primary Decision Card
    st.markdown("""
    <div class="kpi-card" style="margin-bottom: 16px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div style="font-size: 0.85rem; font-weight: 600; text-transform: uppercase; color: #94A3B8;">AI Underwriting Recommendation</div>
            <div style="font-size: 0.85rem; color: #38BDF8;">Model v1.0.0 (Champion Gradient Boosting)</div>
        </div>
        <div style="margin-top: 12px; margin-bottom: 12px;">
    """, unsafe_allow_html=True)
    render_eligibility_badge(res["predicted_eligibility"], res["confidence"])
    st.markdown("</div></div>", unsafe_allow_html=True)

    # Probability & Max EMI Split
    p_col1, p_col2 = st.columns(2)
    with p_col1:
        render_kpi_card(
            title="Max Safe Monthly EMI",
            value=res["formatted_max_emi"],
            subtext=f"Requested: {format_currency(res['ratios']['requested_monthly_installment_estimate'])}/mo",
            delta_color="#10B981" if res["predicted_max_emi"] >= res["ratios"]["requested_monthly_installment_estimate"] else "#EF4444"
        )
    with p_col2:
        foir_val = round(res["ratios"]["foir"] * 100, 1)
        render_kpi_card(
            title="Fixed Obligation Ratio (FOIR)",
            value=f"{foir_val}%",
            subtext="Ceiling Guideline: < 50.0%",
            delta_color="#10B981" if foir_val <= 50.0 else "#EF4444"
        )

    # Probability distribution chart
    st.plotly_chart(plot_probability_distribution(res["class_probabilities"]), use_container_width=True)

    # Decision Drivers
    st.markdown("#### 🔍 Primary Decision Drivers & Explanatory Factors")
    for factor in res["driving_factors"]:
        render_factor_card(factor["type"], factor["title"], factor["detail"])

    # Action buttons
    st.markdown("<br>", unsafe_allow_html=True)
    b1, b2 = st.columns(2)
    with b1:
        if st.button("💾 Record Decision in Audit Trail", use_container_width=True):
            cust_id = loaded_cust["id"] if loaded_cust else None
            pred_id = PredictionRepository.save_prediction({
                "customer_id": cust_id,
                "emi_scenario": emi_scenario,
                "requested_amount": requested_amount,
                "requested_tenure": requested_tenure,
                "predicted_eligibility": res["predicted_eligibility"],
                "confidence_score": res["confidence"],
                "prob_eligible": res["class_probabilities"].get("Eligible", 0.0),
                "prob_high_risk": res["class_probabilities"].get("High_Risk", 0.0),
                "prob_not_eligible": res["class_probabilities"].get("Not_Eligible", 0.0),
                "predicted_max_emi": res["predicted_max_emi"],
                "disposable_income": res["ratios"]["disposable_income"],
                "foir": res["ratios"]["foir"],
                "model_version": "1.0.0",
                "notes": f"Underwriting assessment conducted via Workstation UI. Score={credit_score}."
            })
            st.success(f"✓ Recorded in audit trail as ID #{pred_id}")
    with b2:
        st.download_button(
            "📄 Export Assessment Dossier (JSON)",
            data=pd.Series({**user_payload, **res}).to_json(indent=2),
            file_name="underwriting_dossier.json",
            mime="application/json",
            use_container_width=True
        )
