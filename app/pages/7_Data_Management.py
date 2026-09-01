import streamlit as st
import pandas as pd
from pathlib import Path
import sys

root_path = Path(__file__).resolve().parents[2]
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from database.repository import CustomerRepository, PredictionRepository
from src.utils.helpers import format_currency

st.set_page_config(page_title="Data Management (CRUD)", page_icon="🗄️", layout="wide")

css_file = root_path / "app" / "assets" / "styles.css"
if css_file.exists():
    with open(css_file, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1 class="header-title">🗄️ Customer Data Management & Audit Trail</h1>
    <div class="header-subtitle">Full Database CRUD Operations, Applicant Profiling, and Historical Underwriting Audit Logs</div>
</div>
""", unsafe_allow_html=True)

tab_view, tab_create, tab_edit, tab_history = st.tabs([
    "📋 View Customer Profiles",
    "➕ Register New Applicant",
    "✏️ Edit / Delete Profile",
    "📜 Prediction Audit History"
])

with tab_view:
    st.subheader("Registered Customer Financial Profiles")
    customers = CustomerRepository.list_customers(limit=100)
    if customers:
        df_cust = pd.DataFrame(customers)
        display_cols = ["id", "full_name", "email", "monthly_salary", "employment_type", "house_type", "credit_score", "created_at"]
        st.dataframe(df_cust[[c for c in display_cols if c in df_cust.columns]], use_container_width=True)
    else:
        st.info("No customer profiles registered yet.")

with tab_create:
    st.subheader("Create New Customer Profile")
    with st.form("create_customer_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            name = st.text_input("Full Name *", value="")
            email = st.text_input("Email Address", value="")
            phone = st.text_input("Phone Number", value="")
            age = st.number_input("Age", 18, 80, 30)
            gender = st.selectbox("Gender", ["Male", "Female"])
        with c2:
            salary = st.number_input("Monthly Salary (₹) *", min_value=5000.0, max_value=1000000.0, value=60000.0, step=2000.0)
            emp_type = st.selectbox("Employment Type", ["Private", "Government", "Self-employed"])
            yoe = st.number_input("Years of Employment", 0.0, 45.0, 3.5, step=0.5)
            comp_type = st.selectbox("Company Type", ["Large Indian", "MNC", "Mid-size", "Startup", "Small"])
            house = st.selectbox("House Type", ["Rented", "Own", "Family"])
        with c3:
            rent = st.number_input("Monthly Rent (₹)", 0.0, 200000.0, 15000.0, step=1000.0)
            cur_emi = st.number_input("Current Monthly EMI (₹)", 0.0, 200000.0, 0.0, step=1000.0)
            credit = st.slider("Credit Score", 300, 900, 750)
            bank = st.number_input("Bank Balance (₹)", 0.0, 5000000.0, 200000.0, step=10000.0)
            emerg = st.number_input("Emergency Reserve (₹)", 0.0, 2000000.0, 60000.0, step=5000.0)

        create_btn = st.form_submit_button("➕ Register Applicant Profile", use_container_width=True)
        if create_btn:
            if not name.strip():
                st.error("Please provide applicant Full Name.")
            else:
                new_id = CustomerRepository.create_customer({
                    "full_name": name,
                    "email": email,
                    "phone": phone,
                    "age": age,
                    "gender": gender,
                    "monthly_salary": salary,
                    "employment_type": emp_type,
                    "years_of_employment": yoe,
                    "company_type": comp_type,
                    "house_type": house,
                    "monthly_rent": rent,
                    "family_size": 3,
                    "dependents": 1,
                    "school_fees": 0.0,
                    "college_fees": 0.0,
                    "travel_expenses": 4000.0,
                    "groceries_utilities": 12000.0,
                    "other_monthly_expenses": 5000.0,
                    "existing_loans": "Yes" if cur_emi > 0 else "No",
                    "current_emi_amount": cur_emi,
                    "credit_score": float(credit),
                    "bank_balance": bank,
                    "emergency_fund": emerg
                })
                st.success(f"✓ Profile successfully created with ID #{new_id}!")
                st.rerun()

with tab_edit:
    st.subheader("Edit or Delete Customer Profile")
    customers = CustomerRepository.list_customers(limit=100)
    if customers:
        cust_map = {f"#{c['id']}: {c['full_name']}": c for c in customers}
        selected_key = st.selectbox("Select Customer to Edit:", list(cust_map.keys()))
        target_cust = cust_map[selected_key]

        with st.form("edit_customer_form"):
            e1, e2, e3 = st.columns(3)
            with e1:
                e_name = st.text_input("Full Name", value=target_cust["full_name"])
                e_email = st.text_input("Email", value=target_cust["email"] or "")
                e_salary = st.number_input("Monthly Salary (₹)", value=float(target_cust["monthly_salary"] or 0), step=2000.0)
            with e2:
                e_rent = st.number_input("Monthly Rent (₹)", value=float(target_cust["monthly_rent"] or 0), step=1000.0)
                e_emi = st.number_input("Current EMI (₹)", value=float(target_cust["current_emi_amount"] or 0), step=1000.0)
                e_credit = st.slider("Credit Score", 300, 900, int(target_cust["credit_score"] or 700))
            with e3:
                e_bank = st.number_input("Bank Balance (₹)", value=float(target_cust["bank_balance"] or 0), step=10000.0)
                e_emerg = st.number_input("Emergency Fund (₹)", value=float(target_cust["emergency_fund"] or 0), step=5000.0)

            save_edit_btn = st.form_submit_button("💾 Update Customer Profile", use_container_width=True)
            if save_edit_btn:
                CustomerRepository.update_customer(target_cust["id"], {
                    "full_name": e_name,
                    "email": e_email,
                    "monthly_salary": e_salary,
                    "monthly_rent": e_rent,
                    "current_emi_amount": e_emi,
                    "credit_score": float(e_credit),
                    "bank_balance": e_bank,
                    "emergency_fund": e_emerg
                })
                st.success(f"✓ Profile #{target_cust['id']} updated successfully!")
                st.rerun()

        st.markdown("---")
        st.markdown("#### Destructive Actions")
        if st.button(f"🗑️ Delete Profile #{target_cust['id']} ({target_cust['full_name']})", type="secondary"):
            CustomerRepository.delete_customer(target_cust["id"])
            st.warning(f"Profile #{target_cust['id']} deleted.")
            st.rerun()

with tab_history:
    st.subheader("Historical Loan Underwriting Predictions")
    history = PredictionRepository.get_prediction_history(limit=200)
    if history:
        df_hist = pd.DataFrame(history)
        cols_show = ["id", "customer_name", "emi_scenario", "requested_amount", "predicted_eligibility", "confidence_score", "predicted_max_emi", "created_at"]
        st.dataframe(df_hist[[c for c in cols_show if c in df_hist.columns]], use_container_width=True)

        csv_hist = df_hist.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Export Full Prediction History (CSV)",
            data=csv_hist,
            file_name="prediction_history_audit_log.csv",
            mime="text/csv"
        )
    else:
        st.info("No predictions recorded yet.")
