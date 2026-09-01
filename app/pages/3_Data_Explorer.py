import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import sys

root_path = Path(__file__).resolve().parents[2]
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from app.services.analytics_service import load_cached_dataset_sample
from src.utils.helpers import format_currency

st.set_page_config(page_title="Data Explorer", page_icon="🔍", layout="wide")

css_file = root_path / "app" / "assets" / "styles.css"
if css_file.exists():
    with open(css_file, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1 class="header-title">🔍 Interactive Loan Dataset Explorer</h1>
    <div class="header-subtitle">High-Performance Analytical Slicing, Scenario Distribution, and Dimensional Cross-Filtering</div>
</div>
""", unsafe_allow_html=True)

with st.spinner("Loading analytical dataset partition..."):
    df = load_cached_dataset_sample(sample_size=40000)

# Sidebar multi-dimensional filters
st.sidebar.header("Filter Criteria")

scenario_filter = st.sidebar.multiselect(
    "EMI Scenario",
    options=list(df["emi_scenario"].dropna().unique()),
    default=list(df["emi_scenario"].dropna().unique())
)

eligibility_filter = st.sidebar.multiselect(
    "Eligibility Status",
    options=list(df["emi_eligibility"].dropna().unique()),
    default=list(df["emi_eligibility"].dropna().unique())
)

salary_range = st.sidebar.slider(
    "Monthly Salary Range (₹)",
    min_value=int(df["monthly_salary"].min()),
    max_value=int(min(300000, df["monthly_salary"].max())),
    value=(15000, 150000),
    step=5000
)

# Apply filters
filtered_df = df[
    df["emi_scenario"].isin(scenario_filter) &
    df["emi_eligibility"].isin(eligibility_filter) &
    (df["monthly_salary"] >= salary_range[0]) &
    (df["monthly_salary"] <= salary_range[1])
]

st.markdown(f"**Showing `{len(filtered_df):,}` matched records** (from {len(df):,} sampled rows)")

tab1, tab2, tab3 = st.tabs(["📊 Interactive Visualizations", "📋 Data Records Table", "📈 Statistical Summary"])

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Salary vs Requested Loan by Eligibility")
        fig_scatter = px.scatter(
            filtered_df.sample(min(len(filtered_df), 3000), random_state=42),
            x="monthly_salary",
            y="requested_amount",
            color="emi_eligibility",
            color_discrete_map={"Eligible": "#10B981", "High_Risk": "#F59E0B", "Not_Eligible": "#EF4444"},
            hover_data=["credit_score", "emi_scenario"],
            labels={"monthly_salary": "Monthly Salary (₹)", "requested_amount": "Requested Loan (₹)"}
        )
        fig_scatter.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#F3F4F6"))
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col2:
        st.subheader("Credit Score Distribution by Scenario")
        fig_box = px.box(
            filtered_df,
            x="emi_scenario",
            y="credit_score",
            color="emi_scenario",
            labels={"credit_score": "Credit Score", "emi_scenario": "Scenario"}
        )
        fig_box.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#F3F4F6"), showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

    st.subheader("Eligibility Breakdown across Loan Scenarios")
    ct = pd.crosstab(filtered_df["emi_scenario"], filtered_df["emi_eligibility"], normalize="index") * 100
    fig_bar = px.bar(
        ct.reset_index(),
        x="emi_scenario",
        y=["Eligible", "High_Risk", "Not_Eligible"],
        color_discrete_map={"Eligible": "#10B981", "High_Risk": "#F59E0B", "Not_Eligible": "#EF4444"},
        barmode="stack",
        labels={"value": "Proportion (%)", "emi_scenario": "Scenario"}
    )
    fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#F3F4F6"))
    st.plotly_chart(fig_bar, use_container_width=True)

with tab2:
    st.dataframe(filtered_df.head(200), use_container_width=True)
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Export Filtered Subset (CSV)",
        data=csv,
        file_name="filtered_emi_dataset.csv",
        mime="text/csv"
    )

with tab3:
    st.dataframe(filtered_df.describe().T.round(2), use_container_width=True)
