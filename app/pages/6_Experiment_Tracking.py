import streamlit as st
import pandas as pd
import json
from pathlib import Path
import sys

root_path = Path(__file__).resolve().parents[2]
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from app.components.ui_cards import render_kpi_card

st.set_page_config(page_title="MLflow Experiment Tracking", page_icon="🧪", layout="wide")

css_file = root_path / "app" / "assets" / "styles.css"
if css_file.exists():
    with open(css_file, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1 class="header-title">🧪 MLflow Experiment Tracking & Registry</h1>
    <div class="header-subtitle">MLOps Lifecycle Audit, Run Metrics, Hyperparameter Comparisons, and Model Versioning</div>
</div>
""", unsafe_allow_html=True)

summary_file = root_path / "models" / "mlflow_summary.json"
summary_data = {}
if summary_file.exists():
    with open(summary_file, "r") as f:
        summary_data = json.load(f)

k1, k2, k3 = st.columns(3)
with k1:
    render_kpi_card("Active MLflow Experiments", "2 Experiments", "Classification & Regression", delta_color="#6366F1")
with k2:
    total_runs = len(summary_data.get("classification_runs", [])) + len(summary_data.get("regression_runs", []))
    render_kpi_card("Total Tracked Model Runs", f"{total_runs} Completed Runs", "Logged with Metrics & Artifacts", delta_color="#10B981")
with k3:
    render_kpi_card("Registry Status", "Production Ready", f"Version 1.0.0 ({summary_data.get('exported_at', 'Active')})", delta_color="#38BDF8")

st.markdown("---")

tab1, tab2 = st.tabs(["🏷️ Experiment: EMI_Eligibility_Classification", "📏 Experiment: Maximum_EMI_Regression"])

with tab1:
    st.subheader("Classification Runs & Metrics")
    clf_runs = summary_data.get("classification_runs", [])
    if clf_runs:
        rows = []
        for r in clf_runs:
            m = r["metrics"]
            rows.append({
                "Run ID": r.get("run_id", "N/A"),
                "Model Architecture": r["model_name"].replace("_", " ").title(),
                "Accuracy": f"{m['accuracy']:.4f}",
                "Macro F1": f"{m['macro_f1']:.4f}",
                "Balanced Accuracy": f"{m['balanced_accuracy']:.4f}",
                "High-Risk Recall": f"{m['high_risk_recall']:.4f}",
                "Status": "FINISHED"
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("Loading experiment runs...")

with tab2:
    st.subheader("Regression Runs & Metrics")
    reg_runs = summary_data.get("regression_runs", [])
    if reg_runs:
        rows = []
        for r in reg_runs:
            m = r["metrics"]
            rows.append({
                "Run ID": r.get("run_id", "N/A"),
                "Model Architecture": r["model_name"].replace("_", " ").title(),
                "RMSE (₹)": f"₹{m['rmse']:,.2f}",
                "MAE (₹)": f"₹{m['mae']:,.2f}",
                "R² Score": f"{m['r2']:.4f}",
                "MAPE": f"{m['mape']:.2f}%",
                "Status": "FINISHED"
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("Loading experiment runs...")

st.markdown("---")
st.subheader("📦 Model Artifacts & Production Registry")
st.code("""
Models Directory:
├── models/
│   ├── classification/
│   │   ├── best_classification_model.joblib
│   │   └── best_classification_metadata.json
│   ├── regression/
│   │   ├── best_regression_model.joblib
│   │   └── best_regression_metadata.json
│   └── mlflow_summary.json
""", language="text")
