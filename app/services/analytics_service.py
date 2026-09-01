import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

root_path = Path(__file__).resolve().parents[2]
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from src.data.ingestion import find_dataset_file

@st.cache_data(ttl=3600)
def load_cached_dataset_sample(sample_size: int = 50000) -> pd.DataFrame:
    """Load and cache a high-speed analytical subset of the 404,800 records."""
    file_path = find_dataset_file()
    df = pd.read_csv(file_path, low_memory=False)

    # Basic quick clean for display
    if "monthly_salary" in df.columns:
        df["monthly_salary"] = pd.to_numeric(
            df["monthly_salary"].astype(str).str.replace(r"(\.0)+$", "", regex=True),
            errors="coerce"
        )
    if "age" in df.columns:
        df["age"] = pd.to_numeric(
            df["age"].astype(str).str.replace(r"(\.0)+$", "", regex=True),
            errors="coerce"
        )
    if "bank_balance" in df.columns:
        df["bank_balance"] = pd.to_numeric(
            df["bank_balance"].astype(str).str.replace(r"(\.0)+$", "", regex=True),
            errors="coerce"
        )
    if "gender" in df.columns:
        df["gender"] = df["gender"].astype(str).str.strip().str.capitalize()
        df["gender"] = df["gender"].replace({"M": "Male", "F": "Female"})

    if len(df) > sample_size:
        return df.sample(sample_size, random_state=42).reset_index(drop=True)
    return df

@st.cache_data
def get_dataset_summary_stats():
    """Compute high-level summary statistics across the entire dataset."""
    file_path = find_dataset_file()
    df = pd.read_csv(file_path, low_memory=False)

    total_records = len(df)
    total_features = df.shape[1]

    elig_dist = df["emi_eligibility"].value_counts(normalize=True).to_dict() if "emi_eligibility" in df.columns else {}
    avg_max_emi = float(df["max_monthly_emi"].mean()) if "max_monthly_emi" in df.columns else 0.0

    return {
        "total_records": total_records,
        "total_features": total_features,
        "eligibility_distribution": elig_dist,
        "average_max_emi": avg_max_emi
    }
