import pandas as pd
import numpy as np
from typing import Dict, Any, List
from dataclasses import dataclass, asdict
from ..utils.logger import setup_logger

logger = setup_logger("DataQuality")

@dataclass
class DataQualityReport:
    total_rows: int
    total_columns: int
    duplicate_rows: int
    completeness_score: float
    missing_by_column: Dict[str, Dict[str, Any]]
    categorical_inconsistencies: Dict[str, List[str]]
    numeric_outlier_summary: Dict[str, Dict[str, Any]]
    data_type_issues: List[str]
    quality_grade: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

def perform_data_quality_audit(df: pd.DataFrame) -> DataQualityReport:
    """Execute complete multi-dimensional data quality assessment."""
    total_rows, total_cols = df.shape
    total_cells = total_rows * total_cols
    total_nulls = df.isnull().sum().sum()
    completeness = ((total_cells - total_nulls) / total_cells) * 100.0
    duplicates = int(df.duplicated().sum())

    # Missing breakdown
    missing_breakdown = {}
    for col in df.columns:
        null_cnt = int(df[col].isnull().sum())
        if null_cnt > 0:
            missing_breakdown[col] = {
                "count": null_cnt,
                "percentage": round((null_cnt / total_rows) * 100, 2),
                "dtype": str(df[col].dtype)
            }

    # Categorical inconsistencies (e.g. gender having 8 labels)
    cat_inconsistencies = {}
    for col in df.select_dtypes(include=["object"]).columns:
        unique_vals = [str(x) for x in df[col].dropna().unique()]
        # Check case folding duplicates
        lowered = [x.strip().lower() for x in unique_vals]
        if len(unique_vals) != len(set(lowered)):
            cat_inconsistencies[col] = unique_vals

    # Outlier / Range checks
    numeric_issues = {}
    if "credit_score" in df.columns:
        cs = pd.to_numeric(df["credit_score"], errors="coerce")
        under_300 = int((cs < 300).sum())
        over_900 = int((cs > 900).sum())
        numeric_issues["credit_score"] = {
            "under_300_anomalies": under_300,
            "over_900_anomalies": over_900,
            "valid_range_pct": round(((len(cs.dropna()) - under_300 - over_900) / max(1, len(cs.dropna()))) * 100, 2)
        }

    # Data type format issues (multi-dot string columns)
    type_issues = []
    for col in ["age", "monthly_salary", "bank_balance"]:
        if col in df.columns and df[col].dtype == "object":
            dirty_count = df[col].astype(str).str.contains(r"\.0\.0", regex=True).sum()
            if dirty_count > 0:
                type_issues.append(f"Column '{col}' contains {dirty_count} records with duplicate dot formatting (e.g. '.0.0').")

    # Quality Grade assignment
    if completeness > 98.0 and duplicates == 0:
        grade = "A (Production Quality with Minor Formatting Normalization Required)"
    elif completeness > 90.0:
        grade = "B (Good Quality, Imputation & Cleaning Needed)"
    else:
        grade = "C (Significant Cleaning Needed)"

    logger.info(f"Data Quality Audit Complete: Grade={grade}, Completeness={completeness:.2f}%, Duplicates={duplicates}")

    return DataQualityReport(
        total_rows=total_rows,
        total_columns=total_cols,
        duplicate_rows=duplicates,
        completeness_score=round(completeness, 2),
        missing_by_column=missing_breakdown,
        categorical_inconsistencies=cat_inconsistencies,
        numeric_outlier_summary=numeric_issues,
        data_type_issues=type_issues,
        quality_grade=grade
    )
