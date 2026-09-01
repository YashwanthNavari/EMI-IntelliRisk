import numpy as np
import pandas as pd
from typing import Dict, List, Any
from sklearn.metrics import accuracy_score, f1_score, recall_score, mean_absolute_error, mean_squared_error, r2_score
from ..utils.logger import setup_logger

logger = setup_logger("FairnessAudit")

def run_demographic_fairness_audit(
    df_test: pd.DataFrame,
    y_true_clf: pd.Series,
    y_pred_clf: np.ndarray,
    y_true_reg: pd.Series,
    y_pred_reg: np.ndarray
) -> Dict[str, List[Dict[str, Any]]]:
    """Audit subgroup predictive parity and error disparities across protected & demographic dimensions."""
    logger.info("Executing demographic fairness & subgroup performance audit...")
    df_eval = df_test.copy()
    df_eval["y_true_clf"] = np.array(y_true_clf)
    df_eval["y_pred_clf"] = np.array(y_pred_clf)
    df_eval["y_true_reg"] = np.array(y_true_reg, dtype=float)
    df_eval["y_pred_reg"] = np.array(y_pred_reg, dtype=float)

    # Demographic bucket creation
    if "age" in df_eval.columns:
        df_eval["age_group"] = pd.cut(
            df_eval["age"],
            bins=[0, 30, 50, 100],
            labels=["Young Adults (<30)", "Mid-Career (30-50)", "Senior (50+)"]
        )

    if "monthly_salary" in df_eval.columns:
        df_eval["income_tier"] = pd.cut(
            df_eval["monthly_salary"],
            bins=[0, 35000, 75000, 10000000],
            labels=["Low Income (<₹35K)", "Middle Income (₹35K-₹75K)", "High Income (>₹75K)"]
        )

    audit_dimensions = {
        "Gender": "gender",
        "Age Group": "age_group",
        "Employment Type": "employment_type",
        "Income Tier": "income_tier"
    }

    subgroup_reports = {}

    for dim_title, dim_col in audit_dimensions.items():
        if dim_col not in df_eval.columns:
            continue

        group_rows = []
        for group_name, group_data in df_eval.groupby(dim_col, observed=True):
            if len(group_data) == 0:
                continue

            yt_c = group_data["y_true_clf"]
            yp_c = group_data["y_pred_clf"]
            yt_r = group_data["y_true_reg"]
            yp_r = group_data["y_pred_reg"]

            acc = accuracy_score(yt_c, yp_c)
            macro_f1 = f1_score(yt_c, yp_c, average="macro", zero_division=0)
            rec_per_class = recall_score(yt_c, yp_c, average=None, labels=["Eligible", "High_Risk", "Not_Eligible"], zero_division=0)
            hr_recall = rec_per_class[1]

            mae = mean_absolute_error(yt_r, yp_r)
            rmse = np.sqrt(mean_squared_error(yt_r, yp_r))
            r2 = r2_score(yt_r, yp_r)

            group_rows.append({
                "subgroup": str(group_name),
                "sample_count": int(len(group_data)),
                "sample_pct": round((len(group_data) / len(df_eval)) * 100, 1),
                "accuracy": round(acc, 4),
                "macro_f1": round(macro_f1, 4),
                "high_risk_recall": round(hr_recall, 4),
                "regression_mae": round(mae, 2),
                "regression_rmse": round(rmse, 2),
                "regression_r2": round(r2, 4)
            })

        subgroup_reports[dim_title] = group_rows

    logger.info("Fairness and subgroup audit complete.")
    return subgroup_reports
