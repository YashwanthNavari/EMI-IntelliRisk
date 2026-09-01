import pandas as pd
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from ..utils.logger import setup_logger
from ..utils.config import load_config

logger = setup_logger("SchemaValidation")

@dataclass
class SchemaValidationResult:
    is_valid: bool
    missing_columns: List[str]
    unexpected_columns: List[str]
    target_status: Dict[str, Any]
    data_type_mismatches: List[str]
    null_summary: Dict[str, int]
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

def validate_schema(df: pd.DataFrame) -> SchemaValidationResult:
    """Validate DataFrame against expected schema specifications."""
    config = load_config()
    expected_numerics = config["columns"]["numeric_features"]
    expected_categoricals = config["columns"]["categorical_features"]
    class_target = config["targets"]["classification"]["name"]
    reg_target = config["targets"]["regression"]["name"]

    required_cols = expected_numerics + expected_categoricals + [class_target, reg_target]
    actual_cols = list(df.columns)

    missing = [c for c in required_cols if c not in actual_cols]
    unexpected = [c for c in actual_cols if c not in required_cols]
    warnings = []

    # Check Targets
    target_status = {
        "classification_target": class_target,
        "classification_present": class_target in actual_cols,
        "classification_classes": df[class_target].unique().tolist() if class_target in actual_cols else [],
        "regression_target": reg_target,
        "regression_present": reg_target in actual_cols,
        "regression_nulls": int(df[reg_target].isnull().sum()) if reg_target in actual_cols else -1
    }

    # Data type inspection
    dtype_mismatches = []
    for col in expected_numerics:
        if col in actual_cols:
            # If numeric column has object dtype, flag as warning for cleaning pipeline
            if df[col].dtype == "object":
                warnings.append(f"Numeric column '{col}' has object dtype (requires string/dot cleaning).")

    null_summary = {col: int(df[col].isnull().sum()) for col in actual_cols if df[col].isnull().sum() > 0}

    is_valid = len(missing) == 0 and target_status["classification_present"] and target_status["regression_present"]

    logger.info(f"Schema validation completed: {'PASSED' if is_valid else 'FAILED'} (Missing={len(missing)}, Warnings={len(warnings)})")

    return SchemaValidationResult(
        is_valid=is_valid,
        missing_columns=missing,
        unexpected_columns=unexpected,
        target_status=target_status,
        data_type_mismatches=dtype_mismatches,
        null_summary=null_summary,
        warnings=warnings
    )
