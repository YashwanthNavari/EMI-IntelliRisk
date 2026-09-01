import pytest
import pandas as pd
import numpy as np
from src.data.ingestion import find_dataset_file, load_raw_dataset
from src.data.schema_validation import validate_schema
from src.data.data_quality import perform_data_quality_audit
from src.data.cleaning import clean_dataset, clean_dirty_numeric_series, normalize_gender

def test_find_dataset():
    path = find_dataset_file()
    assert path.exists()
    assert path.name.endswith(".csv")

def test_clean_dirty_numeric_series():
    s = pd.Series(["58.0.0", "38.0.0.0", "12500.5", np.nan, "100"])
    cleaned = clean_dirty_numeric_series(s)
    assert cleaned.iloc[0] == 58.0
    assert cleaned.iloc[1] == 38.0
    assert cleaned.iloc[2] == 12500.5
    assert np.isnan(cleaned.iloc[3])
    assert cleaned.iloc[4] == 100.0

def test_normalize_gender():
    genders = pd.Series(["Male", "female", "M", "FEMALE", "f", "MALE"])
    normalized = normalize_gender(genders)
    assert set(normalized.unique()) == {"Male", "Female"}

def test_schema_validation_on_sample():
    df_sample = load_raw_dataset(nrows=50)
    res = validate_schema(df_sample)
    assert res.target_status["classification_present"] is True
    assert res.target_status["regression_present"] is True
    assert len(res.missing_columns) == 0

def test_data_quality_audit_on_sample():
    df_sample = load_raw_dataset(nrows=100)
    dq = perform_data_quality_audit(df_sample)
    assert dq.total_rows == 100
    assert dq.total_columns == 27
    assert dq.completeness_score > 90.0
