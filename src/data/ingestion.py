import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from ..utils.logger import setup_logger
from ..utils.config import get_project_root, load_config

logger = setup_logger("DataIngestion")

def find_dataset_file() -> Path:
    """Locate the dataset file in standard locations."""
    root = get_project_root()
    candidates = [
        root / "emi_prediction_dataset.csv",
        root / "data" / "raw" / "emi_prediction_dataset.csv",
        Path("emi_prediction_dataset.csv").resolve(),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("Could not locate 'emi_prediction_dataset.csv' in workspace root or data/raw/.")

def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric types and convert categorical columns for memory optimization."""
    initial_memory = df.memory_usage(deep=True).sum() / (1024 * 1024)

    for col in df.columns:
        col_type = df[col].dtype
        if col_type == "object" and df[col].nunique() < 30:
            df[col] = df[col].astype("category")
        elif np.issubdtype(col_type, np.integer):
            df[col] = pd.to_numeric(df[col], downcast="integer")
        elif np.issubdtype(col_type, np.floating):
            df[col] = pd.to_numeric(df[col], downcast="float")

    final_memory = df.memory_usage(deep=True).sum() / (1024 * 1024)
    logger.info(f"Memory reduced from {initial_memory:.2f} MB to {final_memory:.2f} MB ({((initial_memory - final_memory) / initial_memory) * 100:.1f}% savings)")
    return df

def load_raw_dataset(file_path: Optional[str] = None, optimize_memory: bool = False, nrows: Optional[int] = None) -> pd.DataFrame:
    """Ingest raw CSV dataset with error handling and logging."""
    path = Path(file_path) if file_path else find_dataset_file()
    logger.info(f"Ingesting raw dataset from: {path} (nrows={nrows or 'ALL'})")

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    # Ingest using low_memory=False to prevent mixed-type chunking warnings
    df = pd.read_csv(str(path), low_memory=False, nrows=nrows)
    logger.info(f"Successfully loaded dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")

    if optimize_memory:
        df = optimize_dtypes(df)

    return df
