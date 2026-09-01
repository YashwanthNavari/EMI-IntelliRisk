import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split
from ..utils.logger import setup_logger
from ..utils.config import load_config

logger = setup_logger("DataSplitting")

def create_stratified_splits(
    df: pd.DataFrame,
    stratify_col: str = "emi_eligibility",
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create reproducible stratified train, validation, and test splits."""
    logger.info(f"Splitting dataset of {len(df):,} records (test_size={test_size}, val_size={val_size}, stratify_col='{stratify_col}')")

    stratify_series = df[stratify_col] if stratify_col in df.columns else None

    # Step 1: Split train+val vs test
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_series
    )

    # Step 2: Split train vs val
    adj_val_size = val_size / (1.0 - test_size)
    stratify_train_val = train_val_df[stratify_col] if stratify_col in train_val_df.columns else None

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=adj_val_size,
        random_state=random_state,
        stratify=stratify_train_val
    )

    logger.info(f"Split completed: Train={len(train_df):,} ({len(train_df)/len(df)*100:.1f}%), Val={len(val_df):,} ({len(val_df)/len(df)*100:.1f}%), Test={len(test_df):,} ({len(test_df)/len(df)*100:.1f}%)")

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)
