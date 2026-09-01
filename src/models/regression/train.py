import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from ...features.feature_pipeline import create_preprocessor_pipeline
from ..evaluation import evaluate_regression_model, RegressionMetrics
from ...utils.logger import setup_logger
from ...utils.config import load_config

# Optional XGBoost / LightGBM import
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMRegressor
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

logger = setup_logger("RegressionTrainer")

def build_regression_pipeline(model_type: str = "random_forest", **kwargs) -> Pipeline:
    """Construct an end-to-end Pipeline with preprocessing and regressor."""
    config = load_config()
    seed = config.get("project", {}).get("random_seed", 42)

    if model_type == "ridge_regression":
        regressor = Ridge(
            alpha=kwargs.get("alpha", 10.0),
            random_state=seed
        )
    elif model_type == "random_forest":
        regressor = RandomForestRegressor(
            n_estimators=kwargs.get("n_estimators", 120),
            max_depth=kwargs.get("max_depth", 16),
            min_samples_split=kwargs.get("min_samples_split", 5),
            min_samples_leaf=kwargs.get("min_samples_leaf", 2),
            n_jobs=-1,
            random_state=seed
        )
    elif model_type == "gradient_boosting":
        if HAS_LIGHTGBM:
            regressor = LGBMRegressor(
                n_estimators=kwargs.get("n_estimators", 150),
                learning_rate=kwargs.get("learning_rate", 0.08),
                max_depth=kwargs.get("max_depth", 6),
                n_jobs=-1,
                random_state=seed,
                verbose=-1
            )
        elif HAS_XGBOOST:
            regressor = XGBRegressor(
                n_estimators=kwargs.get("n_estimators", 150),
                learning_rate=kwargs.get("learning_rate", 0.08),
                max_depth=kwargs.get("max_depth", 6),
                random_state=seed,
                n_jobs=-1
            )
        else:
            regressor = HistGradientBoostingRegressor(
                max_iter=kwargs.get("n_estimators", 150),
                learning_rate=kwargs.get("learning_rate", 0.08),
                max_depth=kwargs.get("max_depth", 6),
                random_state=seed
            )
    else:
        raise ValueError(f"Unknown regression model type: {model_type}")

    preprocessor = create_preprocessor_pipeline(scale_numerics=(model_type == "ridge_regression"))

    pipeline = Pipeline(steps=[
        ("feature_pipeline", preprocessor),
        ("regressor", regressor)
    ])

    return pipeline

def train_regression_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series
) -> Dict[str, Tuple[Pipeline, RegressionMetrics]]:
    """Train multiple regression models, evaluate on validation set, and return pipelines & metrics."""
    logger.info("Starting regression model training suite...")
    models_to_train = ["ridge_regression", "random_forest", "gradient_boosting"]
    results = {}

    for model_name in models_to_train:
        logger.info(f"--- Training {model_name} ---")
        pipeline = build_regression_pipeline(model_type=model_name)
        pipeline.fit(X_train, y_train)

        # Validation prediction
        y_val_pred = pipeline.predict(X_val)
        # Ensure predictions are non-negative with minimum threshold
        y_val_pred = np.maximum(500.0, y_val_pred)

        metrics = evaluate_regression_model(
            model_name=model_name,
            y_true=y_val,
            y_pred=y_val_pred
        )

        results[model_name] = (pipeline, metrics)

    logger.info("Regression model training suite completed.")
    return results
