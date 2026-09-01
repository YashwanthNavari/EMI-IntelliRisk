import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from ...features.feature_pipeline import create_preprocessor_pipeline
from ..evaluation import evaluate_classification_model, ClassificationMetrics
from ...utils.logger import setup_logger
from ...utils.config import load_config

# Optional XGBoost / LightGBM import
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

logger = setup_logger("ClassificationTrainer")

def build_classification_pipeline(model_type: str = "random_forest", **kwargs) -> Pipeline:
    """Construct an end-to-end Pipeline with preprocessing and classifier."""
    config = load_config()
    seed = config.get("project", {}).get("random_seed", 42)

    if model_type == "logistic_regression":
        classifier = LogisticRegression(
            C=kwargs.get("C", 1.0),
            max_iter=kwargs.get("max_iter", 1000),
            class_weight="balanced",
            solver="lbfgs",
            random_state=seed
        )
    elif model_type == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=kwargs.get("n_estimators", 120),
            max_depth=kwargs.get("max_depth", 14),
            min_samples_split=kwargs.get("min_samples_split", 5),
            min_samples_leaf=kwargs.get("min_samples_leaf", 2),
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed
        )
    elif model_type == "gradient_boosting":
        if HAS_LIGHTGBM:
            classifier = LGBMClassifier(
                n_estimators=kwargs.get("n_estimators", 150),
                learning_rate=kwargs.get("learning_rate", 0.08),
                max_depth=kwargs.get("max_depth", 6),
                class_weight="balanced",
                n_jobs=-1,
                random_state=seed,
                verbose=-1
            )
        elif HAS_XGBOOST:
            classifier = XGBClassifier(
                n_estimators=kwargs.get("n_estimators", 150),
                learning_rate=kwargs.get("learning_rate", 0.08),
                max_depth=kwargs.get("max_depth", 6),
                random_state=seed,
                eval_metric="mlogloss",
                n_jobs=-1
            )
        else:
            classifier = HistGradientBoostingClassifier(
                max_iter=kwargs.get("n_estimators", 150),
                learning_rate=kwargs.get("learning_rate", 0.08),
                max_depth=kwargs.get("max_depth", 6),
                class_weight="balanced",
                random_state=seed
            )
    else:
        raise ValueError(f"Unknown classification model type: {model_type}")

    preprocessor = create_preprocessor_pipeline(scale_numerics=(model_type == "logistic_regression"))

    pipeline = Pipeline(steps=[
        ("feature_pipeline", preprocessor),
        ("classifier", classifier)
    ])

    return pipeline

def train_classification_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series
) -> Dict[str, Tuple[Pipeline, ClassificationMetrics]]:
    """Train multiple classification models, evaluate on validation set, and return pipelines & metrics."""
    logger.info("Starting classification model training suite...")
    models_to_train = ["logistic_regression", "random_forest", "gradient_boosting"]
    results = {}

    classes = sorted(list(np.unique(y_train)))

    for model_name in models_to_train:
        logger.info(f"--- Training {model_name} ---")
        pipeline = build_classification_pipeline(model_type=model_name)
        pipeline.fit(X_train, y_train)

        # Validation prediction
        y_val_pred = pipeline.predict(X_val)
        y_val_proba = pipeline.predict_proba(X_val) if hasattr(pipeline, "predict_proba") else None

        metrics = evaluate_classification_model(
            model_name=model_name,
            y_true=y_val,
            y_pred=y_val_pred,
            y_proba=y_val_proba,
            classes=classes
        )

        results[model_name] = (pipeline, metrics)

    logger.info("Classification model training suite completed.")
    return results
