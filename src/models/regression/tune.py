import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.pipeline import Pipeline
from .train import build_regression_pipeline
from ..evaluation import evaluate_regression_model, RegressionMetrics
from ...utils.logger import setup_logger
from ...utils.config import load_config

logger = setup_logger("RegressionTuning")

def tune_regression_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = "gradient_boosting",
    n_iter: int = 6,
    cv: int = 3
) -> Tuple[Pipeline, Dict[str, Any]]:
    """Tune hyperparameters using RandomizedSearchCV with KFold."""
    logger.info(f"Tuning {model_type} with {n_iter} iterations and {cv}-fold CV...")
    config = load_config()
    seed = config.get("project", {}).get("random_seed", 42)

    pipeline = build_regression_pipeline(model_type=model_type)

    param_distributions = {}
    if model_type == "random_forest":
        param_distributions = {
            "regressor__n_estimators": [100, 150, 200],
            "regressor__max_depth": [12, 16, 20, None],
            "regressor__min_samples_split": [2, 5, 10],
            "regressor__min_samples_leaf": [1, 2, 4]
        }
    elif model_type == "gradient_boosting":
        param_distributions = {
            "regressor__n_estimators": [100, 150, 200],
            "regressor__learning_rate": [0.03, 0.08, 0.15],
            "regressor__max_depth": [4, 6, 8]
        }
    elif model_type == "ridge_regression":
        param_distributions = {
            "regressor__alpha": [0.1, 1.0, 10.0, 100.0]
        }

    kf = KFold(n_splits=cv, shuffle=True, random_state=seed)

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=min(n_iter, len(param_distributions)),
        scoring="neg_root_mean_squared_error",
        cv=kf,
        random_state=seed,
        n_jobs=-1,
        verbose=1
    )

    if len(X_train) > 50000:
        sample_idx = X_train.sample(min(len(X_train), 25000), random_state=seed).index
        search.fit(X_train.loc[sample_idx], y_train.loc[sample_idx])
    else:
        search.fit(X_train, y_train)

    logger.info(f"Best params for {model_type}: {search.best_params_} (Best RMSE: {-search.best_score_:.2f})")
    return search.best_estimator_, search.best_params_
