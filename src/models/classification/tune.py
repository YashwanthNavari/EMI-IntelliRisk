import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from .train import build_classification_pipeline
from ..evaluation import evaluate_classification_model, ClassificationMetrics
from ...utils.logger import setup_logger
from ...utils.config import load_config

logger = setup_logger("ClassificationTuning")

def tune_classification_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = "gradient_boosting",
    n_iter: int = 6,
    cv: int = 3
) -> Tuple[Pipeline, Dict[str, Any]]:
    """Tune hyperparameters using RandomizedSearchCV with StratifiedKFold."""
    logger.info(f"Tuning {model_type} with {n_iter} iterations and {cv}-fold Stratified CV...")
    config = load_config()
    seed = config.get("project", {}).get("random_seed", 42)

    pipeline = build_classification_pipeline(model_type=model_type)

    param_distributions = {}
    if model_type == "random_forest":
        param_distributions = {
            "classifier__n_estimators": [100, 150, 200],
            "classifier__max_depth": [10, 14, 18, None],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__min_samples_leaf": [1, 2, 4]
        }
    elif model_type == "gradient_boosting":
        param_distributions = {
            "classifier__n_estimators": [100, 150, 200],
            "classifier__learning_rate": [0.03, 0.08, 0.15],
            "classifier__max_depth": [4, 6, 8]
        }
    elif model_type == "logistic_regression":
        param_distributions = {
            "classifier__C": [0.01, 0.1, 1.0, 10.0]
        }

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=min(n_iter, len(param_distributions)),
        scoring="f1_macro",
        cv=skf,
        random_state=seed,
        n_jobs=-1,
        verbose=1
    )

    # Use a stratified subsample for tuning speed if training set is massive
    if len(X_train) > 50000:
        sample_idx = X_train.groupby(y_train, group_keys=False).apply(lambda x: x.sample(min(len(x), 20000), random_state=seed)).index
        search.fit(X_train.loc[sample_idx], y_train.loc[sample_idx])
    else:
        search.fit(X_train, y_train)

    logger.info(f"Best params for {model_type}: {search.best_params_} (Best Macro-F1: {search.best_score_:.4f})")
    return search.best_estimator_, search.best_params_
