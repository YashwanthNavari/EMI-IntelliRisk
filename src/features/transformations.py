import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from .financial_features import compute_financial_features
from .risk_features import compute_risk_features

class FinancialFeatureTransformer(BaseEstimator, TransformerMixin):
    """Scikit-Learn compatible transformer for dynamic financial and risk feature generation."""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy() if not isinstance(X, pd.DataFrame) else X.copy()
        X_df = compute_financial_features(X_df)
        X_df = compute_risk_features(X_df)
        return X_df
