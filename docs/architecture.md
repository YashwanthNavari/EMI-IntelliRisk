# System Architecture — EMIPredict AI

```
                                  [ RAW DATASET ]
                           (emi_prediction_dataset.csv)
                                        │
                                        ▼
                            [ INGESTION & VALIDATION ]
                             (schema_validation.py)
                                        │
                                        ▼
                             [ DATA CLEANING PIPELINE ]
                               (cleaning.py / regex)
                                        │
                                        ▼
                         [ FINANCIAL FEATURE ENGINEERING ]
                         (financial_features / risk_features)
                                        │
                                        ▼
                            [ TARGET LEAKAGE AUDIT ]
                                        │
                                        ▼
                       [ STRATIFIED TRAIN / VAL / TEST ]
                                 (70 / 15 / 15)
                                        │
                    ┌───────────────────┴───────────────────┐
                    ▼                                       ▼
         [ CLASSIFICATION PIPELINE ]             [ REGRESSION PIPELINE ]
         • Logistic Regression                   • Ridge Regression
         • Random Forest Classifier              • Random Forest Regressor
         • Gradient Boosting (LGBM/XGB)          • Gradient Boosting (LGBM/XGB)
                    │                                       │
                    └───────────────────┬───────────────────┘
                                        ▼
                           [ MLFLOW EXPERIMENT TRACKING ]
                          (Metrics, Params, Artifacts, DB)
                                        │
                                        ▼
                             [ BEST MODEL SELECTION ]
                                        │
                                        ▼
                             [ MODEL PERSISTENCE ]
                          (models/classification, models/regression)
                                        │
                                        ▼
                         [ STREAMLIT WEB APPLICATION ]
    ┌───────────────────────────┼───────────────────────────┐
    ▼                           ▼                           ▼
[ REAL-TIME RISK ]    [ FINANCIAL HEALTH ]         [ DATABASE CRUD ]
(Dual Prediction)     (Stress Testing & FOIR)      (SQLite Profile Ops)
```
