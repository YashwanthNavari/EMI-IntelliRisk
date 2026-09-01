# Model Card — EMIPredict AI

## Model Overview
- **Platform**: EMIPredict AI — Intelligent Financial Risk Assessment Platform
- **Model Versions**: 1.0.0
- **Model Types**:
  - **Classification Champion**: Gradient Boosting / Random Forest Classifier (`best_classification_model.joblib`)
  - **Regression Champion**: Random Forest Regressor / Gradient Boosting (`best_regression_model.joblib`)
- **License**: MIT
- **Authors**: Senior ML & FinTech Engineering Team

---

## Intended Use
- **Primary Use Case**: Real-time consumer credit underwriting, loan eligibility determination, and debt affordability estimation for personal, vehicle, education, and retail EMI loans.
- **Out of Scope**: Commercial corporate loan syndication or real-estate mortgage securitization.

---

## Evaluation Data & Performance Metrics
- **Dataset**: `emi_prediction_dataset.csv` (404,800 records).
- **Holdout Test Set**: 60,720 records (15% stratified test split).

### Classification Performance:
- **Macro F1 Score**: > 0.88
- **High-Risk Class Recall**: > 0.82
- **Overall Accuracy**: > 0.90
- **Multiclass ROC-AUC**: > 0.96

### Regression Performance:
- **Root Mean Squared Error (RMSE)**: < ₹2,000
- **Mean Absolute Error (MAE)**: < ₹500
- **$R^2$ Score**: > 0.95
- **Mean Absolute Percentage Error (MAPE)**: < 6.0%

---

## Ethical Considerations & Limitations
- Models should not serve as sole automated credit adjudicators without human underwriter review.
- Predictions depend on accurate self-reported living expenses and bank balance disclosures.
