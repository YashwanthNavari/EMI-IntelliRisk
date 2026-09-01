# EMIPredict AI — Detailed Methodology & Academic Validation

## 1. Problem Formulation
EMIPredict AI models two concurrent financial risk problems from applicant disclosures:
1. **Multiclass Underwriting Classification**: Predicts `emi_eligibility` $\in \{\text{Eligible, High\_Risk, Not\_Eligible}\}$.
2. **Continuous Affordability Regression**: Predicts `max_monthly_emi` $\in [\text{₹}500, \text{₹}91,040]$.

---

## 2. Experimental Splitting & Holdout Sanctity
To avoid data snooping and model selection bias:
```
Raw Dataset (404,800 records)
           │
           ▼
Stratified Partitioning (emi_eligibility)
  ├── 70% Training (283,359 records) ──► K-Fold CV & Hyperparameter Optimization
  ├── 15% Validation (60,721 records) ──► Champion Model Selection
  └── 15% Untouched Test (60,720 records) ──► Final Reported Metrics & Bootstrap CIs
```
The final holdout test set remained completely isolated from feature selection and hyperparameter tuning.

---

## 3. Financial Domain Feature Derivation
1. **Total Living Expenses**: $\text{Rent} + \text{School} + \text{College} + \text{Travel} + \text{Groceries} + \text{Other}$
2. **True Disposable Surplus**: $\text{Salary} - \text{Total Expenses} - \text{Current EMI}$
3. **Fixed Obligation to Income Ratio (FOIR)**: $\frac{\text{Current EMI} + \text{Requested EMI Estimate}}{\text{Monthly Salary}}$
4. **Expense to Income Ratio (ETI)**: $\frac{\text{Total Expenses}}{\text{Monthly Salary}}$
5. **Emergency Fund Buffer**: $\frac{\text{Emergency Fund}}{\text{Total Expenses}}$ (Months of runway)

---

## 4. Evaluation Metrics & Statistical Uncertainty

### 4.1 Classification Metrics & Class Imbalance
Because `Not_Eligible` represents 77.29% of samples, overall accuracy is secondary to:
- **Macro $F_1$-Score**: Unweighted mean $F_1$ across all 3 classes.
- **High-Risk Class Recall**: Sensitivity to vulnerable borrowers prone to default.
- **Multiclass Brier Score ($0.0561$)**: Mean squared error between predicted probabilities and one-hot true labels.

### 4.2 Regression Metrics & Error Behavior Discussion
- **Mean Absolute Error (MAE)**: ₹263.36 [95% CI: ₹257.98 – ₹268.78].
- **Root Mean Squared Error (RMSE)**: ₹757.13 [95% CI: ₹715.03 – ₹801.00].
- **Coefficient of Determination ($R^2$)**: 0.9905 [95% CI: 0.9894 – 0.9915].
- **Symmetric MAPE (SMAPE)**: 5.40% [95% CI: 5.31% – 5.49%].

> **MAPE Interpretation Note**: Standard MAPE was interpreted cautiously as percentage error metrics become mathematically disproportionate when actual target values approach the lower bound (e.g. ₹500 minimum EMI). Symmetric MAPE (SMAPE) provides a robust, symmetric percentage error metric bounded between 0% and 200%.

---

## 5. Statistical Uncertainty via Bootstrap Resampling
We computed empirical 95% confidence intervals via 500 bootstrap iterations on the untouched holdout test partition ($N=60,720$):
- **Accuracy**: $0.9593$ [95% CI: $0.9577 - 0.9610$]
- **Balanced Accuracy**: $0.9445$ [95% CI: $0.9410 - 0.9482$]
- **Macro $F_1$**: $0.8725$ [95% CI: $0.8675 - 0.8771$]
- **High-Risk Recall**: $0.9289$ [95% CI: $0.9192 - 0.9384$]
- **Regression MAE**: ₹$263.36$ [95% CI: ₹$257.98 - ₹268.78$]
