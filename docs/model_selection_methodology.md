# Model Selection Methodology — EMIPredict AI

## 1. Executive Rationale
In automated consumer credit underwriting, conventional overall accuracy is a misleading metric due to target class imbalance (`Not_Eligible`: 77.29%, `Eligible`: 18.39%, `High_Risk`: 4.32%). A naive majority classifier could achieve 77.3% accuracy while completely failing to identify risky or eligible borrowers.

Therefore, EMIPredict AI uses a **weighted multi-criteria decision framework** to select champion models.

---

## 2. Classification Multi-Criteria Decision Formulation

The composite selection score $S_{\text{clf}}$ is defined as:

$$S_{\text{clf}} = 0.40 \cdot \text{Macro } F_1 + 0.30 \cdot \text{Recall}_{\text{High\_Risk}} + 0.20 \cdot \text{Balanced Accuracy} + 0.10 \cdot \text{ROC-AUC}_{\text{OVR}}$$

### Weight Justification & Credit Risk Implications:
1. **Macro $F_1$ Score ($w_1 = 0.40$)**:
   - Calculates the unweighted mean of $F_1$ scores across all 3 classes:
     $$\text{Macro } F_1 = \frac{F_{1,\text{Eligible}} + F_{1,\text{High\_Risk}} + F_{1,\text{Not\_Eligible}}}{3}$$
   - Penalizes models that sacrifice minority classes to boost overall accuracy.
2. **High-Risk Class Recall ($w_2 = 0.30$)**:
   - In lending underwriting, a **False Negative** on the `High_Risk` class (i.e. approving a delinquent applicant) causes direct credit default and principal loss.
   - High sensitivity on this class is paramount for capital preservation.
3. **Balanced Accuracy ($w_3 = 0.20$)**:
   - The arithmetic mean of recall across all classes, ensuring that `Eligible` applicants are also treated equitably.
4. **Multiclass ROC-AUC ($w_4 = 0.10$)**:
   - Measures threshold-invariant class separability across One-vs-Rest (OVR) ROC curves.

### Empirical Classification Scores:
| Model Architecture | Macro $F_1$ | High-Risk Recall | Balanced Acc | ROC-AUC | Composite Score ($S_{\text{clf}}$) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.6775 | 0.7594 | 0.8534 | 0.9508 | **0.7646** |
| **Random Forest** | 0.8013 | 0.7808 | 0.8862 | 0.9827 | **0.8303** |
| **Gradient Boosting (LightGBM)** | **0.8719** | **0.9363** | **0.9464** | **0.9949** | 🏆 **0.9184** |

---

## 3. Regression Selection Formulation

The composite selection score $S_{\text{reg}}$ evaluates variance explained and error magnitude:

$$S_{\text{reg}} = R^2 - \frac{\text{RMSE}}{10,000}$$

### Empirical Regression Scores:
| Model Architecture | RMSE (₹) | MAE (₹) | $R^2$ Score | Selection Outcome |
| :--- | :--- | :--- | :--- | :--- |
| **Ridge Linear Regression** | ₹3,782.04 | ₹2,461.50 | 0.7636 | Baseline |
| **Random Forest Regressor** | ₹1,005.67 | ₹313.05 | 0.9833 | Benchmarked |
| **Gradient Boosting (LightGBM)** | **₹762.69** | **₹261.89** | **0.9904** | 🏆 **Champion** |
