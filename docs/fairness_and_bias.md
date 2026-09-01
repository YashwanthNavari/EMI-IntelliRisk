# Demographic Fairness & Subgroup Performance Audit — EMIPredict AI

## 1. Ethical Governance & Purpose
In compliance with fair lending guidelines, automated credit risk systems must be audited across demographic and socio-economic subgroups to detect predictive disparities.

> **Academic Disclaimer**: This audit assesses predictive parity across subgroups present in the dataset. It does not establish that the historical dataset itself is free from systemic lending bias.

---

## 2. Demographic Subgroup Empirical Benchmarks ($N=60,720$ Holdout Set)

### 2.1 Gender Subgroups
| Subgroup | Sample Size ($N$) | Proportion | Accuracy | Macro $F_1$ | High-Risk Recall | Regression MAE (₹) | Regression $R^2$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Female** | 24,189 | 39.8% | 95.90% | 0.8732 | 93.08% | ₹261.25 | 0.9909 |
| **Male** | 36,531 | 60.2% | 95.95% | 0.8717 | 92.73% | ₹264.84 | 0.9902 |

*Parity Analysis: Across Gender, Macro $F_1$ disparity is negligible ($\Delta F_1 = 0.0015$), and High-Risk Recall is virtually identical (93.08% vs 92.73%), with regression MAE difference under ₹3.60.*

---

### 2.2 Age Bracket Subgroups
| Subgroup | Age Range | Sample Size ($N$) | Proportion | Accuracy | Macro $F_1$ | High-Risk Recall | Regression MAE (₹) | Regression $R^2$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Young Adults** | $< 30$ | 9,096 | 15.0% | 95.45% | 0.8646 | 91.85% | ₹278.14 | 0.9894 |
| **Mid-Career** | $30 - 50$ | 45,538 | 75.0% | 95.99% | 0.8728 | 93.01% | ₹261.88 | 0.9904 |
| **Senior** | $> 50$ | 6,086 | 10.0% | 96.19% | 0.8813 | 93.43% | ₹252.82 | 0.9924 |

---

### 2.3 Employment Sector Subgroups
| Subgroup | Sector | Sample Size ($N$) | Proportion | Accuracy | Macro $F_1$ | High-Risk Recall | Regression MAE (₹) | Regression $R^2$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Government** | Public Sector / PSU | 12,294 | 20.2% | 96.12% | 0.8779 | 91.19% | ₹297.36 | 0.9917 |
| **Private** | Corporate / IT / Services | 42,319 | 69.7% | 95.95% | 0.8741 | 93.69% | ₹253.61 | 0.9900 |
| **Self-Employed** | Business / Freelance | 6,107 | 10.1% | 95.43% | 0.8475 | 90.43% | ₹263.00 | 0.9897 |

---

### 2.4 Income Tier Subgroups
| Subgroup | Monthly Salary Range | Sample Size ($N$) | Proportion | Accuracy | Macro $F_1$ | High-Risk Recall | Regression MAE (₹) | Regression $R^2$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Low Income** | $< ₹35,000$ | 14,998 | 24.7% | 97.89% | 0.8609 | 91.58% | ₹191.64 | 0.9336 |
| **Middle Income** | $₹35,000 - ₹75,000$ | 31,481 | 51.8% | 95.91% | 0.8719 | 94.07% | ₹180.92 | 0.9953 |
| **High Income** | $> ₹75,000$ | 14,241 | 23.5% | 93.90% | 0.8693 | 91.48% | ₹521.33 | 0.9870 |

---

## 3. Disparity Mitigation & Governance Guidelines
1. **Regular Recalibration**: Continuous monitoring of subgroup approval rates across income quartiles.
2. **Human-in-the-Loop Review**: Mandatory manual underwriter review for applicants flagged as `High_Risk`.
3. **Transparent Factor Explanations**: Requiring clear financial reasoning (e.g., FOIR, liquid buffer) for every adverse credit action.
