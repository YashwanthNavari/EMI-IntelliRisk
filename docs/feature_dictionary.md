# Feature Dictionary — EMIPredict AI

This document mathematically defines all **engineered financial domain features** created in `src/features/`.

---

### 1. Total Living Expenses (`total_expenses`)
- **Formula**:
  $$\text{Total Expenses} = \text{monthly\_rent} + \text{school\_fees} + \text{college\_fees} + \text{travel\_expenses} + \text{groceries\_utilities} + \text{other\_monthly\_expenses}$$
- **Data Type**: Float (₹)
- **Business Meaning**: Aggregate monthly committed cost of living.

---

### 2. True Disposable Income (`disposable_income`)
- **Formula**:
  $$\text{Disposable Income} = \text{monthly\_salary} - \text{Total Expenses} - \text{current\_emi\_amount}$$
- **Data Type**: Float (₹)
- **Business Meaning**: Net discretionary cash flow remaining before new debt obligations.

---

### 3. Expense-to-Income Ratio (`expense_to_income_ratio`)
- **Formula**:
  $$\text{ETI} = \frac{\text{Total Expenses}}{\text{monthly\_salary} + 10^{-6}}$$
- **Data Type**: Float (Ratio)
- **Business Meaning**: Proportion of monthly salary consumed by living expenses. Prudent benchmark $\le 0.60$.

---

### 4. Fixed Obligation to Income Ratio (`foir`)
- **Formula**:
  $$\text{FOIR} = \frac{\text{monthly\_rent} + \text{current\_emi\_amount}}{\text{monthly\_salary} + 10^{-6}}$$
- **Data Type**: Float (Ratio)
- **Business Meaning**: Core FinTech underwriting indicator. Values $> 0.50$ trigger high risk or rejection.

---

### 5. Existing EMI Burden Ratio (`emi_burden_ratio`)
- **Formula**:
  $$\text{EMI Burden} = \frac{\text{current\_emi\_amount}}{\text{monthly\_salary} + 10^{-6}}$$
- **Data Type**: Float (Ratio)
- **Business Meaning**: Debt service burden prior to new loan request.

---

### 6. Savings Strength / Liquidity Coverage (`savings_to_income_ratio`)
- **Formula**:
  $$\text{Savings Strength} = \frac{\text{bank\_balance} + \text{emergency\_fund}}{\text{monthly\_salary} + 10^{-6}}$$
- **Data Type**: Float (Ratio)
- **Business Meaning**: Liquid asset reserve multiple relative to monthly earnings.

---

### 7. Emergency Fund Buffer in Months (`emergency_fund_buffer_months`)
- **Formula**:
  $$\text{Emergency Buffer} = \frac{\text{emergency\_fund}}{\text{Total Expenses} + \text{current\_emi\_amount} + 10^{-6}}$$
- **Data Type**: Float (Months)
- **Business Meaning**: Duration applicant can sustain household commitments during zero-income shocks.

---

### 8. Loan Pressure Ratio (`loan_to_income_ratio`)
- **Formula**:
  $$\text{Loan Pressure} = \frac{\text{requested\_amount}}{12 \times \text{monthly\_salary} + 10^{-6}}$$
- **Data Type**: Float (Ratio)
- **Business Meaning**: Requested principal amount expressed as a multiple of annual salary.

---

### 9. Requested Monthly Installment Estimate (`requested_monthly_installment_estimate`)
- **Formula**:
  $$\text{Est. Installment} = \frac{\text{requested\_amount}}{\text{requested\_tenure} + 10^{-6}}$$
- **Data Type**: Float (₹/Month)
- **Business Meaning**: Approximate monthly linear installment obligation requested by applicant.

---

### 10. Per-Capita Disposable Income (`per_capita_disposable_income`)
- **Formula**:
  $$\text{Per-Capita Disposable} = \frac{\text{disposable\_income}}{\text{family\_size}}$$
- **Data Type**: Float (₹)
- **Business Meaning**: Discretionary surplus distributed across all household members.

---

### 11. Credit Risk Tier (`credit_risk_tier`)
- **Formula**:
  $$\begin{cases}
  \text{Poor} & \text{if credit\_score} < 650 \\
  \text{Fair} & \text{if } 650 \le \text{credit\_score} < 700 \\
  \text{Good} & \text{if } 700 \le \text{credit\_score} < 750 \\
  \text{Excellent} & \text{if credit\_score} \ge 750
  \end{cases}$$
- **Data Type**: Categorical

---

### 12. Composite Financial Health Score (`financial_health_score`)
- **Formula**:
  $$\text{Score} = 100 - \max(0, (\text{ETI}-0.6)\times 80) - \max(0, (\text{FOIR}-0.4)\times 70) - \max(0, (\text{EMI Burden}-0.3)\times 60) - \max(0, (3-\text{Buffer})\times 5) - (25 \text{ if disp} < 0)$$
- **Data Type**: Float (0.0 to 100.0)
- **Business Meaning**: Unified 100-point credit resilience index.
