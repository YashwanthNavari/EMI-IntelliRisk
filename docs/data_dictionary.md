# Data Dictionary — EMIPredict AI

This document describes all **27 raw and source variables** present in `emi_prediction_dataset.csv` (404,800 records).

| Column Name | Data Type | Null Count | Null % | Domain / Range | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `age` | Float | 0 | 0.0% | 18 – 80 | Applicant age in years. Cleaned from multi-dot string formatting. |
| `gender` | Categorical | 0 | 0.0% | `Male`, `Female` | Self-reported gender, normalized from 8 raw variations. |
| `marital_status` | Categorical | 0 | 0.0% | `Married`, `Single` | Legal marital status. |
| `education` | Categorical | 2,404 | 0.59% | `Graduate`, `Post Graduate`, `High School`, `Professional` | Highest educational qualification attained. |
| `monthly_salary` | Float | 0 | 0.0% | ₹3,967 – ₹500,000 | Verified monthly gross income. Cleaned from string formatting. |
| `employment_type` | Categorical | 0 | 0.0% | `Private`, `Government`, `Self-employed` | Nature of employment contract. |
| `years_of_employment` | Float | 0 | 0.0% | 0.0 – 45.0 | Total professional work experience in years. |
| `company_type` | Categorical | 0 | 0.0% | `Large Indian`, `MNC`, `Mid-size`, `Startup`, `Small` | Employer organization classification. |
| `house_type` | Categorical | 0 | 0.0% | `Rented`, `Own`, `Family` | Residential ownership arrangement. |
| `monthly_rent` | Float | 2,426 | 0.60% | ₹0 – ₹200,000 | Monthly residential rental cost (₹0 for Own/Family). |
| `family_size` | Integer | 0 | 0.0% | 1 – 6 | Total household member count. |
| `dependents` | Integer | 0 | 0.0% | 0 – 5 | Number of financially dependent household members. |
| `school_fees` | Float | 0 | 0.0% | ₹0 – ₹50,000 | Monthly school tuition and academic expenses. |
| `college_fees` | Float | 0 | 0.0% | ₹0 – ₹100,000 | Monthly higher education expenses. |
| `travel_expenses` | Float | 0 | 0.0% | ₹0 – ₹50,000 | Monthly commute and transportation costs. |
| `groceries_utilities` | Float | 0 | 0.0% | ₹0 – ₹100,000 | Food, utilities, electricity, water, and grocery expenses. |
| `other_monthly_expenses`| Float | 0 | 0.0% | ₹0 – ₹50,000 | Miscellaneous living and discretionary expenses. |
| `existing_loans` | Categorical | 0 | 0.0% | `Yes`, `No` | Indicator whether applicant has active debt obligations. |
| `current_emi_amount` | Float | 0 | 0.0% | ₹0 – ₹56,300 | Ongoing monthly debt service commitments. |
| `credit_score` | Float | 2,420 | 0.60% | 300 – 900 | CIBIL/Experian credit score. Bound-clipped from raw anomalies. |
| `bank_balance` | Float | 2,426 | 0.60% | ₹0 – ₹2,500,000 | Liquid savings account balance. |
| `emergency_fund` | Float | 2,351 | 0.58% | ₹0 – ₹1,000,000 | Dedicated emergency reserve balance. |
| `emi_scenario` | Categorical | 0 | 0.0% | 5 categories | Target loan purpose (Personal, Vehicle, Education, E-Commerce, Appliances). |
| `requested_amount` | Float | 0 | 0.0% | ₹10,000 – ₹1,500,000 | Loan principal requested. |
| `requested_tenure` | Integer | 0 | 0.0% | 3 – 84 | Requested repayment duration in months. |
| **`emi_eligibility`** | Categorical | 0 | 0.0% | `Eligible`, `High_Risk`, `Not_Eligible` | **Classification Target**: Credit approval recommendation. |
| **`max_monthly_emi`** | Float | 0 | 0.0% | ₹500 – ₹91,040.40 | **Regression Target**: Maximum sustainable monthly EMI amount. |
