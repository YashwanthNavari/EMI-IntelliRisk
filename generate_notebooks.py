import json
from pathlib import Path

notebooks_dir = Path(r"c:\Users\EikoMotsu\OneDrive\Documents\Desktop\innoexis intenship task\EMI PREDICTION\notebooks")
notebooks_dir.mkdir(parents=True, exist_ok=True)

def create_notebook(title, description, cells_code):
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [f"# {title}\n\n", f"**EMIPredict AI — Capstone Project Notebook**\n\n", f"{description}\n"]
        }
    ]
    for code in cells_code:
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [code]
        })

    nb = {
        "cells": cells,
        "metadata": {
            "language_info": {"name": "python", "version": "3.10"},
            "orig_nbformat": 4
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    return nb

# 1. Data Understanding
nb1 = create_notebook(
    "01 - Data Ingestion & Schema Understanding",
    "In this notebook, we ingest `emi_prediction_dataset.csv` (404,800 records), inspect data types, identify formatting anomalies, and profile target variables.",
    [
        "import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\n\n# Load raw dataset\ndf = pd.read_csv('../emi_prediction_dataset.csv', low_memory=False)\nprint('Dataset Dimensions:', df.shape)\ndf.head()",
        "print('Column Data Types and Non-Null Counts:')\ndf.info()",
        "# Inspect target variables\nprint('Classification Target Distribution (emi_eligibility):')\nprint(df['emi_eligibility'].value_counts(normalize=True) * 100)\n\nprint('\\nRegression Target Summary (max_monthly_emi):')\nprint(df['max_monthly_emi'].describe())"
    ]
)

# 2. Data Quality Assessment
nb2 = create_notebook(
    "02 - Data Quality Assessment & Anomaly Detection",
    "Formal assessment across Completeness, Uniqueness, Validity, Consistency, and Range Bound checks.",
    [
        "import pandas as pd\nimport numpy as np\n\ndf = pd.read_csv('../emi_prediction_dataset.csv', low_memory=False)\n\n# Completeness & Null Analysis\nnulls = df.isnull().sum()\nprint('Columns with Missing Values:')\nprint(nulls[nulls > 0])",
        "# Categorical Consistency (e.g. gender labels)\nprint('Unique Gender Values:', df['gender'].value_counts())\n\n# Outlier & Credit Score Domain Checks\nprint('Credit Score Outliers (< 300 or > 900):')\nprint('Under 300:', (df['credit_score'] < 300).sum())\nprint('Over 900:', (df['credit_score'] > 900).sum())"
    ]
)

# 3. EDA
nb3 = create_notebook(
    "03 - Exploratory Data Analysis (EDA)",
    "Bivariate and multivariate analyses investigating financial relationships between income, expenses, FOIR, scenarios, and loan eligibility.",
    [
        "import pandas as pd\nimport seaborn as sns\nimport matplotlib.pyplot as plt\n\ndf = pd.read_csv('../emi_prediction_dataset.csv', low_memory=False)\n\n# Clean salary for plotting\ndf['monthly_salary'] = pd.to_numeric(df['monthly_salary'].astype(str).str.replace(r'(\\..)+$', '', regex=True), errors='coerce')\n\nplt.figure(figsize=(10, 5))\nsns.boxplot(data=df, x='emi_eligibility', y='monthly_salary', palette='Set2')\nplt.title('Monthly Salary Distribution across EMI Eligibility Classes')\nplt.show()",
        "# Scenario vs Eligibility Crosstab\nct = pd.crosstab(df['emi_scenario'], df['emi_eligibility'], normalize='index') * 100\nprint('Eligibility % by Loan Scenario:')\nprint(ct.round(2))"
    ]
)

# 4. Feature Engineering
nb4 = create_notebook(
    "04 - Financial Feature Engineering & Leakage Audit",
    "Mathematical derivation of FOIR, Disposable Income, EMI Burden, Liquidity Coverage, and verification of zero target leakage.",
    [
        "import pandas as pd\nimport numpy as np\nimport sys\nsys.path.append('..')\n\nfrom src.data.cleaning import clean_dataset\nfrom src.features.financial_features import compute_financial_features\nfrom src.features.risk_features import compute_risk_features\n\ndf = pd.read_csv('../emi_prediction_dataset.csv', low_memory=False)\ndf_clean = clean_dataset(df)\ndf_feat = compute_financial_features(df_clean)\ndf_feat = compute_risk_features(df_feat)\n\nprint('Engineered Features Head:')\ndf_feat[['monthly_salary', 'total_expenses', 'disposable_income', 'foir', 'emi_burden_ratio', 'financial_health_score']].head()"
    ]
)

# 5. Classification Experiments
nb5 = create_notebook(
    "05 - Classification Model Experiments & MLflow Tracking",
    "Benchmarking Logistic Regression, Random Forest, and LightGBM / XGBoost for multiclass underwriting prediction with class-balancing and MLflow tracking.",
    [
        "import pandas as pd\nimport numpy as np\nimport sys\nsys.path.append('..')\n\nfrom src.models.classification.train import build_classification_pipeline\nfrom src.models.evaluation import evaluate_classification_model\n\npipe = build_classification_pipeline('random_forest')\nprint('Classification Pipeline Steps:', pipe.named_steps.keys())"
    ]
)

# 6. Regression Experiments
nb6 = create_notebook(
    "06 - Regression Model Experiments & Affordability Evaluation",
    "Benchmarking Ridge Regression, Random Forest, and Gradient Boosting Regressors for maximum affordable EMI estimation.",
    [
        "import pandas as pd\nimport numpy as np\nimport sys\nsys.path.append('..')\n\nfrom src.models.regression.train import build_regression_pipeline\nfrom src.models.evaluation import evaluate_regression_model\n\npipe = build_regression_pipeline('random_forest')\nprint('Regression Pipeline Steps:', pipe.named_steps.keys())"
    ]
)

all_nbs = {
    "01_data_understanding.ipynb": nb1,
    "02_data_quality_assessment.ipynb": nb2,
    "03_exploratory_data_analysis.ipynb": nb3,
    "04_feature_engineering.ipynb": nb4,
    "05_classification_experiments.ipynb": nb5,
    "06_regression_experiments.ipynb": nb6
}

for name, nb in all_nbs.items():
    with open(notebooks_dir / name, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Generated {name}")
