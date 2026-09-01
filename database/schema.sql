-- EMIPredict AI - Application Database Schema
-- Separate from raw ML training dataset

CREATE TABLE IF NOT EXISTS customers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name TEXT NOT NULL,
    email TEXT,
    phone TEXT,
    age REAL,
    gender TEXT,
    marital_status TEXT,
    education TEXT,
    monthly_salary REAL,
    employment_type TEXT,
    years_of_employment REAL,
    company_type TEXT,
    house_type TEXT,
    monthly_rent REAL,
    family_size INTEGER,
    dependents INTEGER,
    school_fees REAL,
    college_fees REAL,
    travel_expenses REAL,
    groceries_utilities REAL,
    other_monthly_expenses REAL,
    existing_loans TEXT,
    current_emi_amount REAL,
    credit_score REAL,
    bank_balance REAL,
    emergency_fund REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS prediction_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id INTEGER,
    emi_scenario TEXT,
    requested_amount REAL,
    requested_tenure INTEGER,
    predicted_eligibility TEXT NOT NULL,
    confidence_score REAL,
    prob_eligible REAL,
    prob_high_risk REAL,
    prob_not_eligible REAL,
    predicted_max_emi REAL NOT NULL,
    disposable_income REAL,
    foir REAL,
    model_version TEXT,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES customers(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_customers_email ON customers(email);
CREATE INDEX IF NOT EXISTS idx_predictions_customer_id ON prediction_history(customer_id);
CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON prediction_history(created_at);
