import pytest
from app.services.prediction_service import predict_risk_and_emi
from database.database import init_db
from database.repository import CustomerRepository, PredictionRepository

def test_database_initialization_and_crud():
    init_db()

    # Create
    cust_id = CustomerRepository.create_customer({
        "full_name": "Test User",
        "email": "test@example.com",
        "monthly_salary": 70000.0,
        "credit_score": 750.0,
        "monthly_rent": 15000.0,
        "current_emi_amount": 0.0
    })
    assert cust_id > 0

    # Read
    cust = CustomerRepository.get_customer(cust_id)
    assert cust is not None
    assert cust["full_name"] == "Test User"
    assert cust["monthly_salary"] == 70000.0

    # Update
    updated = CustomerRepository.update_customer(cust_id, {"monthly_salary": 80000.0})
    assert updated is True
    cust_upd = CustomerRepository.get_customer(cust_id)
    assert cust_upd["monthly_salary"] == 80000.0

    # Save prediction
    pred_id = PredictionRepository.save_prediction({
        "customer_id": cust_id,
        "emi_scenario": "Personal Loan EMI",
        "requested_amount": 200000.0,
        "requested_tenure": 24,
        "predicted_eligibility": "Eligible",
        "confidence_score": 0.90,
        "predicted_max_emi": 18000.0,
        "model_version": "1.0.0"
    })
    assert pred_id > 0

    # Delete
    del_pred = PredictionRepository.delete_prediction(pred_id)
    assert del_pred is True
    del_cust = CustomerRepository.delete_customer(cust_id)
    assert del_cust is True

def test_prediction_service_structure():
    sample_payload = {
        "age": 35.0,
        "gender": "Female",
        "marital_status": "Married",
        "education": "Post Graduate",
        "monthly_salary": 90000.0,
        "employment_type": "Private",
        "years_of_employment": 5.0,
        "company_type": "MNC",
        "house_type": "Rented",
        "monthly_rent": 18000.0,
        "family_size": 3,
        "dependents": 1,
        "school_fees": 3000.0,
        "college_fees": 0.0,
        "travel_expenses": 5000.0,
        "groceries_utilities": 14000.0,
        "other_monthly_expenses": 6000.0,
        "existing_loans": "No",
        "current_emi_amount": 0.0,
        "credit_score": 770.0,
        "bank_balance": 350000.0,
        "emergency_fund": 120000.0,
        "emi_scenario": "Personal Loan EMI",
        "requested_amount": 400000.0,
        "requested_tenure": 24
    }

    res = predict_risk_and_emi(sample_payload)
    assert "predicted_eligibility" in res
    assert "predicted_max_emi" in res
    assert "class_probabilities" in res
    assert "driving_factors" in res
    assert res["predicted_max_emi"] >= 500.0
    assert len(res["driving_factors"]) >= 3
