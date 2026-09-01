import sqlite3
import pandas as pd
from typing import Dict, List, Any, Optional
from .database import get_db_connection
from src.utils.logger import setup_logger

logger = setup_logger("Repository")

class CustomerRepository:
    """CRUD operations for customer profiles."""

    @staticmethod
    def create_customer(data: Dict[str, Any]) -> int:
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cols = [
                "full_name", "email", "phone", "age", "gender", "marital_status",
                "education", "monthly_salary", "employment_type", "years_of_employment",
                "company_type", "house_type", "monthly_rent", "family_size", "dependents",
                "school_fees", "college_fees", "travel_expenses", "groceries_utilities",
                "other_monthly_expenses", "existing_loans", "current_emi_amount",
                "credit_score", "bank_balance", "emergency_fund"
            ]
            vals = [data.get(c) for c in cols]
            placeholders = ", ".join(["?"] * len(cols))
            sql = f"INSERT INTO customers ({', '.join(cols)}) VALUES ({placeholders})"
            cursor.execute(sql, vals)
            conn.commit()
            cust_id = cursor.lastrowid
            logger.info(f"Created customer profile ID={cust_id}")
            return cust_id
        finally:
            conn.close()

    @staticmethod
    def get_customer(customer_id: int) -> Optional[Dict[str, Any]]:
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM customers WHERE id = ?", (customer_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    @staticmethod
    def list_customers(limit: int = 100) -> List[Dict[str, Any]]:
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM customers ORDER BY id DESC LIMIT ?", (limit,))
            rows = cursor.fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    @staticmethod
    def update_customer(customer_id: int, data: Dict[str, Any]) -> bool:
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            allowed = [
                "full_name", "email", "phone", "age", "gender", "marital_status",
                "education", "monthly_salary", "employment_type", "years_of_employment",
                "company_type", "house_type", "monthly_rent", "family_size", "dependents",
                "school_fees", "college_fees", "travel_expenses", "groceries_utilities",
                "other_monthly_expenses", "existing_loans", "current_emi_amount",
                "credit_score", "bank_balance", "emergency_fund"
            ]
            set_clauses = [f"{k} = ?" for k in data.keys() if k in allowed]
            if not set_clauses:
                return False
            vals = [data[k] for k in data.keys() if k in allowed] + [customer_id]
            sql = f"UPDATE customers SET {', '.join(set_clauses)}, updated_at = CURRENT_TIMESTAMP WHERE id = ?"
            cursor.execute(sql, vals)
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    @staticmethod
    def delete_customer(customer_id: int) -> bool:
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM customers WHERE id = ?", (customer_id,))
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()


class PredictionRepository:
    """CRUD operations for underwriting prediction history."""

    @staticmethod
    def save_prediction(data: Dict[str, Any]) -> int:
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cols = [
                "customer_id", "emi_scenario", "requested_amount", "requested_tenure",
                "predicted_eligibility", "confidence_score", "prob_eligible",
                "prob_high_risk", "prob_not_eligible", "predicted_max_emi",
                "disposable_income", "foir", "model_version", "notes"
            ]
            vals = [data.get(c) for c in cols]
            placeholders = ", ".join(["?"] * len(cols))
            sql = f"INSERT INTO prediction_history ({', '.join(cols)}) VALUES ({placeholders})"
            cursor.execute(sql, vals)
            conn.commit()
            pred_id = cursor.lastrowid
            logger.info(f"Saved prediction record ID={pred_id}")
            return pred_id
        finally:
            conn.close()

    @staticmethod
    def get_prediction_history(limit: int = 200) -> List[Dict[str, Any]]:
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            sql = """
            SELECT p.*, c.full_name as customer_name, c.email as customer_email
            FROM prediction_history p
            LEFT JOIN customers c ON p.customer_id = c.id
            ORDER BY p.id DESC LIMIT ?
            """
            cursor.execute(sql, (limit,))
            rows = cursor.fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    @staticmethod
    def delete_prediction(prediction_id: int) -> bool:
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM prediction_history WHERE id = ?", (prediction_id,))
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()
