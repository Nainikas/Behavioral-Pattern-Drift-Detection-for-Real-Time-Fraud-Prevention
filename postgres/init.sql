CREATE TABLE IF NOT EXISTS prediction_logs (
    id SERIAL PRIMARY KEY,
    customer_age FLOAT,
    zip_count_4w FLOAT,
    income FLOAT,
    prediction INTEGER,
    fraud_probability FLOAT,
    is_risky_override BOOLEAN,
    drift_customer_age BOOLEAN,
    drift_zip_count_4w BOOLEAN,
    drift_income BOOLEAN,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
