"""Data loading and realistic synthetic dataset generator."""

import os
from pathlib import Path
from typing import Optional, Union, Tuple
import numpy as np
import pandas as pd


SAMPLE_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "sample" / "customer_churn_sample.csv"


def generate_synthetic_data(num_records: int = 12000, random_seed: int = 42) -> pd.DataFrame:
    """Generate a realistic customer churn dataset.
    
    Contains realistic relationships and noise, without artificial perfection.
    Higher churn tendency is correlated with:
      - Short tenure
      - Month-to-month contract
      - High monthly charges
      - Frequent complaints
      - Low satisfaction score
      - Payment failures
      - Low engagement
    """
    rng = np.random.default_rng(random_seed)

    # 1. Customer Demographics
    customer_ids = [f"CUST-{10000 + i}" for i in range(num_records)]
    ages = rng.integers(18, 76, size=num_records)
    genders = rng.choice(["Male", "Female", "Other"], size=num_records, p=[0.49, 0.49, 0.02])
    locations = rng.choice(
        ["Mumbai", "Delhi NCR", "Bangalore", "Hyderabad", "Pune", "Chennai", "Kolkata", "Ahmedabad"],
        size=num_records,
        p=[0.22, 0.20, 0.18, 0.12, 0.10, 0.08, 0.05, 0.05]
    )
    occupations = rng.choice(
        ["Salaried Professional", "Self-Employed", "Business Owner", "Freelancer", "Student", "Retired"],
        size=num_records,
        p=[0.45, 0.20, 0.15, 0.10, 0.05, 0.05]
    )
    # Income in INR (Annual)
    income_base = rng.lognormal(mean=13.5, sigma=0.6, size=num_records)  # ~ ₹3L to ₹30L
    income = np.clip(np.round(income_base, -3), 200000, 4500000).astype(int)

    # 2. Service Information
    tenures = rng.integers(1, 73, size=num_records)  # 1 to 72 months
    service_types = rng.choice(
        ["Fiber Optic", "DSL", "Wireless Broadband", "Enterprise Dedicated"],
        size=num_records,
        p=[0.55, 0.25, 0.15, 0.05]
    )
    subscription_plans = rng.choice(
        ["Basic", "Standard", "Premium", "Ultra 4K"],
        size=num_records,
        p=[0.25, 0.40, 0.25, 0.10]
    )
    contract_types = rng.choice(
        ["Month-to-Month", "One-Year", "Two-Year"],
        size=num_records,
        p=[0.52, 0.28, 0.20]
    )
    number_of_products = rng.integers(1, 6, size=num_records)
    add_on_services = rng.choice(["Yes", "No"], size=num_records, p=[0.48, 0.52])

    # 3. Usage Information
    # Higher for fiber and ultra plans
    plan_mult = np.where(subscription_plans == "Ultra 4K", 1.5,
                 np.where(subscription_plans == "Premium", 1.25,
                 np.where(subscription_plans == "Standard", 1.0, 0.7)))

    login_freq = np.clip(np.round(rng.normal(20 * plan_mult, 8)), 1, 60).astype(int)
    call_duration = np.clip(np.round(rng.normal(320, 150)), 15, 1200).astype(int)  # mins/mo
    data_usage = np.clip(np.round(rng.normal(160 * plan_mult, 70)), 10, 800).astype(int)  # GB/mo
    num_transactions = np.clip(np.round(rng.poisson(lam=8)), 0, 40).astype(int)
    app_usage = np.clip(np.round(rng.normal(18 * plan_mult, 10), 1), 0.5, 90.0)  # hrs/mo
    website_visits = np.clip(np.round(rng.normal(14, 7)), 1, 55).astype(int)

    # 4. Financial Information
    base_price = {"Basic": 499, "Standard": 999, "Premium": 1699, "Ultra 4K": 2499}
    service_price = {"Fiber Optic": 250, "DSL": 100, "Wireless Broadband": 150, "Enterprise Dedicated": 800}
    addon_price = np.where(add_on_services == "Yes", 250, 0)
    
    monthly_charges = np.array([
        base_price[p] + service_price[s] + a + rng.integers(-50, 50)
        for p, s, a in zip(subscription_plans, service_types, addon_price)
    ], dtype=float)
    monthly_charges = np.clip(monthly_charges, 399.0, 3999.0)

    # Total Charges with realistic tenure compounding and occasional slight discount variation
    total_charges = np.round(monthly_charges * tenures * rng.uniform(0.95, 1.02, size=num_records), 2)

    payment_methods = rng.choice(
        ["UPI", "Credit Card", "Net Banking", "Debit Card", "Electronic Wallet"],
        size=num_records,
        p=[0.40, 0.25, 0.15, 0.12, 0.08]
    )
    payment_failures = rng.choice([0, 1, 2, 3, 4], size=num_records, p=[0.72, 0.16, 0.07, 0.03, 0.02])
    outstanding_balance = np.where(
        payment_failures > 0,
        np.round(monthly_charges * payment_failures * rng.uniform(0.8, 1.2, size=num_records), 2),
        0.0
    )

    # 5. Customer Support & Satisfaction
    # Base complaints
    complaints = rng.choice([0, 1, 2, 3, 4, 5, 6], size=num_records, p=[0.50, 0.24, 0.13, 0.07, 0.03, 0.02, 0.01])
    support_calls = np.clip(complaints * 2 + rng.integers(0, 4, size=num_records), 0, 15)
    tickets_raised = np.clip(complaints + rng.integers(0, 3, size=num_records), 0, 12)
    resolution_time = np.clip(np.round(rng.normal(16 + complaints * 4, 8), 1), 1.0, 72.0)

    # Satisfaction Score (1 to 5) strongly negatively impacted by complaints and resolution time
    sat_latent = 4.2 - (0.45 * complaints) - (0.02 * resolution_time) + rng.normal(0, 0.5, size=num_records)
    satisfaction_score = np.clip(np.round(sat_latent), 1, 5).astype(int)

    # 6. Realistic Latent Churn Propensity (Logit Model with Noise)
    # Realistic weights based on empirical churn research:
    # Contract: Month-to-month strongly increases risk (+1.3) vs Two-year (-1.1)
    contract_risk = np.where(contract_types == "Month-to-Month", 1.25,
                    np.where(contract_types == "One-Year", -0.2, -1.1))
    
    # Tenure: shorter tenure has higher risk
    tenure_risk = -0.04 * (tenures - 36)
    
    # Charges: higher monthly charges elevate risk
    charges_risk = 0.0007 * (monthly_charges - 1400)
    
    # Support & Complaints: strong churn driver
    support_risk = 0.45 * complaints + 0.15 * payment_failures
    
    # Satisfaction: low satisfaction increases risk
    sat_risk = -0.65 * (satisfaction_score - 3)
    
    # Engagement: low login & app usage
    engagement_risk = -0.025 * (login_freq - 25)
    
    # Baseline logit offset to achieve ~ 18-22% realistic churn rate
    baseline = -1.95
    
    # Latent logit with Gaussian noise for realism (not deterministic!)
    latent_logit = (
        baseline
        + contract_risk
        + tenure_risk
        + charges_risk
        + support_risk
        + sat_risk
        + engagement_risk
        + rng.normal(0, 0.75, size=num_records)
    )
    
    # Sigmoid function for ground truth probability
    prob_churn = 1.0 / (1.0 + np.exp(-latent_logit))
    
    # Sample binary outcome from Bernoulli(prob_churn)
    churn = (rng.uniform(0, 1, size=num_records) < prob_churn).astype(int)

    df = pd.DataFrame({
        "Customer_ID": customer_ids,
        "Age": ages,
        "Gender": genders,
        "Location": locations,
        "Occupation": occupations,
        "Income": income,
        "Tenure": tenures,
        "Service_Type": service_types,
        "Subscription_Plan": subscription_plans,
        "Contract_Type": contract_types,
        "Number_of_Products": number_of_products,
        "Add_On_Services": add_on_services,
        "Login_Frequency": login_freq,
        "Call_Duration": call_duration,
        "Data_Usage": data_usage,
        "Number_of_Transactions": num_transactions,
        "App_Usage": app_usage,
        "Website_Visits": website_visits,
        "Monthly_Charges": monthly_charges,
        "Total_Charges": total_charges,
        "Payment_Method": payment_methods,
        "Payment_Failures": payment_failures,
        "Outstanding_Balance": outstanding_balance,
        "Complaints": complaints,
        "Support_Calls": support_calls,
        "Tickets_Raised": tickets_raised,
        "Resolution_Time": resolution_time,
        "Satisfaction_Score": satisfaction_score,
        "Churn": churn,
    })

    return df


def load_dataset(file_source: Optional[Union[str, Path, object]] = None) -> pd.DataFrame:
    """Load dataset from file path, buffer, or generate sample dataset if missing."""
    if file_source is not None:
        if isinstance(file_source, (str, Path)):
            return pd.read_csv(file_source)
        # Streamlit UploadedFile buffer
        return pd.read_csv(file_source)

    # Check default sample path
    if SAMPLE_DATA_PATH.exists():
        return pd.read_csv(SAMPLE_DATA_PATH)

    # Generate and save sample
    os.makedirs(SAMPLE_DATA_PATH.parent, exist_ok=True)
    df = generate_synthetic_data(num_records=12000, random_seed=42)
    df.to_csv(SAMPLE_DATA_PATH, index=False)
    return df
