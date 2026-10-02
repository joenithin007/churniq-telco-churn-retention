# Customer Churn Prediction & Retention Intelligence Dashboard

> **Predict customer churn. Understand the risk. Prioritize retention.**

A full-stack, production-quality Customer Churn Prediction & Retention Intelligence application built end-to-end with Python, Scikit-Learn, XGBoost, SHAP, and Streamlit.

---

## 📌 Executive Summary & Business Problem

Customer churn is one of the most critical threats to recurring revenue and long-term business health. Acquiring a new customer costs between **5× to 7× more** than retaining an existing one. However, traditional reporting answers *who already churned* after the revenue has been lost.

This dashboard transforms reactive reporting into **proactive retention intelligence** by answering four core operational questions:
1. **Which customers are likely to leave?**
2. **How likely are they to leave?** (True statistical churn probability via `predict_proba`)
3. **Why are they at risk?** (Personalized Tree SHAP attribution & behavioral drivers)
4. **Which customers should the business review first?** (Strategic 2×2 Value-Risk prioritization matrix)

---

## 🏗️ Architecture & Pipeline Flow

```
[Raw Customer Data / CSV Upload]
               │
               ▼
[Data Validation & Schema Standardizer] (utils/validation.py)
  • Duplicate detection, boundary checks, synonym mapping
               │
               ▼
[Data Cleaning Pipeline] (src/data_cleaning.py)
  • Median numerical imputation, mode categorical imputation, anomaly audits
               │
               ▼
[Feature Engineering Engine] (src/feature_engineering.py)
  • Spend run-rate, Support Intensity, Complaint Rate, Engagement Score, Customer Value
               │
               ▼
[ML Training & Strict Holdout Evaluation] (src/train_model.py, src/evaluate_model.py)
  • Stratified 80/20 split, leak-free ColumnTransformer
  • Logistic Regression vs. Random Forest vs. XGBoost Classifier
               │
               ▼
[Inference & Probability Engine] (src/predict.py)
  • Continuous Churn Probability [0.0 - 1.0]
               │
               ▼
[Risk Segmentation & Retention Engine] (src/risk_segmentation.py, src/retention.py)
  • Low (≤30%), Medium (31-70%), High (>70%) Risk Tiers
  • P1 Critical, P2 High, P3 Medium, P4 Low Customer Review Prioritization
               │
               ▼
[Explainable AI Engine] (src/explainability.py)
  • Global SHAP feature importances & personalized individual waterfall attributions
               │
               ▼
[Interactive Enterprise Dashboard] (app.py & pages/)
  • Executive Overview • Customer Deep Dive • Churn Drivers • Retention Queue
  • Model Performance • Data Quality Audit • Dynamic Multi-dimensional Filters
```

---

## 📂 Project Structure

```
DS Customer churn project/
├── app.py                      # Main Streamlit application entry point & router
├── requirements.txt            # Project dependencies
├── README.md                   # System documentation & usage manual
├── .gitignore                  # Git ignore rules
│
├── data/
│   ├── raw/                    # Raw uploaded customer datasets
│   ├── processed/              # Cleaned & engineered datasets
│   └── sample/
│       └── customer_churn_sample.csv   # Realistic 12,000-customer benchmark dataset
│
├── models/
│   ├── churn_model.pkl         # Serialized optimal ML classifier
│   ├── preprocessor.pkl        # Serialized ColumnTransformer
│   └── model_metadata.json     # Model version, hyperparameters & evaluation metrics
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Synthetic generator & CSV loading pipeline
│   ├── data_cleaning.py        # Automated imputation & anomaly cleaning
│   ├── feature_engineering.py  # Spend, support intensity, engagement & value proxy
│   ├── eda.py                  # Dynamic exploratory aggregations & Plotly figures
│   ├── train_model.py          # Multi-algorithm training, evaluation & persistence
│   ├── evaluate_model.py       # Confusion matrix, ROC-AUC curve & benchmark comparison
│   ├── predict.py              # Inference pipeline for churn probabilities & risk
│   ├── explainability.py       # Global & local SHAP attributions
│   ├── risk_segmentation.py    # Configurable risk tier engine
│   └── retention.py            # Prioritization matrix & rule-based review engine
│
├── pages/
│   ├── __init__.py
│   ├── overview.py             # Page 1: Executive Overview & Revenue at Risk
│   ├── customer_risk.py        # Page 2: Customer Search, Profile & Individual SHAP
│   ├── churn_drivers.py        # Page 3: Behavioral breakdowns & Global Importance
│   ├── retention.py            # Page 4: Retention Quadrants & Review Queue
│   ├── model_performance.py    # Page 5: Test Metrics, Confusion Matrix, ROC & Retrain
│   └── data_quality.py         # Page 6: Schema Integrity, Missingness & Audit Log
│
├── utils/
│   ├── __init__.py
│   ├── calculations.py         # Analytical KPIs & customer value proxies
│   ├── formatting.py           # Indian currency (INR), percentages, badges
│   └── validation.py           # Schema validator, duplicate & boundary checking
│
└── tests/
    ├── __init__.py
    ├── test_data.py            # Data loading, generation & cleaning tests
    ├── test_model.py           # ML training, inference & probability range tests
    └── test_calculations.py    # KPI, risk categorization & priority tests
```

---

## 📊 Dataset Schema

The system ingests enterprise telecom/SaaS customer datasets with 28+ features across 5 operational dimensions:

| Category | Features |
| :--- | :--- |
| **Demographics** | `Customer_ID`, `Age`, `Gender`, `Location`, `Occupation`, `Income` |
| **Service Info** | `Tenure`, `Service_Type`, `Subscription_Plan`, `Contract_Type`, `Number_of_Products`, `Add_On_Services` |
| **Usage Info** | `Login_Frequency`, `Call_Duration`, `Data_Usage`, `Number_of_Transactions`, `App_Usage`, `Website_Visits` |
| **Financials** | `Monthly_Charges`, `Total_Charges`, `Payment_Method`, `Payment_Failures`, `Outstanding_Balance` |
| **Customer Support** | `Complaints`, `Support_Calls`, `Tickets_Raised`, `Resolution_Time`, `Satisfaction_Score` |
| **Target Variable** | `Churn` (Binary: `1` = Churned, `0` = Retained) |

---

## 🧹 Data Cleaning & Preprocessing

- **Missing Values:** Numerical columns imputed with training median; categorical features imputed with mode.
- **Deduplication:** Enforces unique `Customer_ID` values; automatically detects and removes identical records.
- **Boundary Audits:** Detects and clips out-of-range inputs (e.g. `Age` < 18 or > 100, negative `Monthly_Charges`, invalid `Satisfaction_Score` outside [1, 5]).
- **Zero-Leakage Guarantee:** Preprocessing transformers (`StandardScaler`, `OneHotEncoder`) are strictly fit **only** on training splits.

---

## 🛠️ Feature Engineering

1. **`Average_Monthly_Spend`**: $\frac{\text{Total Charges}}{\max(\text{Tenure}, 1)}$
2. **`Support_Intensity`**: Monthly support inquiries per month of account tenure.
3. **`Complaint_Rate`**: Frequency of formal complaints relative to tenure.
4. **`Payment_Failure_Rate`**: Ratio of failed payment attempts against total transactions.
5. **`Engagement_Score`**: Normalized 0–100 composite index combining login frequency, app usage, data consumption, and web portal visits.
6. **`Customer_Value`**: Transparent analytical proxy: $\text{Total Charges} + (\text{Monthly Charges} \times 12)$.
7. **`Value_Tier`**: Categorized into Low, Medium, High based on business thresholds.

---

## 🤖 Machine Learning & Model Evaluation

This binary classification task compares multiple supervised learning algorithms:
- **Logistic Regression** (Balanced class weights, baseline benchmark)
- **Random Forest Classifier** (Ensemble bagging with depth pruning)
- **XGBoost Classifier** (Gradient boosted trees with weighted positive class scaling)

### Selection Criterion
Models are selected based on **ROC-AUC** and **Churn Class Recall**. In commercial retention, missing a churned customer (False Negative) is significantly more expensive than reviewing an account that stays (False Positive).

---

## 🔬 Explainable AI (SHAP)

- **Global Feature Importance:** Ranks top features across the portfolio (Tenure, Contract Type, Monthly Charges, Satisfaction, Complaints).
- **Individual Customer Attribution:** Local SHAP waterfall charts illustrating which features pushed an individual customer's churn risk up or down.
- **Analytical Rule:** Language strictly adheres to *association and contribution* rather than claiming causal certainty.

---

## 💼 Retention Intelligence & Review Areas

Customer accounts are segmented into four strategic quadrants:
1. **High Risk + High Value (P1 - Critical):** High-touch account management outreach.
2. **High Risk + Medium Value (P2 - High):** Automated targeted incentive or pricing review.
3. **Medium Risk + High Value (P2 - High):** Proactive check-in and relationship review.
4. **High Risk + Low Value (P3 - Medium):** Digital re-engagement campaign.

### Rule-Based Review Areas:
- **Pricing / Plan Review:** High churn risk coinciding with high monthly charges.
- **Customer Support Review:** Multiple complaints or low satisfaction score.
- **Engagement Campaign:** Low engagement score / login frequency.
- **Payment Assistance:** Recurring payment failures or outstanding balance.
- **Contract Review:** Month-to-Month contract status.

---

## 💻 Installation & Local Execution

### Prerequisites
- Python 3.9+ (Python 3.9 – 3.12 supported)
- pip

### 1. Clone or Open Workspace
```bash
cd "c:\Users\Asus\Desktop\DS Customer churn project"
```

### 2. Activate Virtual Environment
```powershell
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Automated Test Suite
```bash
pytest tests/ -v
```

### 5. Launch the Dashboard
```bash
streamlit run app.py
```

The application will launch in your browser at `http://localhost:8501`.

---

## 🧪 Testing Coverage

The automated test suite verifies:
- `test_data.py`: Synthetic generation, unique customer IDs, missing value imputation, boundary correction, and flexible column renaming.
- `test_model.py`: Model training pipeline, holdout metrics calculation, and bounded probability generation ($0.0 \le p \le 1.0$).
- `test_calculations.py`: KPI accuracy, risk categorization thresholds, customer value proxies, and priority assignments.

---

## ⚠️ Limitations & Analytical Disclaimers

1. **Non-Causal Associations:** Machine learning models identify statistical correlations. Identifying high monthly charges as an important feature does not prove charges caused the churn.
2. **No Retention Guarantee:** Suggested business review areas are decision-support guidelines, not guaranteed remedies. Commercial interventions should always be tested with controlled A/B experiments.
3. **Revenue at Risk:** The analytical metric represents exposed run-rate revenue, not guaranteed financial write-offs.
