# Fraud Detection — Adey Innovations Inc.

> Improved detection of fraud cases for e-commerce and bank credit card transactions using machine learning and SHAP explainability.

---

## Project Overview

This project builds a **unified fraud detection capability** for two distinct transaction streams:

| Dataset | Description | Target |
|---|---|---|
| `Fraud_Data.csv` | E-commerce transactions with user, device & behavioral context | `class` |
| `creditcard.csv` | Bank credit card transactions with anonymized PCA features | `Class` |

Both datasets are highly imbalanced. The pipeline handles this using **SMOTE** on the training set only, and evaluates models using **AUC-PR** and **F1-Score** rather than raw accuracy.

---

## Repository Structure

```
fraud-detection/
├── .vscode/settings.json          # VS Code workspace settings
├── .github/workflows/
│   └── unittests.yml              # CI/CD — automated test runs
├── data/
│   ├── raw/                       # Original CSVs (gitignored)
│   └── processed/                 # Cleaned & feature-engineered data
├── notebooks/
│   ├── eda-fraud-data.ipynb       # Task 1 — E-commerce EDA
│   ├── eda-creditcard.ipynb       # Task 1 — Credit card EDA
│   ├── feature-engineering.ipynb  # Task 1 — Feature engineering & resampling
│   ├── modeling.ipynb             # Task 2 — Model training & evaluation
│   └── shap-explainability.ipynb  # Task 3 — SHAP analysis
├── src/
│   ├── data_loader.py             # Dataset loading utilities
│   ├── preprocessing.py           # Cleaning, IP→country, encoding, scaling
│   ├── feature_engineering.py     # Temporal, velocity, behavioral features
│   ├── modeling.py                # Model builders, training, evaluation
│   ├── resampling.py              # SMOTE, undersampling strategies
│   └── explainability.py          # SHAP wrapper utilities
├── scripts/
│   ├── run_preprocessing.py       # Full preprocessing pipeline
│   ├── run_modeling.py            # Full modeling pipeline
│   └── run_explainability.py      # SHAP analysis pipeline
├── tests/
│   ├── test_preprocessing.py      # Unit tests for preprocessing
│   └── test_feature_engineering.py # Unit tests for feature engineering
├── models/                        # Saved model artifacts (.pkl)
├── requirements.txt
└── README.md
```

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/fraud-detection.git
cd fraud-detection
```

### 2. Create a Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add Raw Data

Place your raw CSV files in `data/raw/`:
```
data/raw/Fraud_Data.csv
data/raw/IpAddress_to_Country.csv
data/raw/creditcard.csv
```

> ⚠️ The `data/` folder is gitignored. Do not commit raw or processed data.

---

## Running the Pipeline

### Option A — Scripts (recommended for reproducibility)

```bash
# Step 1: Preprocess both datasets
python scripts/run_preprocessing.py

# Step 2: Train and evaluate models
python scripts/run_modeling.py

# Step 3: Generate SHAP explanations
python scripts/run_explainability.py
```

### Option B — Jupyter Notebooks

```bash
jupyter notebook notebooks/
```

Run notebooks in this order:
1. `eda-fraud-data.ipynb`
2. `eda-creditcard.ipynb`
3. `feature-engineering.ipynb`
4. `modeling.ipynb`
5. `shap-explainability.ipynb`

---

## Running Tests

```bash
pytest tests/ -v --cov=src
```

---

## Task Summary

### Task 1 — Data Analysis & Preprocessing
- Cleaned both datasets (duplicates, missing values, dtype correction)
- Merged e-commerce data with IP-to-country ranges using integer range lookup
- Engineered features: `time_since_signup`, `hour_of_day`, `day_of_week`, `tx_count_24h`, `device_user_count`
- Applied **SMOTE** on training set to handle severe class imbalance
- One-hot encoded categoricals; StandardScaler applied to numerics

### Task 2 — Model Building & Training
- **Baseline:** Logistic Regression (`class_weight='balanced'`)
- **Ensemble:** XGBoost with `scale_pos_weight` tuned to imbalance ratio
- Evaluated with AUC-PR, F1-Score, Confusion Matrix
- Stratified 5-Fold Cross-Validation for reliable performance estimation
- **Best model: XGBoost** — higher AUC-PR on both datasets

### Task 3 — Model Explainability
- Built-in XGBoost feature importance (baseline comparison)
- SHAP Summary Plot — global feature importance across the test set
- SHAP Force Plots — individual explanations for TP, FP, FN cases
- 5 actionable business recommendations derived from SHAP insights

---

## Key Findings

| Feature | Dataset | Fraud Signal |
|---|---|---|
| `time_since_signup` | E-Commerce | ★★★ High — short gaps strongly predict fraud |
| `tx_count_24h` | E-Commerce | ★★★ High — burst velocity = suspicious |
| `V14` | Credit Card | ★★★ High — top SHAP driver |
| `country` | E-Commerce | ★★ Medium — certain geos show 3–5× fraud rate |
| `device_user_count` | E-Commerce | ★★ Medium — shared devices = risk signal |

---

## Business Recommendations

1. **Step-up auth for new users** — require OTP for purchases within 24h of signup
2. **Real-time velocity limits** — flag users with 3+ transactions in 1 hour
3. **Geographic risk tiers** — auto-hold: high-risk country + new account + large purchase
4. **Monitor V14/V4 anomalies** — work with card networks to identify real-world proxies
5. **Device fingerprint alerts** — flag devices linked to more than 2 distinct user accounts

---

## Key Dates

| Milestone | Date |
|---|---|
| Interim-1 (Task 1) | Sunday, 07 Jun 2026 |
| Interim-2 (Task 2) | Sunday, 14 Jun 2026 |
| Final Submission | Tuesday, 16 Jun 2026 |

---

## Tech Stack

`Python 3.10+` · `pandas` · `scikit-learn` · `XGBoost` · `imbalanced-learn` · `SHAP` · `matplotlib` · `seaborn`
