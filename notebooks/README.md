# Notebooks

This directory contains all Jupyter notebooks for the fraud detection project, organized by task.

| Notebook | Purpose |
|---|---|
| `eda-fraud-data.ipynb` | EDA on `Fraud_Data.csv` — distributions, imbalance, country analysis |
| `eda-creditcard.ipynb` | EDA on `creditcard.csv` — PCA features, Amount/Time patterns |
| `feature-engineering.ipynb` | Full feature engineering pipeline for e-commerce data |
| `modeling.ipynb` | Model training and evaluation for both datasets |
| `shap-explainability.ipynb` | SHAP global and individual explanations |

## Running Notebooks

```bash
# From the project root
jupyter notebook notebooks/
```

## Order of Execution

1. `eda-fraud-data.ipynb`
2. `eda-creditcard.ipynb`
3. `feature-engineering.ipynb`
4. `modeling.ipynb`
5. `shap-explainability.ipynb`
