"""
scripts/run_explainability.py
------------------------------
Generate SHAP plots and top feature importance for saved models.
Run from the project root: python scripts/run_explainability.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.modeling import load_model
from src.explainability import (
    get_shap_explainer, compute_shap_values,
    plot_shap_summary, plot_shap_bar, get_top_features
)

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"


def explain_fraud_model():
    print("\n" + "="*60)
    print("SHAP EXPLAINABILITY: E-Commerce Fraud Model")
    print("="*60)

    xgb = load_model("xgb_fraud")
    X_test = pd.read_csv(PROCESSED / "fraud_X_test.csv")
    sample = X_test.sample(min(2000, len(X_test)), random_state=42)

    explainer   = get_shap_explainer(xgb, sample, model_type="tree")
    shap_values = compute_shap_values(explainer, sample)

    plot_shap_summary(shap_values, sample, title="SHAP Summary — E-Commerce Fraud")
    plot_shap_bar(shap_values, sample, title="Mean |SHAP| — E-Commerce Fraud", top_n=15)

    top = get_top_features(shap_values, list(sample.columns), top_n=10)
    print("\nTop 10 SHAP features (E-Commerce Fraud):")
    print(top.to_string(index=False))


def explain_creditcard_model():
    print("\n" + "="*60)
    print("SHAP EXPLAINABILITY: Credit Card Model")
    print("="*60)

    xgb   = load_model("xgb_creditcard")
    cc_df = pd.read_csv(PROCESSED / "creditcard_cleaned.csv")
    X     = cc_df.drop(columns=["Class"])
    y     = cc_df["Class"]

    scaler = StandardScaler()
    X[["Time", "Amount"]] = scaler.fit_transform(X[["Time", "Amount"]])
    _, X_test, _, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    sample = X_test.sample(min(2000, len(X_test)), random_state=42)

    explainer   = get_shap_explainer(xgb, sample, model_type="tree")
    shap_values = compute_shap_values(explainer, sample)

    plot_shap_summary(shap_values, sample, title="SHAP Summary — Credit Card Fraud")
    plot_shap_bar(shap_values, sample, title="Mean |SHAP| — Credit Card Fraud", top_n=15)

    top = get_top_features(shap_values, list(sample.columns), top_n=10)
    print("\nTop 10 SHAP features (Credit Card):")
    print(top.to_string(index=False))


if __name__ == "__main__":
    explain_fraud_model()
    explain_creditcard_model()
    print("\nSHAP explainability analysis complete.")
