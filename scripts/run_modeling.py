"""
scripts/run_modeling.py
------------------------
Train, evaluate, and save models for both datasets.
Run from the project root: python scripts/run_modeling.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.modeling import (
    build_logistic_regression, build_xgboost,
    train_and_evaluate, cross_val_evaluate,
    save_model, compare_models
)
from src.resampling import apply_smote

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"


def run_fraud_models():
    print("\n" + "="*60)
    print("MODELING: E-Commerce Fraud Data")
    print("="*60)

    X_train = pd.read_csv(PROCESSED / "fraud_X_train.csv")
    y_train = pd.read_csv(PROCESSED / "fraud_y_train.csv").squeeze()
    X_test  = pd.read_csv(PROCESSED / "fraud_X_test.csv")
    y_test  = pd.read_csv(PROCESSED / "fraud_y_test.csv").squeeze()

    # Logistic Regression baseline
    lr = build_logistic_regression()
    lr_m = train_and_evaluate(lr, X_train, y_train, X_test, y_test, "LogisticRegression (Fraud)")
    save_model(lr, "lr_fraud")

    # XGBoost
    spw = (y_train == 0).sum() / (y_train == 1).sum()
    xgb = build_xgboost(scale_pos_weight=spw)
    xgb_m = train_and_evaluate(xgb, X_train, y_train, X_test, y_test, "XGBoost (Fraud)")
    save_model(xgb, "xgb_fraud")

    # Cross-validation
    X_all = pd.concat([X_train, X_test])
    y_all = pd.concat([y_train, y_test])
    print("\n--- CV: Logistic Regression (Fraud) ---")
    cross_val_evaluate(build_logistic_regression(), X_all, y_all, k=5)
    print("\n--- CV: XGBoost (Fraud) ---")
    cross_val_evaluate(build_xgboost(scale_pos_weight=spw), X_all, y_all, k=5)

    print("\n--- Fraud Model Comparison ---")
    compare_models([lr_m, xgb_m])


def run_creditcard_models():
    print("\n" + "="*60)
    print("MODELING: Credit Card Transactions")
    print("="*60)

    cc_df = pd.read_csv(PROCESSED / "creditcard_cleaned.csv")
    X = cc_df.drop(columns=["Class"])
    y = cc_df["Class"]

    scaler = StandardScaler()
    X[["Time", "Amount"]] = scaler.fit_transform(X[["Time", "Amount"]])

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_tr_res, y_tr_res = apply_smote(X_tr, y_tr)

    spw = (y_tr == 0).sum() / (y_tr == 1).sum()

    lr_cc = build_logistic_regression()
    lr_m  = train_and_evaluate(lr_cc, X_tr_res, y_tr_res, X_te, y_te, "LogisticRegression (CC)")
    save_model(lr_cc, "lr_creditcard")

    xgb_cc = build_xgboost(scale_pos_weight=spw)
    xgb_m  = train_and_evaluate(xgb_cc, X_tr_res, y_tr_res, X_te, y_te, "XGBoost (CC)")
    save_model(xgb_cc, "xgb_creditcard")

    print("\n--- Credit Card Model Comparison ---")
    compare_models([lr_m, xgb_m])


if __name__ == "__main__":
    run_fraud_models()
    run_creditcard_models()
    print("\nAll modeling pipelines finished successfully.")
