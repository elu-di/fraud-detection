"""
scripts/run_preprocessing.py
-----------------------------
End-to-end data cleaning, feature engineering, and resampling pipeline.
Run from the project root: python scripts/run_preprocessing.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

from src.data_loader import load_fraud_data, load_ip_country, load_creditcard, basic_info
from src.preprocessing import (
    drop_duplicates, handle_missing, merge_ip_country,
    scale_features, encode_categoricals
)
from src.feature_engineering import engineer_all_features
from src.resampling import get_class_distribution, apply_smote


PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
MODELS    = Path(__file__).resolve().parents[1] / "models"
PROCESSED.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)


def run_fraud_pipeline():
    print("\n" + "="*60)
    print("PIPELINE: E-Commerce Fraud Data")
    print("="*60)

    # Load
    fraud_df = load_fraud_data()
    ip_df    = load_ip_country()
    basic_info(fraud_df, "Fraud_Data")

    # Clean
    fraud_df = drop_duplicates(fraud_df)
    fraud_df = handle_missing(fraud_df, strategy="drop")

    # Geo-enrich
    fraud_df = merge_ip_country(fraud_df, ip_df)

    # Feature engineering
    fraud_df = engineer_all_features(fraud_df, window_hours=24)

    # Drop non-model columns
    drop_cols = ["user_id", "device_id", "signup_time", "purchase_time",
                 "ip_address", "ip_int"]
    fraud_df = fraud_df.drop(columns=[c for c in drop_cols if c in fraud_df.columns])

    # Encode categoricals
    cat_cols = ["source", "browser", "sex", "country"]
    fraud_df = encode_categoricals(fraud_df, cols=cat_cols)

    # Split
    X = fraud_df.drop(columns=["class"])
    y = fraud_df["class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Scale
    num_cols = ["purchase_value", "age", "time_since_signup",
                "tx_count_24h", "device_user_count", "hour_of_day", "day_of_week"]
    num_cols = [c for c in num_cols if c in X_train.columns]
    X_train, scaler = scale_features(X_train, num_cols, method="standard")
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    # Class distribution before resampling
    get_class_distribution(y_train, "Train before SMOTE")

    # SMOTE (training only)
    X_train_res, y_train_res = apply_smote(X_train, y_train)
    get_class_distribution(y_train_res, "Train after SMOTE")

    # Save
    pd.DataFrame(X_train_res, columns=X_train.columns).to_csv(
        PROCESSED / "fraud_X_train.csv", index=False)
    pd.Series(y_train_res, name="class").to_csv(
        PROCESSED / "fraud_y_train.csv", index=False)
    X_test.to_csv(PROCESSED / "fraud_X_test.csv", index=False)
    y_test.to_csv(PROCESSED / "fraud_y_test.csv", index=False)
    joblib.dump(scaler, MODELS / "fraud_scaler.pkl")
    print("Fraud pipeline complete. Files saved to data/processed/")


def run_creditcard_pipeline():
    print("\n" + "="*60)
    print("PIPELINE: Credit Card Transactions")
    print("="*60)

    cc_df = load_creditcard()
    basic_info(cc_df, "creditcard")

    cc_df = drop_duplicates(cc_df)
    cc_df = handle_missing(cc_df, strategy="drop")

    cc_df.to_csv(PROCESSED / "creditcard_cleaned.csv", index=False)
    print("Creditcard pipeline complete. Files saved to data/processed/")


if __name__ == "__main__":
    run_fraud_pipeline()
    run_creditcard_pipeline()
    print("\nAll preprocessing pipelines finished successfully.")
