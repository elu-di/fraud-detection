"""
feature_engineering.py
-----------------------
Temporal, behavioral, and velocity features for Fraud_Data.csv.
"""
import pandas as pd
import numpy as np


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add hour_of_day and day_of_week from purchase_time.
    Requires: purchase_time column (datetime).
    """
    df = df.copy()
    df["hour_of_day"] = df["purchase_time"].dt.hour
    df["day_of_week"] = df["purchase_time"].dt.dayofweek  # 0=Mon, 6=Sun
    print("[add_time_features] Added hour_of_day, day_of_week")
    return df


def add_time_since_signup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute time_since_signup in hours between signup_time and purchase_time.
    Short time-since-signup is a strong fraud signal.
    """
    df = df.copy()
    delta = df["purchase_time"] - df["signup_time"]
    df["time_since_signup"] = delta.dt.total_seconds() / 3600  # hours
    # Clip negatives (data errors)
    df["time_since_signup"] = df["time_since_signup"].clip(lower=0)
    print("[add_time_since_signup] min={:.2f}h, max={:.2f}h, median={:.2f}h".format(
        df["time_since_signup"].min(),
        df["time_since_signup"].max(),
        df["time_since_signup"].median()
    ))
    return df


def add_transaction_velocity(df: pd.DataFrame, window_hours: int = 24) -> pd.DataFrame:
    """
    For each transaction, count how many prior transactions the same user
    made in the preceding `window_hours` hours.
    Requires: user_id, purchase_time columns (purchase_time sorted ascending).
    """
    df = df.copy().sort_values(["user_id", "purchase_time"]).reset_index(drop=True)
    col_name = f"tx_count_{window_hours}h"

    counts = []
    for _, group in df.groupby("user_id"):
        times = group["purchase_time"].values
        cnt = []
        for i, t in enumerate(times):
            cutoff = t - np.timedelta64(window_hours * 3600, "s")
            n = np.sum((times[:i] >= cutoff) & (times[:i] < t))
            cnt.append(n)
        counts.extend(cnt)

    df[col_name] = counts
    print(f"[add_transaction_velocity] '{col_name}' added. max={df[col_name].max()}")
    return df


def add_device_frequency(df: pd.DataFrame) -> pd.DataFrame:
    """Count how many distinct users share the same device_id (device sharing = risk signal)."""
    device_user_counts = df.groupby("device_id")["user_id"].nunique().rename("device_user_count")
    df = df.merge(device_user_counts, on="device_id", how="left")
    print("[add_device_frequency] device_user_count: max={}".format(df["device_user_count"].max()))
    return df


def engineer_all_features(df: pd.DataFrame, window_hours: int = 24) -> pd.DataFrame:
    """Run the full feature engineering pipeline on Fraud_Data."""
    df = add_time_features(df)
    df = add_time_since_signup(df)
    df = add_transaction_velocity(df, window_hours=window_hours)
    df = add_device_frequency(df)
    return df
