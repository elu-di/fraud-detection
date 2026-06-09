"""
preprocessing.py
----------------
Data cleaning, geolocation merge, and transformation utilities.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# ──────────────────────────────────────────────
# Cleaning
# ──────────────────────────────────────────────

def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    print(f"[drop_duplicates] removed {before - len(df)} rows")
    return df


def handle_missing(df: pd.DataFrame, strategy: str = "drop") -> pd.DataFrame:
    """Handle missing values. strategy: 'drop' | 'median' | 'mode'"""
    missing = df.isnull().sum()
    print(f"[handle_missing] missing before:\n{missing[missing > 0]}")
    if strategy == "drop":
        df = df.dropna()
    elif strategy == "median":
        num_cols = df.select_dtypes(include="number").columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    elif strategy == "mode":
        for col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])
    print(f"[handle_missing] missing after: {df.isnull().sum().sum()}")
    return df


# ──────────────────────────────────────────────
# IP → Country Geolocation
# ──────────────────────────────────────────────

def ip_to_int(ip_str: str) -> int:
    """Convert dotted-decimal IP string to integer."""
    try:
        parts = str(ip_str).split(".")
        return sum(int(p) << (8 * (3 - i)) for i, p in enumerate(parts))
    except Exception:
        return -1


def merge_ip_country(fraud_df: pd.DataFrame, ip_df: pd.DataFrame) -> pd.DataFrame:
    """
    Range-based IP → country lookup using a sorted merge.
    Converts ip_address to int, then finds the matching range in ip_df.
    """
    fraud_df = fraud_df.copy()
    ip_df = ip_df.copy()

    fraud_df["ip_int"] = fraud_df["ip_address"].apply(ip_to_int)
    ip_df["lower_int"] = ip_df["lower_bound_ip_address"].apply(ip_to_int)
    ip_df["upper_int"] = ip_df["upper_bound_ip_address"].apply(ip_to_int)
    ip_df = ip_df.sort_values("lower_int").reset_index(drop=True)

    # Merge using searchsorted for efficiency
    lower_bounds = ip_df["lower_int"].values
    upper_bounds = ip_df["upper_int"].values
    countries = ip_df["country"].values

    def lookup_country(ip_int):
        idx = np.searchsorted(lower_bounds, ip_int, side="right") - 1
        if idx >= 0 and lower_bounds[idx] <= ip_int <= upper_bounds[idx]:
            return countries[idx]
        return "Unknown"

    fraud_df["country"] = fraud_df["ip_int"].apply(lookup_country)
    print(f"[merge_ip_country] country value counts (top 10):")
    print(fraud_df["country"].value_counts().head(10))
    return fraud_df


# ──────────────────────────────────────────────
# Scaling & Encoding
# ──────────────────────────────────────────────

def scale_features(df: pd.DataFrame, cols: list, method: str = "standard") -> tuple:
    """Scale numerical columns. Returns (df, fitted_scaler)."""
    scaler = StandardScaler() if method == "standard" else MinMaxScaler()
    df = df.copy()
    df[cols] = scaler.fit_transform(df[cols])
    return df, scaler


def encode_categoricals(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """One-hot encode specified categorical columns."""
    df = pd.get_dummies(df, columns=cols, drop_first=False)
    print(f"[encode_categoricals] shape after encoding: {df.shape}")
    return df
