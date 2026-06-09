"""
data_loader.py
--------------
Utilities for loading and initial validation of raw datasets.
"""
import pandas as pd
import numpy as np
from pathlib import Path


RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def load_fraud_data(filepath: str | Path = None) -> pd.DataFrame:
    """Load Fraud_Data.csv and parse datetime columns."""
    path = filepath or RAW_DIR / "Fraud_Data.csv"
    df = pd.read_csv(path, parse_dates=["signup_time", "purchase_time"])
    print(f"[load_fraud_data] shape={df.shape}, cols={list(df.columns)}")
    return df


def load_ip_country(filepath: str | Path = None) -> pd.DataFrame:
    """Load IpAddress_to_Country.csv."""
    path = filepath or RAW_DIR / "IpAddress_to_Country.csv"
    df = pd.read_csv(path)
    print(f"[load_ip_country] shape={df.shape}")
    return df


def load_creditcard(filepath: str | Path = None) -> pd.DataFrame:
    """Load creditcard.csv."""
    path = filepath or RAW_DIR / "creditcard.csv"
    df = pd.read_csv(path)
    print(f"[load_creditcard] shape={df.shape}, cols={list(df.columns)}")
    return df


def basic_info(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """Print a structured summary of a dataframe."""
    print(f"\n{'='*60}")
    print(f"Dataset: {name}")
    print(f"Shape   : {df.shape}")
    print(f"Dtypes  :\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"\nDuplicates: {df.duplicated().sum()}")
    print(f"{'='*60}\n")
