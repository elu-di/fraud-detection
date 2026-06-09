"""
tests/test_feature_engineering.py
-----------------------------------
Unit tests for feature engineering utilities.
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.feature_engineering import (
    add_time_features, add_time_since_signup,
    add_transaction_velocity, add_device_frequency
)


def make_sample_df():
    return pd.DataFrame({
        "user_id": [1, 1, 2],
        "device_id": ["d1", "d1", "d2"],
        "signup_time": pd.to_datetime(["2023-01-01 08:00", "2023-01-01 08:00", "2023-01-02 10:00"]),
        "purchase_time": pd.to_datetime(["2023-01-01 10:00", "2023-01-01 12:00", "2023-01-02 11:00"]),
        "class": [0, 1, 0],
    })


class TestTimeFeatures:
    def test_hour_of_day(self):
        df = make_sample_df()
        result = add_time_features(df)
        assert "hour_of_day" in result.columns
        assert result["hour_of_day"].iloc[0] == 10

    def test_day_of_week(self):
        df = make_sample_df()
        result = add_time_features(df)
        assert "day_of_week" in result.columns
        assert 0 <= result["day_of_week"].iloc[0] <= 6


class TestTimeSinceSignup:
    def test_positive_values(self):
        df = make_sample_df()
        result = add_time_since_signup(df)
        assert "time_since_signup" in result.columns
        assert (result["time_since_signup"] >= 0).all()

    def test_correct_hours(self):
        df = make_sample_df()
        result = add_time_since_signup(df)
        assert result["time_since_signup"].iloc[0] == pytest.approx(2.0)


class TestTransactionVelocity:
    def test_column_added(self):
        df = make_sample_df()
        result = add_transaction_velocity(df, window_hours=24)
        assert "tx_count_24h" in result.columns

    def test_first_transaction_is_zero(self):
        df = make_sample_df()
        result = add_transaction_velocity(df, window_hours=24)
        # First tx for each user should be 0
        user1 = result[result["user_id"] == 1].sort_values("purchase_time")
        assert user1["tx_count_24h"].iloc[0] == 0


class TestDeviceFrequency:
    def test_column_added(self):
        df = make_sample_df()
        result = add_device_frequency(df)
        assert "device_user_count" in result.columns

    def test_shared_device(self):
        df = make_sample_df()
        result = add_device_frequency(df)
        # Both rows with d1 share the same device
        d1_rows = result[result["device_id"] == "d1"]
        assert (d1_rows["device_user_count"] == 1).all()
