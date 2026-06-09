"""
tests/test_preprocessing.py
----------------------------
Unit tests for preprocessing utilities.
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocessing import ip_to_int, drop_duplicates, handle_missing, encode_categoricals


class TestIpToInt:
    def test_known_value(self):
        assert ip_to_int("192.168.1.1") == 3232235777

    def test_zeros(self):
        assert ip_to_int("0.0.0.0") == 0

    def test_max(self):
        assert ip_to_int("255.255.255.255") == 4294967295

    def test_invalid(self):
        assert ip_to_int("not_an_ip") == -1


class TestDropDuplicates:
    def test_removes_duplicates(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
        result = drop_duplicates(df)
        assert len(result) == 2

    def test_no_duplicates(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = drop_duplicates(df)
        assert len(result) == 3


class TestHandleMissing:
    def test_drop_strategy(self):
        df = pd.DataFrame({"a": [1, None, 3], "b": [4, 5, 6]})
        result = handle_missing(df, strategy="drop")
        assert len(result) == 2
        assert result.isnull().sum().sum() == 0

    def test_median_strategy(self):
        df = pd.DataFrame({"a": [1.0, None, 3.0]})
        result = handle_missing(df, strategy="median")
        assert result["a"].isnull().sum() == 0
        assert result["a"].iloc[1] == 2.0  # median of [1, 3]


class TestEncodeCategories:
    def test_one_hot_encoding(self):
        df = pd.DataFrame({"browser": ["Chrome", "Firefox", "Safari"]})
        result = encode_categoricals(df, cols=["browser"])
        assert "browser_Chrome" in result.columns
        assert "browser" not in result.columns
