"""
resampling.py
-------------
Strategies for handling class imbalance on training data only.
"""
import pandas as pd
import numpy as np
from collections import Counter

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline


def get_class_distribution(y: pd.Series | np.ndarray, label: str = "") -> dict:
    """Print and return class counts and ratio."""
    counts = Counter(y)
    total = sum(counts.values())
    minority = counts.get(1, 0)
    majority = counts.get(0, 0)
    ratio = minority / majority if majority > 0 else float("inf")
    print(f"[class_distribution] {label}")
    print(f"  Legit (0): {majority:,} ({majority/total*100:.2f}%)")
    print(f"  Fraud (1): {minority:,} ({minority/total*100:.2f}%)")
    print(f"  Imbalance ratio: 1:{ratio:.0f}" if ratio < 1 else f"  Imbalance ratio: 1:{1/ratio:.0f}")
    return dict(counts)


def apply_smote(X_train, y_train, random_state: int = 42, k_neighbors: int = 5):
    """
    SMOTE oversampling — synthesizes new minority samples.
    Choice justification: SMOTE is preferred when the minority class
    has sufficient examples (>50) and we want to avoid information loss
    from undersampling the majority class.
    """
    sm = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print("[apply_smote] Before:", Counter(y_train))
    print("[apply_smote] After :", Counter(y_res))
    return X_res, y_res


def apply_undersampling(X_train, y_train, sampling_strategy: float = 0.5,
                        random_state: int = 42):
    """
    Random undersampling of the majority class.
    Choice justification: preferred on very large datasets where SMOTE
    is computationally prohibitive. sampling_strategy=0.5 means
    minority:majority = 1:2 after resampling.
    """
    rus = RandomUnderSampler(sampling_strategy=sampling_strategy,
                             random_state=random_state)
    X_res, y_res = rus.fit_resample(X_train, y_train)
    print("[apply_undersampling] Before:", Counter(y_train))
    print("[apply_undersampling] After :", Counter(y_res))
    return X_res, y_res


def apply_smote_tomek(X_train, y_train, random_state: int = 42):
    """
    Combined SMOTE + Tomek Links — oversample minority then clean boundary noise.
    Provides a cleaner decision boundary than SMOTE alone.
    """
    from imblearn.combine import SMOTETomek
    smt = SMOTETomek(random_state=random_state)
    X_res, y_res = smt.fit_resample(X_train, y_train)
    print("[apply_smote_tomek] Before:", Counter(y_train))
    print("[apply_smote_tomek] After :", Counter(y_res))
    return X_res, y_res
