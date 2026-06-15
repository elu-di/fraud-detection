"""
modeling.py
-----------
Model training, evaluation, and comparison utilities.
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report, precision_recall_curve
)
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODELS_DIR.mkdir(exist_ok=True)


# ──────────────────────────────────────────────
# Model Builders
# ──────────────────────────────────────────────

def build_logistic_regression(**kwargs) -> LogisticRegression:
    defaults = dict(max_iter=1000, class_weight="balanced", random_state=42, n_jobs=-1)
    defaults.update(kwargs)
    return LogisticRegression(**defaults)


def build_random_forest(**kwargs) -> RandomForestClassifier:
    defaults = dict(n_estimators=200, max_depth=12, class_weight="balanced",
                    random_state=42, n_jobs=-1)
    defaults.update(kwargs)
    return RandomForestClassifier(**defaults)


def build_xgboost(scale_pos_weight: float = 1.0, **kwargs) -> XGBClassifier:
    defaults = dict(n_estimators=300, max_depth=6, learning_rate=0.05,
                    scale_pos_weight=scale_pos_weight,
                    use_label_encoder=False, eval_metric="aucpr",
                    random_state=42, n_jobs=-1, tree_method="hist")
    defaults.update(kwargs)
    return XGBClassifier(**defaults)


# ──────────────────────────────────────────────
# Training & Evaluation
# ──────────────────────────────────────────────

def train_and_evaluate(model, X_train, y_train, X_test, y_test, model_name: str) -> dict:
    """Train model and return a metrics dict."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    metrics = {
        "model": model_name,
        "f1": f1_score(y_test, y_pred),
        "auc_pr": average_precision_score(y_test, y_proba),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }

    print(f"\n{'='*50}")
    print(f"Model: {model_name}")
    print(f"  F1         : {metrics['f1']:.4f}")
    print(f"  AUC-PR     : {metrics['auc_pr']:.4f}")
    print(f"  ROC-AUC    : {metrics['roc_auc']:.4f}")
    print(f"  Confusion  :\n{metrics['confusion_matrix']}")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))
    return metrics


def cross_val_evaluate(model, X, y, k: int = 5) -> dict:
    """Stratified K-Fold cross-validation returning mean ± std metrics."""
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    scoring = ["f1", "average_precision", "roc_auc"]
    results = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)

    summary = {}
    for metric in scoring:
        key = f"test_{metric}"
        summary[metric] = {
            "mean": results[key].mean(),
            "std": results[key].std(),
        }
        print(f"  {metric:25s}: {results[key].mean():.4f} ± {results[key].std():.4f}")
    return summary


# ──────────────────────────────────────────────
# Saving / Loading
# ──────────────────────────────────────────────

def save_model(model, name: str) -> Path:
    path = MODELS_DIR / f"{name}.pkl"
    joblib.dump(model, path)
    print(f"[save_model] saved -> {path}")
    return path


def load_model(name: str):
    path = MODELS_DIR / f"{name}.pkl"
    model = joblib.load(path)
    print(f"[load_model] loaded <- {path}")
    return model


# ──────────────────────────────────────────────
# Visualizations
# ──────────────────────────────────────────────

def plot_confusion_matrix(cm: np.ndarray, title: str = "Confusion Matrix", ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Legit", "Fraud"],
                yticklabels=["Legit", "Fraud"], ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    plt.tight_layout()


def plot_precision_recall_curve(y_test, y_proba, model_name: str, ax=None):
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(recall, precision, label=f"{model_name} (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    plt.tight_layout()


def compare_models(metrics_list: list) -> pd.DataFrame:
    """Build a side-by-side comparison table from a list of metrics dicts."""
    rows = [
        {
            "Model": m["model"],
            "F1": round(m["f1"], 4),
            "AUC-PR": round(m["auc_pr"], 4),
            "ROC-AUC": round(m["roc_auc"], 4),
        }
        for m in metrics_list
    ]
    df = pd.DataFrame(rows).sort_values("AUC-PR", ascending=False).reset_index(drop=True)
    print("\nModel Comparison:\n", df.to_string(index=False))
    return df
