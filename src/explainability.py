"""
explainability.py
-----------------
SHAP-based model interpretation utilities.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap


def get_shap_explainer(model, X_background, model_type: str = "tree"):
    """
    Create a SHAP explainer.
    model_type: 'tree' for RF/XGB/LGB, 'linear' for LogReg, 'kernel' for any.
    """
    if model_type == "tree":
        explainer = shap.TreeExplainer(model)
    elif model_type == "linear":
        explainer = shap.LinearExplainer(model, X_background)
    else:
        explainer = shap.KernelExplainer(model.predict_proba, X_background)
    return explainer


def compute_shap_values(explainer, X):
    """Compute SHAP values. Returns array of shape (n_samples, n_features)."""
    shap_values = explainer.shap_values(X)
    # For binary classifiers, TreeExplainer returns list [class0, class1]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    return shap_values


def plot_shap_summary(shap_values, X, title: str = "SHAP Summary Plot"):
    """Beeswarm summary plot — global feature importance."""
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X, show=False)
    plt.title(title, fontsize=14, pad=12)
    plt.tight_layout()
    import os
    os.makedirs("plots", exist_ok=True)
    filename = f"plots/{title.replace(' ', '_').replace('—', '-')}.png"
    plt.savefig(filename)
    print(f"Saved plot: {filename}")
    plt.close()


def plot_shap_bar(shap_values, X, title: str = "Mean |SHAP| Feature Importance", top_n: int = 15):
    """Bar chart of mean absolute SHAP values."""
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X, plot_type="bar", max_display=top_n, show=False)
    plt.title(title, fontsize=13)
    plt.tight_layout()
    import os
    os.makedirs("plots", exist_ok=True)
    filename = f"plots/{title.replace(' ', '_').replace('|', '').replace('—', '-')}.png"
    plt.savefig(filename)
    print(f"Saved plot: {filename}")
    plt.close()


def plot_force(explainer, shap_values_row, X_row, idx: int = 0,
               title: str = "SHAP Force Plot"):
    """Force plot for a single prediction (rendered inline in Jupyter)."""
    print(f"\n--- {title} (index={idx}) ---")
    shap.initjs()
    return shap.force_plot(
        explainer.expected_value if not isinstance(explainer.expected_value, list)
        else explainer.expected_value[1],
        shap_values_row,
        X_row
    )


def get_top_features(shap_values, feature_names: list, top_n: int = 10) -> pd.DataFrame:
    """Return a DataFrame of top_n features by mean |SHAP| importance."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs
    }).sort_values("mean_abs_shap", ascending=False).head(top_n)
    return importance_df
