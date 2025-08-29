import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap
import numpy as np


def explain_model(model_name, model_filename, X_test):
    """Loads a single model and generates its SHAP bar plot."""
    print(f"\n--- Explaining {model_name} ---")

    # Load the trained model
    model = joblib.load(model_filename)

    # SHAP uses different "explainers" for different model types
    if model_name == "Logistic Regression":
        explainer = shap.LinearExplainer(model, X_test)
        shap_values = explainer.shap_values(X_test)
    else:
        # TreeExplainer is for tree-based models (Random Forest, Gradient Boosting)
        explainer = shap.TreeExplainer(model)
        shap_values_raw = explainer.shap_values(X_test)

        # THE FIX: Check if the output is a 3D array and slice it.
        # If it's already a 2D array (like for Gradient Boosting), use it directly.
        if isinstance(shap_values_raw, np.ndarray) and shap_values_raw.ndim == 3:
            shap_values = shap_values_raw[:, :, 1]  # Slice the 3D array
        else:
            shap_values = shap_values_raw  # Use the 2D array as is

    # Calculate the mean absolute value for each feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    # Create and save the plot
    feature_importance = pd.Series(mean_abs_shap, index=X_test.columns)
    feature_importance_sorted = feature_importance.sort_values(ascending=True)

    plt.figure(figsize=(10, 12))
    feature_importance_sorted.plot(kind='barh')

    plt.title(f"SHAP Feature Importance ({model_name})")
    plt.xlabel("Mean Absolute SHAP Value")

    filename = f"{model_name.lower().replace(' ', '_')}_shap_summary.png"
    plt.savefig(filename, bbox_inches='tight')
    print(f"SHAP plot saved to {filename}")
    plt.close()


def main():
    """Main function to load test data and explain all models."""
    print("--- Loading test data ---")
    X_test = pd.read_csv('X_test.csv')

    # Dictionary of models to explain
    models_to_explain = {
        "Logistic Regression": "lr_model.joblib",
        "Random Forest": "rf_model.joblib",
        "Gradient Boosting": "gb_model.joblib"
    }

    # Loop through and explain each model
    for name, filename in models_to_explain.items():
        explain_model(name, filename, X_test)


if __name__ == "__main__":
    main()

# Terminal prompt to run script: python 4_model_explanation.py