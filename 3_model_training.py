import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score, RocCurveDisplay



def evaluate_model(model, model_name, X_test, y_test):
    """Makes predictions, evaluates the model, and saves confusion matrix and ROC curve plots."""

    # --- Standard Predictions and Accuracy ---
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    # --- Confusion Matrix ---
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Benign', 'Pathogenic'])
    disp.plot()
    plt.title(f'{model_name} Confusion Matrix')
    cm_filename = f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png"
    plt.savefig(cm_filename)
    print(f"Confusion matrix plot saved to {cm_filename}")
    plt.close()  # Close the plot to prepare for the next one

    # --- NEW: ROC Curve and AUC Score ---
    # Get the probability scores for the positive class (Pathogenic)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate the AUC score
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"AUC Score: {auc:.4f}")

    # Plot the ROC curve
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title(f'{model_name} ROC Curve')

    # Save the ROC curve plot
    #roc_filename = f"{model_name.lower().replace(' ', '_')}_roc_curve.png"
    #plt.savefig(roc_filename)
    #print(f"ROC curve plot saved to {roc_filename}")
    #plt.close()  # Close the plot


def main():
    """Main function to run the model training and evaluation pipeline."""
    # --- 1. Load the Data ---
    print("Loading engineered features and labels...")
    X = pd.read_csv('features.csv')
    y = pd.read_csv('labels.csv').values.ravel()

    # --- 2. Split the Data ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Data partitioned. Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

    # --- 3. Train and Evaluate Logistic Regression ---
    print("\n--- Training Logistic Regression model ---")
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train, y_train)
    print("Logistic Regression model trained successfully.")
    evaluate_model(lr_model, "Logistic Regression", X_test, y_test)

    # --- 4. Train and Evaluate Random Forest ---
    print("\n--- Training Random Forest model ---")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    print("Random Forest model trained successfully.")
    evaluate_model(rf_model, "Random Forest", X_test, y_test)

    # --- 5. Train and Evaluate Gradient Boosting ---
    print("\n--- Training Gradient Boosting model ---")
    gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb_model.fit(X_train, y_train)
    print("Gradient Boosting model trained successfully.")
    evaluate_model(gb_model, "Gradient Boosting", X_test, y_test)

    # --- 7. Save All Models and Test Data ---
    print("\n--- Saving all models and test data ---")

    # Save each trained model to a file
    joblib.dump(lr_model, 'lr_model.joblib')
    joblib.dump(rf_model, 'rf_model.joblib')
    joblib.dump(gb_model, 'gb_model.joblib')

    # Save the test data
    X_test.to_csv('X_test.csv', index=False)
    y_test_df = pd.DataFrame(y_test, columns=['LABEL'])
    y_test_df.to_csv('y_test.csv', index=False)

    # Save the list of feature columns the model was trained on
    joblib.dump(X_train.columns.tolist(), 'model_feature_names.joblib')

if __name__ == "__main__":
    main()


# Terminal prompt to run script: python 3_model_training.py