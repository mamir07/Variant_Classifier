import pandas as pd
import numpy as np # Add this import at the top of your file

# Load the preprocessed data
df = pd.read_csv('preprocessed_variants.csv')

# Display the first few rows to confirm it loaded correctly
print("Data loaded successfully.")
print(df.head())

# Feature Engineering

# Columns used for the model: REVEL, CADD, SIFT, Polyphen
feature_columns = [
    'REVEL',
    'SIFT',
    'PolyPhen',
    'Consequence'
]
label_column = 'LABEL'

# Smaller DataFrame with just these columns for simplicity
model_df = df[feature_columns + [label_column]].copy()


# FIX 1: Define the list of numeric columns to clean before the loop
numeric_cols = ['REVEL']

print("\n--- Cleaning numeric score columns ---")
# Loop through each column in the list
for col in numeric_cols:
    # FIX 2: All lines inside the loop must be indented
    # Convert the column to a numeric type. 'coerce' will turn any non-numeric text into NaN (a missing value)
    model_df[col] = pd.to_numeric(model_df[col], errors='coerce')

    # Calculate the median of the column (the middle value)
    median_value = model_df[col].median()

    # Fill any missing values (NaNs) with the median
    model_df[col] = model_df[col].fillna(median_value)

    print(f"'{col}' column cleaned. Missing values filled with median: {median_value:.4f}")

# --- Clean SIFT and PolyPhen scores ---
print("\n--- Cleaning SIFT and PolyPhen columns ---")

# This helper function extracts a number from a string like "prediction(score)"
def extract_score(score_string):
    if isinstance(score_string, str) and '(' in score_string:
        try:
            return float(score_string.split('(')[-1].replace(')', ''))
        except (ValueError, IndexError):
            return np.nan
    return np.nan

# Create new columns for the numeric scores by applying the function
model_df['SIFT_score'] = model_df['SIFT'].apply(extract_score)
model_df['PolyPhen_score'] = model_df['PolyPhen'].apply(extract_score)

# Fill missing values for the new score columns
sift_median = model_df['SIFT_score'].median()
model_df['SIFT_score'].fillna(sift_median, inplace=True)

polyphen_median = model_df['PolyPhen_score'].median()
model_df['PolyPhen_score'].fillna(polyphen_median, inplace=True)

print("SIFT and PolyPhen scores extracted and cleaned.")

# Display info to confirm the changes
print("\nInfo about the DataFrame after cleaning all scores:")
model_df.info()

# Handle Categorical Features
print("\n--- One-hot encoding the 'Consequence' feature ---")

# This automatically finds all categories in the 'Consequence' column and creates a new binary column for each one.
model_df = pd.get_dummies(model_df, columns=['Consequence'], prefix='Consequence')

print("Categorical features encoded.")
print(model_df.head())

# Create Final Feature Matrix (X) and Target Vector (y)

# 1. Create the feature matrix 'X' We drop the original text columns and the label column
X = model_df.drop(columns=['LABEL', 'SIFT', 'PolyPhen'])

# 2. Create the target vector 'y'
y = model_df['LABEL']

# 3. Save the final data for the next script
X.to_csv('features.csv', index=False)
y.to_csv('labels.csv', index=False)


# Display the final results
print("\n--- Final feature matrix X (first 5 rows): ---")
print(X.head())

print("\n--- Final target vector y (first 5 rows): ---")
print(y.head())

print(f"\nFeature matrix X has {X.shape[0]} rows and {X.shape[1]} columns.")
print("Feature engineering complete. Data saved to features.csv and labels.csv.")

# Terminal prompt to run script: python 2_feature_engineering.py