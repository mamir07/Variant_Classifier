import pandas as pd
import numpy as np # Add this import at the top of your file

# Load the preprocessed data
df = pd.read_csv('preprocessed_variants.csv')

# Display the first few rows to confirm it loaded correctly
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

model_df = df[feature_columns + [label_column]].copy()


numeric_cols = ['REVEL']
for col in numeric_cols:
    model_df[col] = pd.to_numeric(model_df[col], errors='coerce')

    # Calculate the median of the column
    median_value = model_df[col].median()

    # Fill any missing values (NaNs) with the median
    model_df[col] = model_df[col].fillna(median_value)

    print(f"'{col}' column cleaned. Missing values filled with median: {median_value:.4f}")

# Clean SIFT and PolyPhen scores
def extract_score(score_string):
    if isinstance(score_string, str) and '(' in score_string:
        try:
            return float(score_string.split('(')[-1].replace(')', ''))
        except (ValueError, IndexError):
            return np.nan
    return np.nan


model_df['SIFT_score'] = model_df['SIFT'].apply(extract_score)
model_df['PolyPhen_score'] = model_df['PolyPhen'].apply(extract_score)


sift_median = model_df['SIFT_score'].median()
model_df['SIFT_score'].fillna(sift_median, inplace=True)

polyphen_median = model_df['PolyPhen_score'].median()
model_df['PolyPhen_score'].fillna(polyphen_median, inplace=True)


# Display info to confirm the changes
print("\nInfo about the DataFrame after cleaning all scores:")
model_df.info()

# Handle Categorical Features with One Hot Encoding
model_df = pd.get_dummies(model_df, columns=['Consequence'], prefix='Consequence')
print(model_df.head())

# Create Final Feature Matrix (X) and Target Vector (y)
X = model_df.drop(columns=['LABEL', 'SIFT', 'PolyPhen'])
y = model_df['LABEL']

# 3. Save the final data for the next script
X.to_csv('features.csv', index=False)
y.to_csv('labels.csv', index=False)


# Display the final results
print(X.head())
print(y.head())

print(f"\nFeature matrix X has {X.shape[0]} rows and {X.shape[1]} columns.")


# Terminal prompt to run script: python 2_feature_engineering.py
