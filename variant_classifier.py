import pandas as pd
import numpy as np
import pysam
import joblib
import argparse


def process_vcf(vcf_path):
    """Parses a VCF file into a pandas DataFrame."""
    print(f"--- Processing file: {vcf_path} ---")
    all_variants_data = []
    vcf_file = pysam.VariantFile(vcf_path)

    try:
        csq_fields = vcf_file.header.info['CSQ'].description.split("Format: ")[1].split('|')
    except (KeyError, IndexError):
        return pd.DataFrame()  # Return empty DataFrame on error

    for record in vcf_file:
        try:
            for csq_entry in record.info['CSQ']:
                csq_data = dict(zip(csq_fields, csq_entry.split('|')))
                variant_info = {
                    'CHROM': record.chrom,
                    'POS': record.pos,
                    'REF': record.ref,
                    'ALT': record.alts[0]
                }
                variant_info.update(csq_data)
                all_variants_data.append(variant_info)
        except KeyError:
            continue

    vcf_file.close()
    return pd.DataFrame(all_variants_data)


def extract_score(score_string):
    """Helper function to extract numeric scores from text fields."""
    if isinstance(score_string, str) and '(' in score_string:
        try:
            return float(score_string.split('(')[-1].replace(')', ''))
        except (ValueError, IndexError):
            return np.nan
    return np.nan


def engineer_features(df, model_features):
    """Applies the feature engineering steps to the DataFrame."""
    print("--- Engineering features ---")

    # Ensure all required base columns exist
    required_cols = ['REVEL', 'SIFT', 'PolyPhen', 'Consequence']
    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan  # Add missing columns

    # 1. Clean numeric scores (only REVEL is used directly)
    df['REVEL'] = pd.to_numeric(df['REVEL'], errors='coerce')
    # Note: We don't impute here, as we should use stats from the training set if needed,
    # or let the model handle NaNs if it can. For simplicity, we'll let NaNs pass through for now.

    # 2. Extract complex scores
    df['SIFT_score'] = df['SIFT'].apply(extract_score)
    df['PolyPhen_score'] = df['PolyPhen'].apply(extract_score)

    # 3. One-hot encode Consequence
    df_encoded = pd.get_dummies(df, columns=['Consequence'], prefix='Consequence')

    # 4. Align columns with the trained model's features
    df_aligned = df_encoded.reindex(columns=model_features, fill_value=0)

    return df_aligned


def main():
    """Main function to run the prediction pipeline."""
    # Set up Command-Line Argument Parsing
    parser = argparse.ArgumentParser(description="Predict variant pathogenicity from a VCF file.")
    parser.add_argument('--vcf', type=str, required=True, help="Path to the input VEP-annotated VCF file.")
    parser.add_argument('--model', type=str, default='rf_model.joblib', help="Path to the trained model file.")
    args = parser.parse_args()

    # Load Model and Feature Names
    print("--- Loading model and feature names ---")
    try:
        model = joblib.load(args.model)
        model_features = joblib.load('model_feature_names.joblib')
    except FileNotFoundError as e:
        print(f"Error: Could not load model or feature file. {e}")
        print("Please ensure the model and feature name files are in the same directory.")
        return

    # Process Input VCF
    raw_df = process_vcf(args.vcf)
    if raw_df.empty:
        print("Processing stopped due to an empty or invalid VCF file.")
        return

    # Engineer Features
    X_predict = engineer_features(raw_df.copy(), model_features)

    # Make Predictions
    print("--- Making predictions ---")
    predictions = model.predict(X_predict)
    prediction_proba = model.predict_proba(X_predict)[:, 1]  # Probability of being pathogenic

    # Create and Save Output
    output_df = raw_df[['CHROM', 'POS', 'REF', 'ALT']].copy()
    output_df['Prediction'] = ['Pathogenic' if pred == 1 else 'Benign' for pred in predictions]
    output_df['Pathogenic_Probability'] = prediction_proba

    output_filename = 'variant_classifier.csv'
    output_df.to_csv(output_filename, index=False)
    print(f"\nPredictions complete. Results saved to {output_filename}")


if __name__ == "__main__":
    main()

# Terminal prompt to run script: python variant_classifier --vcf(file path to vcf file)