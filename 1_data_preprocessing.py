import pysam
import pandas as pd
import os

data_folder = "data"
vcf_filenames = [
    "Horizon_HD701_Nova_R2_S16.dragen.concat_snv_sv.vep_an.vcf.gz",
    "Horizon_HD798_Nova_R2_S15.dragen.concat_snv_sv.vep_an.vcf.gz"
]

all_variants_data = []
for filename in vcf_filenames:
    vcf_path = os.path.join(data_folder, filename)
    vcf_file = pysam.VariantFile(vcf_path)
    csq_fields = vcf_file.header.info['CSQ'].description.split("Format: ")[1].split('|')

    for record in vcf_file:
        try:
            for csq_entry in record.info['CSQ']:
                csq_data = dict(zip(csq_fields, csq_entry.split('|')))

                sample_id = filename.split('.')[0]

                variant_info = {
                    'SampleID': sample_id,
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

# 3. Create a single DataFrame after the loop has finished
df = pd.DataFrame(all_variants_data)


print("\nFirst 5 rows of the combined DataFrame:")
print(df.head())
print("\nValue counts for 'ClinVar_CLNSIG' across both files:")
print(df['ClinVar_CLNSIG'].value_counts(dropna=False))

# Define which text labels correspond to pathogenic and benign
pathogenic_labels = [
    'Pathogenic',
    'Likely_pathogenic',
    'Pathogenic/Likely_pathogenic'
]
benign_labels = [
    'Benign',
    'Likely_benign',
    'Benign/Likely_benign'
]

# Create a new label column: 1 for pathogenic, 0 for benign, and None for others
df['LABEL'] = df['ClinVar_CLNSIG'].apply(
    lambda x: 1 if x in pathogenic_labels else (0 if x in benign_labels else None)
)

# Save cleaned dataframe into CSV file
df_cleaned = df.dropna(subset=['LABEL']).copy()
df_cleaned['LABEL'] = df_cleaned['LABEL'].astype(int)

print("\nValue counts for the new binary 'Label' column:")
print(df_cleaned['LABEL'].value_counts())

df_cleaned.to_csv('preprocessed_variants.csv', index=False)

# Terminal prompt to run script: python 1_data_preprocessing.py



