import pandas as pd

# Load the ACP2 dataset (replace 'file_path_acp2' with your actual file path)
file_path_acp2 = 'ACP2_main_test.tsv'
acp2_data = pd.read_csv(file_path_acp2, sep='\t')

# Extract the 'text' column from the ACP2 dataset and rename it to 'sequence'
sequence_data = acp2_data[['text']].rename(columns={'text': 'sequence'})

# Save the modified dataset as CSV with only one column named 'sequence' (replace 'output_path' with your desired output path)
output_path = 'modified_input_sequence.csv'
sequence_data.to_csv(output_path, index=False)

print(f"Modified dataset saved as: {output_path}")
