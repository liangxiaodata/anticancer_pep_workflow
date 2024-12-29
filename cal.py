import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score

# Load the prediction and ground truth data
prediction_file_path = 'output.csv'
ground_truth_file_path = 'ACP2_main_test.tsv'

# Load the prediction data
pred_data = pd.read_csv(prediction_file_path)
ground_truth_data = pd.read_csv(ground_truth_file_path,delimiter='\t')

# Ensure that sequence matches between prediction and ground truth data
# merged_data = pd.merge(pred_data, ground_truth_data, left_on='sequence', right_on='text', suffixes=('_pred', '_true'))

# Extract prediction probabilities and ground truth labels
# print(ground_truth_data)
y_true = ground_truth_data['label']
y_pred_prob = pred_data['prediction_probability']

# Convert probabilities to binary predictions with a threshold of 0.5
y_pred = (y_pred_prob >= 0.5).astype(int)

# Calculate classification metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
mcc = matthews_corrcoef(y_true, y_pred)
auc = roc_auc_score(y_true, y_pred_prob)

# Display the metrics
metrics = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1,
    "Matthews Correlation Coefficient (MCC)": mcc,
    "AUC": auc
}

print(metrics)
