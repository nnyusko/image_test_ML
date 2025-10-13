import pandas as pd
import joblib
import os

# This script is designed to be run in the competition's isolated environment.
# It loads a pre-trained model and generates predictions on the test data provided by the server.

# --- Constants ---
# The mean age calculated from the original training data. Used to fill missing age values.
TRAIN_AGE_MEAN = 49.330078209759655

# The list of features the model was trained on, in the exact order.
MODEL_FEATURES = ['Age', 'TestDate', 'A1-1', 'A1-2', 'A1-3', 'A1-4', 'A2-1', 'A2-2', 'A2-3', 'A2-4', 'A3-1', 'A3-2', 'A3-3', 'A3-4', 'A3-5', 'A3-6', 'A3-7', 'A4-1', 'A4-2', 'A4-3', 'A4-4', 'A4-5', 'A5-1', 'A5-2', 'A5-3', 'A6-1', 'A7-1', 'A8-1', 'A8-2', 'A9-1', 'A9-2', 'A9-3', 'A9-4', 'A9-5', 'B1-1', 'B1-2', 'B1-3', 'B2-1', 'B2-2', 'B2-3', 'B3-1', 'B3-2', 'B4-1', 'B4-2', 'B5-1', 'B5-2', 'B6', 'B7', 'B8', 'B9-1', 'B9-2', 'B9-3', 'B9-4', 'B9-5', 'B10-1', 'B10-2', 'B10-3', 'B10-4', 'B10-5', 'B10-6', 'Test_B']

# --- Data Loading ---
# The script assumes the data is in a directory structure like:
# /data/
#   test.csv
#   test/
#     A.csv
#     B.csv
#   sample_submission.csv
# /code/
#   inference.py (this script)
#   stacking_model.pkl
# We use relative paths to navigate from the code directory to the data directory.

print("Loading test data...")
# The base path for data is assumed to be one level up from the code directory
base_data_path = '../data' 
test = pd.read_csv(os.path.join(base_data_path, 'test.csv'))
test_A = pd.read_csv(os.path.join(base_data_path, 'test', 'A.csv'))
test_B = pd.read_csv(os.path.join(base_data_path, 'test', 'B.csv'))
sample_submission = pd.read_csv(os.path.join(base_data_path, 'sample_submission.csv'))

# --- Preprocessing ---
print("Preprocessing data...")
# 1. Merge data
test_A_merged = pd.merge(test[test['Test'] == 'A'], test_A, on='Test_id')
test_B_merged = pd.merge(test[test['Test'] == 'B'], test_B, on='Test_id')
test_df = pd.concat([test_A_merged, test_B_merged], ignore_index=True)

# 2. Handle 'Age' column by removing non-numeric parts and converting to numeric
test_df['Age'] = test_df['Age'].str.replace('a', '').str.replace('b', '')
test_df['Age'] = pd.to_numeric(test_df['Age'], errors='coerce')
test_df['Age'].fillna(TRAIN_AGE_MEAN, inplace=True)

# 3. Convert other object columns to numeric, coercing errors to NaN
for col in test_df.columns:
    if test_df[col].dtype == 'object' and col not in ['Test_id', 'Test', 'PrimaryKey']:
        test_df[col] = pd.to_numeric(test_df[col], errors='coerce')

# 4. Fill all remaining NaNs with 0
test_df.fillna(0, inplace=True)

# 5. One-Hot Encode the 'Test' column
test_df = pd.get_dummies(test_df, columns=['Test'], drop_first=True)

# 6. Align columns with the model's training features
# Ensure all required features are present, fill with 0 if a feature was not in the test set
for col in MODEL_FEATURES:
    if col not in test_df.columns:
        test_df[col] = 0

# Select and reorder columns to match the exact order used for training
X_test = test_df[MODEL_FEATURES]

# --- Prediction ---
print("Loading model and making predictions...")
# Load the pre-trained model from the same directory as the script
model = joblib.load('stacking_model.pkl')

# Predict probabilities
predictions = model.predict_proba(X_test)[:, 1]

# --- Create Submission File ---
print("Creating submission file...")
# Create a dataframe with the predictions
submission_df = pd.DataFrame({'Test_id': test_df['Test_id'], 'Label': predictions})

# Merge with sample_submission to ensure all Test_ids are included in the correct order
final_submission = pd.merge(sample_submission[['Test_id']], submission_df, on='Test_id', how='left')

# Fill any potential missing predictions with a neutral value (0.5)
final_submission['Label'].fillna(0.5, inplace=True)

# Save the final submission file in the root directory
final_submission.to_csv('submission.csv', index=False)

print("Inference complete. submission.csv created successfully.")
