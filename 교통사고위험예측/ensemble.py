import pandas as pd

# Load the submission files
baseline_df = pd.read_csv('C:/Users/user/Documents/GitHub/image_test_ML/교통사고위험예측/submit/baseline_submit.csv')
optuna_df = pd.read_csv('C:/Users/user/Documents/GitHub/image_test_ML/교통사고위험예측/submit/optuna_submit.csv')
xgboost_df = pd.read_csv('C:/Users/user/Documents/GitHub/image_test_ML/교통사고위험예측/submit/xgboost_submit.csv')

# Ensemble the predictions by averaging
ensemble_df = baseline_df.copy()
ensemble_df['Label'] = (baseline_df['Label'] + optuna_df['Label'] + xgboost_df['Label']) / 3

# Save the ensemble submission file
ensemble_df.to_csv('C:/Users/user/Documents/GitHub/image_test_ML/교통사고위험예측/submit/ensemble_submit.csv', index=False)

print("Ensemble submission file created successfully!")
