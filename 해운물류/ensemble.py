import pandas as pd
from scipy.stats import mode

# Load all submission files
rf_preds = pd.read_csv('baseline_submit.csv')
lgbm_preds = pd.read_csv('lightgbm_submit.csv')
lgbm_tuned_preds = pd.read_csv('lgbm_tuned_submit.csv')
xgb_preds = pd.read_csv('xgboost_submit.csv')

# Combine the predictions into a single DataFrame
ensemble_df = pd.DataFrame({
    'ID': rf_preds['ID'],
    'rf': rf_preds['target'],
    'lgbm': lgbm_preds['target'],
    'lgbm_tuned': lgbm_tuned_preds['target'],
    'xgb': xgb_preds['target']
})

# Calculate the mode (most frequent prediction) for each row
predictions = ensemble_df.drop('ID', axis=1).apply(lambda x: mode(x, keepdims=True)[0][0], axis=1)

# Create the final submission file
submission = pd.read_csv('assets/sample_submission.csv')
submission['target'] = predictions
submission.to_csv('./ensemble_submit.csv', index=False, encoding='utf-8-sig')

print("ensemble_submit.csv 파일 생성이 완료되었습니다.")
