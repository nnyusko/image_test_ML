import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Load the cleaned data
train_df = pd.read_csv('C:/Users/user/Documents/GitHub/image_test_ML/교통사고위험예측/data/train_cleaned.csv')
test_df = pd.read_csv('C:/Users/user/Documents/GitHub/image_test_ML/교통사고위험예측/data/test_cleaned.csv')
sample_submission = pd.read_csv('C:/Users/user/Documents/GitHub/image_test_ML/교통사고위험예측/data/sample_submission.csv')

# Feature Engineering - One-Hot Encode 'Test' column
train_df = pd.get_dummies(train_df, columns=['Test'], drop_first=True)
test_df = pd.get_dummies(test_df, columns=['Test'], drop_first=True)

# Define features (X) and target (y)
features = [col for col in train_df.columns if col not in ['Test_id', 'PrimaryKey', 'Label']]
X = train_df[features]
y = train_df['Label']
X_test = test_df[features]

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize and train the XGBoost model
xgb_clf = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='auc', early_stopping_rounds=100)

print("Training XGBoost model...")
xgb_clf.fit(X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=True)

# Evaluate the model
val_preds = xgb_clf.predict_proba(X_val)[:, 1]
auc_score = roc_auc_score(y_val, val_preds)
print(f'Validation ROC AUC Score: {auc_score}')

# Make predictions on the test data
print("Making predictions on the test data...")
test_preds = xgb_clf.predict_proba(X_test)[:, 1]

# Create the submission file
submission_df = pd.DataFrame({'Test_id': test_df['Test_id'], 'Label': test_preds})
final_submission = pd.merge(sample_submission[['Test_id']], submission_df, on='Test_id', how='left')
final_submission['Label'].fillna(0.5, inplace=True)

final_submission.to_csv('C:/Users/user/Documents/GitHub/image_test_ML/교통사고위험예측/submit/xgboost_submit.csv', index=False)

print("Submission file created successfully!")