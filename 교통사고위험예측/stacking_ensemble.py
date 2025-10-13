import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb

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

# Define the base models
estimators = [
    ('lgbm', lgb.LGBMClassifier(random_state=42)),
    ('xgb', xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='auc'))
]

# Define the meta-model
meta_model = LogisticRegression()

# Initialize the StackingClassifier
stacking_clf = StackingClassifier(
    estimators=estimators, final_estimator=meta_model, cv=5, passthrough=True
)

print("Training Stacking Ensemble model...")
stacking_clf.fit(X, y)

# Save the trained model
import joblib
joblib.dump(stacking_clf, 'C:/Users/user/Documents/GitHub/image_test_ML/교통사고위험예측/stacking_model.pkl')
print("Model saved to stacking_model.pkl")

# Make predictions on the test data
print("Making predictions on the test data...")
test_preds = stacking_clf.predict_proba(X_test)[:, 1]

# Create the submission file
submission_df = pd.DataFrame({'Test_id': test_df['Test_id'], 'Label': test_preds})
final_submission = pd.merge(sample_submission[['Test_id']], submission_df, on='Test_id', how='left')
final_submission['Label'].fillna(0.5, inplace=True)

final_submission.to_csv('C:/Users/user/Documents/GitHub/image_test_ML/교통사고위험예측/submit/stacking_submit.csv', index=False)

print("Stacking ensemble submission file created successfully!")
