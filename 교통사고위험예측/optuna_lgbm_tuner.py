import pandas as pd
import lightgbm as lgb
import optuna
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

def objective(trial):
    param = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
    }

    lgb_clf = lgb.LGBMClassifier(**param, random_state=42)
    lgb_clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='auc', callbacks=[lgb.early_stopping(100, verbose=False)])

    preds = lgb_clf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, preds)
    return auc

print("Running Optuna hyperparameter search...")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50) # Running for 50 trials for demonstration

print("Best trial:")
trial = study.best_trial
print(f"  Value: {trial.value}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# Train the final model with the best hyperparameters
print("\nTraining final model with best hyperparameters...")
best_params = trial.params
best_params['random_state'] = 42
final_model = lgb.LGBMClassifier(**best_params)
final_model.fit(X, y)

# Make predictions on the test data
print("Making predictions on the test data...")
test_preds = final_model.predict_proba(X_test)[:, 1]

# Create the submission file
submission_df = pd.DataFrame({'Test_id': test_df['Test_id'], 'Label': test_preds})
final_submission = pd.merge(sample_submission[['Test_id']], submission_df, on='Test_id', how='left')
final_submission['Label'].fillna(0.5, inplace=True)

final_submission.to_csv('C:/Users/user/Documents/GitHub/image_test_ML/교통사고위험예측/submit/optuna_submit.csv', index=False)

print("Submission file created successfully!")
