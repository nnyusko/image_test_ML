import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import cross_val_score, StratifiedKFold

print("Optuna Hyperparameter Tuning Script Started...")

# --- 1. 데이터 로드 ---
print("Step 1: Loading original data...")
try:
    train_df = pd.read_csv('assets/train.csv')
    test_df = pd.read_csv('assets/test.csv')
    submission = pd.read_csv('assets/sample_submission.csv')
    print("Data loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    exit()

# 데이터 준비
TARGET = 'target'
features = [col for col in train_df.columns if col not in ['ID', TARGET]]
X = train_df[features]
y = train_df[TARGET]
X_test = test_df[features]

# --- 2. Optuna Objective 함수 정의 ---
print("Step 2: Defining Optuna objective function...")
def objective(trial):
    # 하이퍼파라미터 탐색 공간 정의
    params = {
        'objective': 'multiclass',
        'num_class': 21,
        'metric': 'multi_logloss',
        'random_state': 42,
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
    }
    
    model = lgb.LGBMClassifier(**params)
    
    # 교차 검증으로 F1 스코어 계산
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring='f1_macro', n_jobs=-1)
    
    return np.mean(scores)

# --- 3. Optuna Study 실행 ---
N_TRIALS = 100
print(f"Step 3: Starting Optuna study for {N_TRIALS} trials...")

# 'maximize'를 위해 새로운 study 생성
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=N_TRIALS)

# --- 4. 결과 출력 ---
print("\n--- Optuna Tuning Results ---")
print(f"Best F1-score: {study.best_value:.4f}")
print("Best hyperparameters:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")

# --- 5. 최적 모델로 제출 파일 생성 ---
print("\nStep 5: Training final model with best params and creating submission file...")
best_params = study.best_params
best_params['objective'] = 'multiclass'
best_params['num_class'] = 21
best_params['metric'] = 'multi_logloss'
best_params['random_state'] = 42

final_model = lgb.LGBMClassifier(**best_params)
final_model.fit(X, y)
predictions = final_model.predict(X_test)

submission['target'] = predictions
submission.to_csv('optuna_submit.csv', index=False)

print("\nOptuna tuning complete!")
print("Submission file 'optuna_submit.csv' has been created.")
