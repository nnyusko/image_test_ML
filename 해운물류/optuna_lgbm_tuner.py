
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

# 1. 데이터 불러오기
print("데이터를 불러옵니다...")
try:
    train_df = pd.read_csv('data/train.csv')
except FileNotFoundError:
    print("train.csv 파일을 찾을 수 없습니다. 'assets' 폴더가 아닌 현재 디렉토리에 파일이 있는지 확인해주세요.")
    exit()

# 레이블 인코딩
le = LabelEncoder()
train_df['target'] = le.fit_transform(train_df['target'])

# 데이터 준비
X = train_df.drop(columns=['ID', 'target'])
y = train_df['target']

# 2. Optuna Objective 함수 정의
def objective(trial):
    # 하이퍼파라미터 탐색 공간 정의
    params = {
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'num_class': len(np.unique(y)),
        'random_state': 42,
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0, step=0.1),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0, step=0.1),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }

    # 교차 검증 설정
    N_SPLITS = 5
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    
    logloss_scores = []
    
    # Optuna 프루닝(가지치기) 콜백 설정
    pruning_callback = optuna.integration.LightGBMPruningCallback(trial, 'multi_logloss')

    for train_idx, val_idx in skf.split(X, y):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  eval_metric='multi_logloss', # eval_metric을 명시적으로 지정
                  callbacks=[lgb.early_stopping(10, verbose=False), pruning_callback])
        
        # Early stopping에 의해 기록된 최적 점수(logloss)를 가져옴
        best_score = model.best_score_['valid_0']['multi_logloss']
        logloss_scores.append(best_score)
        
    return np.mean(logloss_scores)

# 3. Optuna Study 실행
print("Optuna를 사용하여 LightGBM 하이퍼파라미터 튜닝을 시작합니다...")
# 프루닝을 위해 중간값 리포트가 필요하므로, pruner 설정
pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
study = optuna.create_study(direction='minimize', pruner=pruner)
study.optimize(objective, n_trials=50) # 시험 횟수를 50회로 늘림

# 4. 결과 출력
print("\n튜닝이 완료되었습니다.")
print(f"최소 LogLoss: {study.best_value:.4f}")
print("최적 하이퍼파라미터:")
for key, value in study.best_params.items():
    print(f"  - {key}: {value}")
