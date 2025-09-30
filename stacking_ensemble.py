import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb

print("Stacking Ensemble Script Started...")

# --- 1. 데이터 로드 ---
print("Step 1: Loading Data...")
try:
    # 원본 데이터
    train_orig = pd.read_csv('assets/train.csv')
    test_orig = pd.read_csv('assets/test.csv')
    
    # 피처 엔지니어링된 데이터
    train_fe = pd.read_csv('assets/train_fe.csv')
    test_fe = pd.read_csv('assets/test_fe.csv')
    
    submission = pd.read_csv('assets/sample_submission.csv')
    print("Data loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    print("Please ensure train.csv, test.csv, train_fe.csv, and test_fe.csv are in the 'assets' directory.")
    exit()

# --- 2. 데이터 준비 ---
print("Step 2: Preparing Data...")
# 원본 데이터셋
TARGET = 'target'
orig_features = [col for col in train_orig.columns if col not in ['ID', TARGET]]
y_train = train_orig[TARGET]
X_train_orig = train_orig[orig_features]
X_test_orig = test_orig[orig_features]

# 피처 엔지니어링된 데이터셋
fe_features = [col for col in train_fe.columns if col not in ['ID', TARGET]]
X_train_fe = train_fe[fe_features]
X_test_fe = test_fe[fe_features]

# --- 3. 기본 모델 정의 ---
print("Step 3: Defining Base Models...")
# 모델 파라미터는 이전 스크립트들에서 성능이 검증된 값들을 사용
rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)
lgbm_tuned_model = lgb.LGBMClassifier(learning_rate=0.1, n_estimators=200, num_leaves=31, random_state=42, n_jobs=-1)
xgb_model = xgb.XGBClassifier(random_state=42, n_jobs=-1)
# FE 데이터용 모델 (별도 튜닝 안했으므로 기본값 사용)
lgbm_fe_model = lgb.LGBMClassifier(random_state=42, n_jobs=-1)

# 각 모델이 사용할 데이터셋을 지정
base_models = {
    'RandomForest': (rf_model, X_train_orig, X_test_orig),
    'LGBM_Tuned': (lgbm_tuned_model, X_train_orig, X_test_orig),
    'XGBoost': (xgb_model, X_train_orig, X_test_orig),
    'LGBM_FE': (lgbm_fe_model, X_train_fe, X_test_fe)
}

# --- 4. Out-of-Fold 예측 생성 ---
print("Step 4: Generating Out-of-Fold (OOF) predictions for Meta-Model training...")
NFOLDS = 5
skf = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=42)

# OOF 예측값을 저장할 배열 초기화
oof_train = np.zeros((len(train_orig), len(base_models)))
oof_test = np.zeros((len(test_orig), len(base_models)))

for i, (model_name, (model, X_train, X_test)) in enumerate(base_models.items()):
    print(f"  - Training model: {model_name}...")
    
    # 테스트 데이터에 대한 예측값을 저장할 배열
    test_preds_per_fold = np.zeros((len(test_orig), NFOLDS))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"    - Fold {fold+1}/{NFOLDS}")
        
        # 훈련/검증 데이터 분리
        X_train_fold, y_train_fold = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_val_fold, y_val_fold = X_train.iloc[val_idx], y_train.iloc[val_idx]
        
        # 모델 학습
        model.fit(X_train_fold, y_train_fold)
        
        # 검증 데이터 예측 (OOF 생성)
        val_preds = model.predict(X_val_fold)
        oof_train[val_idx, i] = val_preds
        
        # 테스트 데이터 예측
        test_preds_per_fold[:, fold] = model.predict(X_test)
        
    # 테스트 데이터 예측값들의 평균을 내어 최종 테스트 예측값으로 사용
    oof_test[:, i] = test_preds_per_fold.mean(axis=1)

print("OOF prediction generation complete.")

# --- 5. 메타 모델 학습 및 예측 ---
print("Step 5: Training Meta-Model and making final predictions...")
meta_model = LogisticRegression(random_state=42, n_jobs=-1)

# OOF 예측값을 새로운 피처로 사용하여 메타 모델 학습
meta_model.fit(oof_train, y_train)

# 최종 예측
# oof_test의 예측값은 실수형(평균값)이므로, 라운딩하여 클래스 라벨로 변환 후 예측
final_predictions = meta_model.predict(np.round(oof_test))

print("Meta-Model training and prediction complete.")

# --- 6. 제출 파일 생성 ---
print("Step 6: Creating submission file...")
submission['target'] = final_predictions
submission.to_csv('stacking_submit.csv', index=False)

print("\nStacking Ensemble complete!")
print("Submission file 'stacking_submit.csv' has been created.")
