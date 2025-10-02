
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

# 1. 피처 엔지니어링 데이터 불러오기
print("피처 엔지니어링 데이터를 불러옵니다 (train_fe.csv, test_fe.csv)...")
train = pd.read_csv('assets/train_fe.csv')
test = pd.read_csv('assets/test_fe.csv')
submission = pd.read_csv('assets/sample_submission.csv')

# 레이블 인코딩
le = LabelEncoder()
train['target'] = le.fit_transform(train['target'])

# 데이터 준비
train_x = train.drop(columns=['ID', 'target'])
train_y = train['target']
test_x = test.drop(columns=['ID'])

print(f"Train 데이터 형태: {train_x.shape}")
print(f"Test 데이터 형태: {test_x.shape}")

# 2. 앙상블할 모델 정의 (FE 데이터 기반)
print("앙상블 모델을 정의합니다...")
models = {
    'lgbm_tuned_fe': lgb.LGBMClassifier(
        random_state=42,
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=60,
        max_depth=20,
        subsample=0.8,
        colsample_bytree=0.8
    ),
    'xgb_fe': xgb.XGBClassifier(
        random_state=42, 
        use_label_encoder=False, 
        eval_metric='mlogloss'
    )
}

# 3. 교차 검증 기반 예측 확률 생성
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

oof_preds = {}  # Out-of-Fold 예측 확률 저장
test_preds = {} # 테스트 데이터 예측 확률 저장

print(f"\n{N_SPLITS}-Fold 교차 검증을 시작합니다...")

for model_name, model in models.items():
    print(f"===== {model_name} 모델 학습 및 예측 중... =====")
    
    oof_model_preds = np.zeros((len(train), len(train_y.unique())))
    test_model_preds = np.zeros((len(test), len(train_y.unique())))

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_x, train_y)):
        print(f"  Fold {fold+1}/{N_SPLITS}...")
        
        X_train, y_train = train_x.iloc[train_idx], train_y.iloc[train_idx]
        X_val, y_val = train_x.iloc[val_idx], train_y.iloc[val_idx]

        model.fit(X_train, y_train)

        oof_model_preds[val_idx] = model.predict_proba(X_val)
        test_model_preds += model.predict_proba(test_x) / N_SPLITS

    oof_preds[model_name] = oof_model_preds
    test_preds[model_name] = test_model_preds
    
    f1 = f1_score(train_y, np.argmax(oof_model_preds, axis=1), average='macro')
    print(f"  => {model_name} OOF F1 Score: {f1:.4f}\n")


# 4. 가중치 앙상블 (Weighted Ensemble)
print("가중치 앙상블을 수행합니다 (LGBM: 70%, XGB: 30%)...")

weights = [0.7, 0.3]  # lgbm_tuned_fe: 0.7, xgb_fe: 0.3
model_keys = list(models.keys())

# OOF 예측에 가중 평균 적용
oof_ensemble_preds = np.average(
    [oof_preds[key] for key in model_keys], axis=0, weights=weights
)
ensemble_f1 = f1_score(train_y, np.argmax(oof_ensemble_preds, axis=1), average='macro')
print(f"\n>> 최종 앙상블 OOF F1 Score: {ensemble_f1:.4f}")

# 5. 제출 파일 생성
print("제출 파일을 생성합니다...")

# 테스트 예측에 가중 평균 적용
test_ensemble_preds = np.average(
    [test_preds[key] for key in model_keys], axis=0, weights=weights
)
final_preds = np.argmax(test_ensemble_preds, axis=1)

submission['target'] = le.inverse_transform(final_preds)
submission.to_csv('./soft_voting_fe_submit.csv', index=False, encoding='utf-8-sig')

print("\nsoft_voting_fe_submit.csv 파일 생성이 완료되었습니다.")
print(f"예상 F1 Score: {ensemble_f1:.4f}")
