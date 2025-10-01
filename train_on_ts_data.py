
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import catboost as ctb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

# 1. 시계열 피처가 추가된 데이터 불러오기
print("시계열 피처 데이터를 불러옵니다 (train_ts_fe.csv, test_ts_fe.csv)...")
train = pd.read_csv('assets/train_ts_fe.csv')
test = pd.read_csv('assets/test_ts_fe.csv')
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

# 2. 앙상블할 모델 정의 (CatBoost 포함)
print("\n앙상블 모델을 정의합니다 (LGBM, XGB, CatBoost)...")

SEED = 42

models = {
    'lgbm': lgb.LGBMClassifier(random_state=SEED, reg_alpha=0.1, reg_lambda=0.1),
    'xgb': xgb.XGBClassifier(random_state=SEED, use_label_encoder=False, eval_metric='mlogloss'),
    'lgbm_tuned': lgb.LGBMClassifier(
        random_state=SEED, 
        n_estimators=200, 
        learning_rate=0.1, 
        num_leaves=50,
        reg_alpha=0.1,
        reg_lambda=0.1
    ),
    'cat': ctb.CatBoostClassifier(random_state=SEED, verbose=0)
}

# 3. 교차 검증 기반 예측 확률 생성
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

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


# 4. 소프트 보팅 앙상블
print("소프트 보팅 앙상블을 수행합니다...")

oof_ensemble_preds = np.mean([preds for preds in oof_preds.values()], axis=0)
ensemble_f1 = f1_score(train_y, np.argmax(oof_ensemble_preds, axis=1), average='macro')
print(f"\n>> 최종 앙상블 (TS Features) OOF F1 Score: {ensemble_f1:.4f}")

# 5. 제출 파일 생성
print("제출 파일을 생성합니다...")
test_ensemble_preds = np.mean([preds for preds in test_preds.values()], axis=0)
final_preds = np.argmax(test_ensemble_preds, axis=1)

submission['target'] = le.inverse_transform(final_preds)
submission.to_csv('./ts_features_submit.csv', index=False, encoding='utf-8-sig')

print("\n_ts_features_submit.csv 파일 생성이 완료되었습니다.")
print(f"예상 F1 Score: {ensemble_f1:.4f}")
