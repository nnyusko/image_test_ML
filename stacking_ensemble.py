
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
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

# 2. 스태킹에 사용할 기본 모델과 메타 모델 정의
print("\n기본 모델과 메타 모델을 정의합니다...")

# 기본 모델 (Level 0)
estimators = [
    ('lgbm', lgb.LGBMClassifier(
        random_state=42,
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=60,
        max_depth=20,
        subsample=0.8,
        colsample_bytree=0.8
    )),
    ('xgb', xgb.XGBClassifier(
        random_state=42, 
        use_label_encoder=False, 
        eval_metric='mlogloss'
    ))
]

# 메타 모델 (Level 1)
meta_model = LogisticRegression(random_state=42, C=0.1) # C: 규제 강도

# 3. 스태킹 앙상블 모델 구성
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

stacking_model = StackingClassifier(
    estimators=estimators,
    final_estimator=meta_model,
    cv=skf, # 교차 검증 방식 전달
    stack_method='predict_proba', # 메타 모델은 각 기본 모델의 예측 확률을 입력으로 받음
    n_jobs=-1
)

# 4. 교차 검증으로 OOF F1 스코어 계산
print(f"\n{N_SPLITS}-Fold 교차 검증으로 스태킹 모델의 성능을 예측합니다...")

# cross_val_predict를 사용해 OOF 예측 생성
oof_preds = cross_val_predict(stacking_model, train_x, train_y, cv=skf, method='predict')

ensemble_f1 = f1_score(train_y, oof_preds, average='macro')
print(f"\n>> 최종 스태킹 앙상블의 예상 OOF F1 Score: {ensemble_f1:.4f}")

# 5. 전체 학습 데이터로 스태킹 모델 학습 및 제출 파일 생성
print("\n전체 데이터로 스태킹 모델을 학습하고 제출 파일을 생성합니다...")
stacking_model.fit(train_x, train_y)
final_preds = stacking_model.predict(test_x)

submission['target'] = le.inverse_transform(final_preds)
submission.to_csv('./stacking_submit.csv', index=False, encoding='utf-8-sig')

print("\nstacking_submit.csv 파일 생성이 완료되었습니다.")
print(f"예상 F1 Score: {ensemble_f1:.4f}")

