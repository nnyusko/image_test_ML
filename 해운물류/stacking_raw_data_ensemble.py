
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

# 1. 원본 데이터 불러오기
print("원본 데이터를 불러옵니다 (train.csv, test.csv)...")
train = pd.read_csv('assets/train.csv')
test = pd.read_csv('assets/test.csv')
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

# 기본 모델 (Level 0) - soft_voting_submit에서 사용된 모델 구성
estimators = [
    ('lgbm', lgb.LGBMClassifier(random_state=42)),
    ('xgb', xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')),
    ('lgbm_tuned', lgb.LGBMClassifier(
        random_state=42, 
        n_estimators=200, 
        learning_rate=0.1, 
        num_leaves=50
    ))
]

# 메타 모델 (Level 1)
meta_model = LogisticRegression(random_state=42, C=0.1)

# 3. 스태킹 앙상블 모델 구성
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

stacking_model = StackingClassifier(
    estimators=estimators,
    final_estimator=meta_model,
    cv=skf,
    stack_method='predict_proba',
    n_jobs=-1
)

# 4. 교차 검증으로 OOF F1 스코어 계산
print(f"\n{N_SPLITS}-Fold 교차 검증으로 스태킹 모델의 성능을 예측합니다...")

oof_preds = cross_val_predict(stacking_model, train_x, train_y, cv=skf, method='predict')

ensemble_f1 = f1_score(train_y, oof_preds, average='macro')
print(f"\n>> 최종 스태킹 앙상블 (Raw Data)의 예상 OOF F1 Score: {ensemble_f1:.4f}")

# 5. 전체 학습 데이터로 스태킹 모델 학습 및 제출 파일 생성
print("\n전체 데이터로 스태킹 모델을 학습하고 제출 파일을 생성합니다...")
stacking_model.fit(train_x, train_y)
final_preds = stacking_model.predict(test_x)

submission['target'] = le.inverse_transform(final_preds)
submission.to_csv('./stacking_raw_submit.csv', index=False, encoding='utf-8-sig')

print("\nstacking_raw_submit.csv 파일 생성이 완료되었습니다.")
print(f"예상 F1 Score: {ensemble_f1:.4f}")
