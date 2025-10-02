import pandas as pd
import xgboost as xgb

# 데이터 불러오기
train = pd.read_csv('assets/train.csv')
test = pd.read_csv('assets/test.csv')
submission = pd.read_csv('assets/sample_submission.csv')

# 데이터 준비
train_x = train.drop(columns=['ID', 'target'])
train_y = train['target']
test_x = test.drop(columns=['ID'])

# XGBoost 모델 생성 및 학습
xgb_model = xgb.XGBClassifier(random_state=42)
xgb_model.fit(train_x, train_y)

# 예측
preds = xgb_model.predict(test_x)

# 제출 파일 생성
submission['target'] = preds
submission.to_csv('./xgboost_submit.csv', index=False, encoding='utf-8-sig')

print("xgboost_submit.csv 파일 생성이 완료되었습니다.")