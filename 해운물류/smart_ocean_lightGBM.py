import pandas as pd
import lightgbm as lgb

# 데이터 불러오기
train = pd.read_csv('assets/train.csv')
test = pd.read_csv('assets/test.csv')
submission = pd.read_csv('assets/sample_submission.csv')

# 데이터 준비
train_x = train.drop(columns=['ID', 'target'])
train_y = train['target']
test_x = test.drop(columns=['ID'])

# LightGBM 모델 생성 및 학습
lgbm = lgb.LGBMClassifier(random_state=42)
lgbm.fit(train_x, train_y)

# 예측
preds = lgbm.predict(test_x)

# 제출 파일 생성
submission['target'] = preds
submission.to_csv('./lightgbm_submit.csv', index=False, encoding='utf-8-sig')

print("lightgbm_submit.csv 파일 생성이 완료되었습니다.")
