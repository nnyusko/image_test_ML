import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV

# 데이터 불러오기
train = pd.read_csv('assets/train.csv')
test = pd.read_csv('assets/test.csv')
submission = pd.read_csv('assets/sample_submission.csv')

# 데이터 준비
train_x = train.drop(columns=['ID', 'target'])
train_y = train['target']
test_x = test.drop(columns=['ID'])

# 하이퍼파라미터 그리드 정의
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'num_leaves': [31, 50]
}

# LightGBM 모델 및 GridSearchCV 초기화
lgbm = lgb.LGBMClassifier(random_state=42)
grid_search = GridSearchCV(estimator=lgbm, param_grid=param_grid, cv=3, scoring='f1_macro', verbose=1)

# 그리드 서치 실행
grid_search.fit(train_x, train_y)

# 최적의 파라미터 출력
print(f"Best parameters found: {grid_search.best_params_}")

# 최적의 모델로 예측
best_model = grid_search.best_estimator_
preds = best_model.predict(test_x)

# 제출 파일 생성
submission['target'] = preds
submission.to_csv('./lgbm_tuned_submit.csv', index=False, encoding='utf-8-sig')

print("lgbm_tuned_submit.csv 파일 생성이 완료되었습니다.")
