import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# 데이터 불러오기
print("Loading data...")
train = pd.read_csv('assets/train.csv')
test = pd.read_csv('assets/test.csv')
submission = pd.read_csv('assets/sample_submission.csv')

# 데이터 준비
train_x = train.drop(columns=['ID', 'target'])
train_y = train['target']
test_x = test.drop(columns=['ID'])

# 하이퍼파라미터 탐색 범위 정의
param_dist = {
    'n_estimators': randint(100, 600),
    'learning_rate': uniform(0.01, 0.2),
    'num_leaves': randint(30, 100),
    'max_depth': [-1, 10, 20, 30, 40],
    'subsample': uniform(0.7, 0.3),  # 0.7 to 1.0
    'colsample_bytree': uniform(0.7, 0.3) # 0.7 to 1.0
}

# RandomizedSearchCV 설정
lgbm = lgb.LGBMClassifier(random_state=42)
random_search = RandomizedSearchCV(
    estimator=lgbm, 
    param_distributions=param_dist, 
    n_iter=50,  # 50개의 조합을 랜덤하게 테스트
    cv=5, 
    scoring='f1_macro', 
    verbose=1, 
    n_jobs=-1, 
    random_state=42
)

# 탐색 실행
print("Starting Randomized Search...")
random_search.fit(train_x, train_y)

# 최적의 파라미터 및 점수 출력
print(f"Best parameters found: {random_search.best_params_}")
print(f"Best F1-score: {random_search.best_score_:.4f}")

# 최적의 모델로 예측
best_model = random_search.best_estimator_
preds = best_model.predict(test_x)

# 제출 파일 생성
submission['target'] = preds
submission.to_csv('./lgbm_random_tuned_submit.csv', index=False, encoding='utf-8-sig')

print("lgbm_random_tuned_submit.csv 파일 생성이 완료되었습니다.")
