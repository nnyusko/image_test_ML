import pandas as pd
import numpy as np
import lightgbm as lgb
from itertools import combinations

print("Interaction Feature Creation Script Started...")

# --- 1. 데이터 로드 ---
print("Step 1: Loading original data...")
try:
    train_df = pd.read_csv('assets/train.csv')
    test_df = pd.read_csv('assets/test.csv')
    print("Data loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    exit()

# 데이터 준비
TARGET = 'target'
features = [col for col in train_df.columns if col not in ['ID', TARGET]]
X_train = train_df[features]
y_train = train_df[TARGET]

# --- 2. 중요 피처 식별 ---
print("Step 2: Identifying important features using LightGBM...")

# 이전 튜닝에서 성능이 좋았던 파라미터로 모델 정의
model = lgb.LGBMClassifier(
    learning_rate=0.1, 
    n_estimators=200, 
    num_leaves=31, 
    random_state=42, 
    n_jobs=-1
)

# 전체 훈련 데이터로 모델 학습
model.fit(X_train, y_train)

# 피처 중요도 추출
feature_importances = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

# 상위 6개 피처 선택
N_TOP_FEATURES = 6
top_features = feature_importances.head(N_TOP_FEATURES)['feature'].tolist()

print(f"Top {N_TOP_FEATURES} important features identified: {top_features}")

# --- 3. 상호작용 피처 생성 ---
print("Step 3: Creating interaction features...")

# 원본 데이터프레임을 복사하여 작업
train_interaction_df = train_df.copy()
test_interaction_df = test_df.copy()

# 상위 피처들의 모든 조합(pairs) 생성
for feat_a, feat_b in combinations(top_features, 2):
    # 곱하기 피처
    train_interaction_df[f'{feat_a}_x_{feat_b}'] = train_df[feat_a] * train_df[feat_b]
    test_interaction_df[f'{feat_a}_x_{feat_b}'] = test_df[feat_a] * test_df[feat_b]
    
    # 나누기 피처 (0으로 나누는 것을 방지하기 위해 작은 값(1e-6)을 더함)
    train_interaction_df[f'{feat_a}_div_{feat_b}'] = train_df[feat_a] / (train_df[feat_b] + 1e-6)
    test_interaction_df[f'{feat_a}_div_{feat_b}'] = test_df[feat_a] / (test_df[feat_b] + 1e-6)

print(f"Created {len(list(combinations(top_features, 2))) * 2} new interaction features.")

# --- 4. 새로운 데이터셋 저장 ---
print("Step 4: Saving new datasets with interaction features...")
train_interaction_df.to_csv('assets/train_interaction.csv', index=False)
test_interaction_df.to_csv('assets/test_interaction.csv', index=False)

print("\nInteraction feature creation complete!")
print(f"New training data saved to: assets/train_interaction.csv (shape: {train_interaction_df.shape})")
print(f"New test data saved to: assets/test_interaction.csv (shape: {test_interaction_df.shape})")
