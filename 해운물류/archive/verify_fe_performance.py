import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import cross_val_score

print("Verifying performance of interaction features...")

# 새로운 피처가 추가된 데이터 불러오기
try:
    train_df = pd.read_csv('assets/train_interaction.csv')
    print("Loaded 'assets/train_interaction.csv'")
except FileNotFoundError:
    print("Error: 'assets/train_interaction.csv' not found. Please run 'interaction_feature_creation.py' first.")
    exit()

# ID와 target을 제외한 모든 칼럼을 피처로 사용
# 원본 X_01~X_52 와 추가된 통계 피처 모두 사용
feature_cols = [col for col in train_df.columns if col not in ['ID', 'target']]
train_x = train_df[feature_cols]
train_y = train_df['target']

print(f"Training with {len(feature_cols)} features.")

# 성능 비교를 위해 기존에 사용했던 모델 중 하나를 정의
# (LightGBM Tuned by GridSearchCV)
model = lgb.LGBMClassifier(
    learning_rate=0.1,
    n_estimators=200,
    num_leaves=31,
    random_state=42
)

print("Performing 5-fold cross-validation with the new features...")
scores = cross_val_score(model, train_x, train_y, cv=5, scoring='f1_macro', n_jobs=-1)
new_f1_score = np.mean(scores)

# 기존 최고 점수와 비교
previous_best_score = 0.8097  # LGBM_Random_Tuned

print("\n--- Performance Comparison ---")
print(f"Previous Best F1-Score (LGBM_Random_Tuned): {previous_best_score:.4f}")
print(f"New F1-Score (with Statistical Features):   {new_f1_score:.4f}")

if new_f1_score > previous_best_score:
    print("\nResult: Performance IMPROVED! The new features are effective.")
else:
    print("\nResult: Performance did not improve. Further tuning or different features might be needed.")
