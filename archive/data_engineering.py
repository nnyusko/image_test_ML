import pandas as pd
import numpy as np

print("Starting feature engineering...")

# 데이터 불러오기
print("Loading original data from 'assets' directory...")
try:
    train_df = pd.read_csv('assets/train.csv')
    test_df = pd.read_csv('assets/test.csv')
except FileNotFoundError:
    print("Error: 'assets/train.csv' or 'assets/test.csv' not found.")
    exit()


# 피처 칼럼 목록 정의 (X_01 ~ X_52)
feature_cols = [f'X_{i:02}' for i in range(1, 53)]

# 통계 피처 생성을 위한 함수
def create_statistical_features(df, features):
    print(f"Creating statistical features for dataframe with shape {df.shape}...")
    df['X_mean'] = df[features].mean(axis=1)
    df['X_std'] = df[features].std(axis=1)
    df['X_min'] = df[features].min(axis=1)
    df['X_max'] = df[features].max(axis=1)
    df['X_sum'] = df[features].sum(axis=1)
    print("Statistical features created: X_mean, X_std, X_min, X_max, X_sum")
    return df

# 훈련 데이터와 테스트 데이터에 피처 생성 함수 적용
train_df_fe = create_statistical_features(train_df.copy(), feature_cols)
test_df_fe = create_statistical_features(test_df.copy(), feature_cols)

# 새로운 피처가 추가된 파일을 저장
print("Saving feature-engineered data to 'assets' directory...")
train_df_fe.to_csv('assets/train_fe.csv', index=False)
test_df_fe.to_csv('assets/test_fe.csv', index=False)

print("\nFeature engineering complete!")
print(f"New training data saved to: assets/train_fe.csv (shape: {train_df_fe.shape})")
print(f"New test data saved to: assets/test_fe.csv (shape: {test_df_fe.shape})")

# 생성된 파일의 상위 5개 행 확인
print("\n--- Top 5 rows of train_fe.csv ---")
print(train_df_fe.head())
