import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 불러오기
print("Loading data...")
train = pd.read_csv('assets/train.csv')

# 데이터 준비
train_x = train.drop(columns=['ID', 'target'])
train_y = train['target']

# 모델 정의
print("Defining models...")
rf_model = RandomForestClassifier(random_state=42)
lgbm_model = lgb.LGBMClassifier(random_state=42)
lgbm_tuned_model = lgb.LGBMClassifier(learning_rate=0.1, n_estimators=200, num_leaves=31, random_state=42)
xgb_model = xgb.XGBClassifier(random_state=42)

models = {
    'RandomForest': rf_model,
    'LightGBM': lgbm_model,
    'LightGBM_Tuned': lgbm_tuned_model,
    'XGBoost': xgb_model
}

# 교차 검증 수행
print("Performing cross-validation...")
results = {}
for name, model in models.items():
    print(f"Validating {name}...")
    # XGBoost는 target 값이 0부터 시작해야 하므로, 분기 처리
    y = train_y
    if name == 'XGBoost':
        # Although our labels are 0-indexed, XGBoost can sometimes be picky
        # about the data type. Let's ensure it's standard int.
        y = train_y.astype(int)

    scores = cross_val_score(model, train_x, y, cv=5, scoring='f1_macro', n_jobs=-1)
    results[name] = scores
    print(f"{name} Mean F1-score: {np.mean(scores):.4f}")

# 결과 시각화
print("Generating plot...")
results_df = pd.DataFrame(results).melt(var_name='Model', value_name='F1-score')

plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='F1-score', data=results_df)
plt.title('Model F1-Score Comparison (5-Fold Cross-Validation)')
plt.xticks(rotation=45)
plt.tight_layout()

# 그래프 파일로 저장
plt.savefig('model_comparison.png')
print("Plot saved to model_comparison.png")
