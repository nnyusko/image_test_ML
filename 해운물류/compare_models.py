import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Previously calculated cross-validation scores
# and the new score from RandomizedSearchCV
results_data = {
    'Model': [
        'RandomForest',
        'LightGBM',
        'LightGBM_Tuned',
        'XGBoost',
        'LGBM_Random_Tuned'
    ],
    'F1-score': [
        0.7714,  # From 250929.md
        0.8029,  # From 250929.md
        0.8077,  # From 250929.md
        0.7959,  # From 250929.md
        0.8097   # Provided by user
    ]
}

results_df = pd.DataFrame(results_data)

# 시각화
print("Generating updated plot...")
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='F1-score', data=results_df.sort_values('F1-score', ascending=False))
plt.title('Model F1-Score Comparison (5-Fold Cross-Validation)')
plt.xticks(rotation=45)
plt.ylim(0.75, 0.82) # Adjust y-axis for better visualization
plt.tight_layout()

# 그래프 파일로 저장
plt.savefig('model_comparison.png')
print("Updated plot saved to model_comparison.png")