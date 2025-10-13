import pandas as pd

# Load data
train = pd.read_csv('C:/Users/user/Documents/GitHub/image_test_ML/교통사고위험예측/data/train.csv')
test = pd.read_csv('C:/Users/user/Documents/GitHub/image_test_ML/교통사고위험예측/data/test.csv')
train_A = pd.read_csv('C:/Users/user/Documents/GitHub/image_test_ML/교통사고위험예측/data/train/A.csv')
train_B = pd.read_csv('C:/Users/user/Documents/GitHub/image_test_ML/교통사고위험예측/data/train/B.csv')
test_A = pd.read_csv('C:/Users/user/Documents/GitHub/image_test_ML/교통사고위험예측/data/test/A.csv')
test_B = pd.read_csv('C:/Users/user/Documents/GitHub/image_test_ML/교통사고위험예측/data/test/B.csv')

# Merge data
train_A_merged = pd.merge(train[train['Test'] == 'A'], train_A, on='Test_id')
train_B_merged = pd.merge(train[train['Test'] == 'B'], train_B, on='Test_id')
train_df = pd.concat([train_A_merged, train_B_merged])

test_A_merged = pd.merge(test[test['Test'] == 'A'], test_A, on='Test_id')
test_B_merged = pd.merge(test[test['Test'] == 'B'], test_B, on='Test_id')
test_df = pd.concat([test_A_merged, test_B_merged])

# Clean up columns
train_df = train_df.drop(columns=['Test_y'])
train_df = train_df.rename(columns={'Test_x': 'Test'})

test_df = test_df.drop(columns=['Test_y'])
test_df = test_df.rename(columns={'Test_x': 'Test'})


# Convert 'Age' to numeric, coercing errors
train_df['Age'] = pd.to_numeric(train_df['Age'], errors='coerce')
test_df['Age'] = pd.to_numeric(test_df['Age'], errors='coerce')

# Fill NaNs in Age with the mean
train_age_mean = train_df['Age'].mean()
train_df['Age'].fillna(train_age_mean, inplace=True)
test_df['Age'].fillna(train_age_mean, inplace=True) # Using train mean for test set

# Convert other object columns to numeric, coercing errors
for col in train_df.columns:
    if train_df[col].dtype == 'object' and col not in ['Test_id', 'Test', 'PrimaryKey']:
        train_df[col] = pd.to_numeric(train_df[col], errors='coerce')

for col in test_df.columns:
    if test_df[col].dtype == 'object' and col not in ['Test_id', 'Test', 'PrimaryKey']:
        test_df[col] = pd.to_numeric(test_df[col], errors='coerce')

# Fill remaining NaNs with 0
train_df.fillna(0, inplace=True)
test_df.fillna(0, inplace=True)


print('Train DF Info after cleaning:')
train_df.info()
print('\nTest DF Info after cleaning:')
test_df.info()
print('\nTrain DF Head after cleaning:')
print(train_df.head())
print('\nTest DF Head after cleaning:')
print(test_df.head())

# Save the cleaned data
train_df.to_csv('C:/Users/user/Documents/GitHub/image_test_ML/교통사고위험예측/data/train_cleaned.csv', index=False)
test_df.to_csv('C:/Users/user/Documents/GitHub/image_test_ML/교통사고위험예측/data/test_cleaned.csv', index=False)
