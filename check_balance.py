import pandas as pd

print("Checking class distribution of the 'target' variable...")

try:
    train_df = pd.read_csv('assets/train.csv')
    print("Loaded 'assets/train.csv'")
except FileNotFoundError:
    print("Error: 'assets/train.csv' not found.")
    exit()

print("\n--- Target Variable Distribution ---")

# Get raw counts
target_counts = train_df['target'].value_counts()
print("\nRaw Counts:")
print(target_counts)

# Get percentage distribution
target_percentages = train_df['target'].value_counts(normalize=True) * 100
print("\nPercentage Distribution:")
print(target_percentages)

# A simple check for imbalance
minority_class_percentage = target_percentages.min()
if minority_class_percentage < 10:
    print(f"\nConclusion: The dataset is highly imbalanced. The smallest class represents only {minority_class_percentage:.2f}% of the data.")
    print("Applying techniques to handle imbalance (like SMOTE) is strongly recommended.")
else:
    print("\nConclusion: The dataset appears to be reasonably balanced.")
