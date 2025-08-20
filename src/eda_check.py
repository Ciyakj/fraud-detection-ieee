import pandas as pd

# Load processed data
train = pd.read_csv("../data/processed/train_processed.csv")
test = pd.read_csv("../data/processed/test_processed.csv")

print("✅ Processed Train Shape:", train.shape)
print("✅ Processed Test Shape:", test.shape)

# Check column types
print("\n📌 Data Types:")
print(train.dtypes.value_counts())

# Quick look at features
print("\n📌 First 5 rows:")
print(train.head())

# Null check
print("\n📌 Missing values per column (top 20):")
print(train.isnull().sum().sort_values(ascending=False).head(20))

# Summary stats
print("\n📌 Summary statistics:")
print(train.describe().T.head(20))

# Outlier check for a few numeric features
numeric_cols = train.select_dtypes(include=['int64','float64']).columns
for col in numeric_cols[:5]:   # just first 5 for preview
    q1 = train[col].quantile(0.25)
    q3 = train[col].quantile(0.75)
    iqr = q3 - q1
    outliers = ((train[col] < (q1 - 1.5 * iqr)) | (train[col] > (q3 + 1.5 * iqr))).sum()
    print(f"{col}: {outliers} outliers")
