import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Setup paths and directories
# -----------------------------
os.makedirs("../reports", exist_ok=True)

DATA_DIR = os.path.join("..", "data", "processed")
train_path = os.path.join(DATA_DIR, "train_processed.csv")
test_path = os.path.join(DATA_DIR, "test_processed.csv")

# -----------------------------
# Load datasets
# -----------------------------
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print("✅ Train shape:", train_df.shape)
print("✅ Test shape:", test_df.shape)
print("\n🔹 Columns:", train_df.columns.tolist())

# -----------------------------
# Missing values
# -----------------------------
print("\n📊 Missing values in Train set:")
missing = train_df.isnull().sum().sort_values(ascending=False).head(20)
print(missing)

# -----------------------------
# Class distribution
# -----------------------------
if "isFraud" in train_df.columns:
    print("\n⚖️ Class distribution (isFraud):")
    print(train_df["isFraud"].value_counts(normalize=True))

    plt.figure(figsize=(5,4))
    sns.countplot(x="isFraud", data=train_df)
    plt.title("Class Distribution (Fraud vs Non-Fraud)")
    plt.savefig("../reports/class_distribution.png")
    plt.close()

# -----------------------------
# Numeric feature analysis
# -----------------------------
numeric_cols = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numeric_cols = [col for col in numeric_cols if col != "isFraud"]  # exclude target

# 1. Distribution & skewness
for col in ["TransactionAmt"]:
    if col in numeric_cols:
        plt.figure(figsize=(6,4))
        sns.histplot(train_df[col], bins=50, kde=True)
        plt.title(f"Distribution of {col}")
        plt.savefig(f"../reports/{col}_distribution.png")
        plt.close()

        # Log-transform plot
        train_df[col + "_log"] = np.log1p(train_df[col])
        plt.figure(figsize=(6,4))
        sns.histplot(train_df[col + "_log"], bins=50, kde=True)
        plt.title(f"Log-Transformed Distribution of {col}")
        plt.savefig(f"../reports/{col}_log_distribution.png")
        plt.close()

# 2. Numeric vs target
for col in ["TransactionAmt"]:
    if col in numeric_cols:
        plt.figure(figsize=(6,4))
        sns.boxplot(x="isFraud", y=col, data=train_df)
        plt.title(f"{col} by Fraud Status")
        plt.savefig(f"../reports/{col}_vs_isFraud.png")
        plt.close()

# 3. Correlation heatmap (first 20 numeric features)
corr = train_df[numeric_cols].iloc[:, :20].corr()
plt.figure(figsize=(15,10))
sns.heatmap(corr, annot=False, cmap="coolwarm")
plt.title("Correlation Heatmap (first 20 numeric features)")
plt.savefig("../reports/corr_heatmap.png")
plt.close()

# -----------------------------
# Categorical feature analysis
# -----------------------------
cat_cols = train_df.select_dtypes(include=['object']).columns.tolist()

for col in cat_cols:
    plt.figure(figsize=(6,4))
    top_vals = train_df[col].value_counts().index[:10]
    sns.countplot(y=col, data=train_df, order=top_vals)
    plt.title(f"Top 10 categories in {col}")
    plt.tight_layout()
    plt.savefig(f"../reports/{col}_top_categories.png")
    plt.close()

    # Category vs target (fraud rate)
    if "isFraud" in train_df.columns:
        plt.figure(figsize=(6,4))
        sns.barplot(x=train_df.groupby(col)["isFraud"].mean().loc[top_vals].index,
                    y=train_df.groupby(col)["isFraud"].mean().loc[top_vals].values)
        plt.xticks(rotation=45)
        plt.ylabel("Fraud Rate")
        plt.title(f"Fraud Rate by {col} (Top 10 categories)")
        plt.tight_layout()
        plt.savefig(f"../reports/{col}_fraud_rate.png")
        plt.close()

print("\n✅ EDA completed. All plots saved in '../reports/'")
