# src/make_dataset.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

def preprocess(train, test, missing_thresh=0.95, target="isFraud"):
    """
    Preprocess train & test datasets:
    - Drop high-missing columns
    - Impute missing values
    - Encode categorical features safely
    - Log-transform TransactionAmt
    """
    # 1️⃣ Drop columns with too many missing values
    missing_train = train.isnull().mean()
    drop_cols = missing_train[missing_train > missing_thresh].index
    train = train.drop(columns=drop_cols)
    test = test.drop(columns=drop_cols, errors="ignore")

    # 2️⃣ Encode categorical features
    cat_cols = train.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        if col in train.columns and col in test.columns:  # encode only if exists in both
            le = LabelEncoder()
            combined = pd.concat([train[col], test[col]], axis=0).astype(str).fillna("NA")
            le.fit(combined)
            train[col] = le.transform(train[col].astype(str).fillna("NA"))
            test[col] = le.transform(test[col].astype(str).fillna("NA"))
        else:
            if col in train.columns:
                train[col] = 0
            if col in test.columns:
                test[col] = 0

    # 3️⃣ Handle numerical missing values (skip target)
    num_cols = train.select_dtypes(include=[np.number]).columns
    num_cols = [c for c in num_cols if c != target]
    for col in num_cols:
        median_val = train[col].median()
        train[col] = train[col].fillna(median_val)
        if col in test.columns:
            test[col] = test[col].fillna(median_val)

    # 4️⃣ Log-transform TransactionAmt
    if "TransactionAmt" in train.columns:
        train["TransactionAmt"] = np.log1p(train["TransactionAmt"])
        test["TransactionAmt"] = np.log1p(test["TransactionAmt"])

    return train, test

if __name__ == "__main__":
    # Paths
    RAW_DIR = os.path.join("..", "data", "raw")
    PROCESSED_DIR = os.path.join("..", "data", "processed")
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Load raw datasets
    print("📂 Loading raw datasets...")
    train_transaction = pd.read_csv(os.path.join(RAW_DIR, "train_transaction.csv"))
    test_transaction = pd.read_csv(os.path.join(RAW_DIR, "test_transaction.csv"))
    train_identity = pd.read_csv(os.path.join(RAW_DIR, "train_identity.csv"))
    test_identity = pd.read_csv(os.path.join(RAW_DIR, "test_identity.csv"))

    # Merge transaction with identity
    train = train_transaction.merge(train_identity, how="left", on="TransactionID")
    test = test_transaction.merge(test_identity, how="left", on="TransactionID")

    print("✅ Shape after merge - Train:", train.shape, " Test:", test.shape)

    # Preprocess
    print("🔄 Preprocessing...")
    train_processed, test_processed = preprocess(train, test)

    # Save processed datasets
    train_processed.to_csv(os.path.join(PROCESSED_DIR, "train_processed.csv"), index=False)
    test_processed.to_csv(os.path.join(PROCESSED_DIR, "test_processed.csv"), index=False)
    print("✅ Preprocessing complete! Files saved in ../data/processed/")
