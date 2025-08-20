import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def feature_engineering():
    # -----------------------------
    # Paths
    # -----------------------------
    PROCESSED_DIR = os.path.join("..", "data", "processed")
    READY_DIR = os.path.join("..", "data", "ready")
    os.makedirs(READY_DIR, exist_ok=True)

    train_path = os.path.join(PROCESSED_DIR, "train_processed.csv")
    test_path = os.path.join(PROCESSED_DIR, "test_processed.csv")

    # -----------------------------
    # Load datasets
    # -----------------------------
    print("📂 Loading processed datasets ...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # -----------------------------
    # Drop mostly empty columns
    # -----------------------------
    mostly_missing = ['id_24','id_25','id_07','id_08','id_21','id_26','id_27','id_23','id_22','dist2']
    for c in mostly_missing:
        if c in train_df.columns:
            train_df.drop(columns=c, inplace=True)
        if c in test_df.columns:
            test_df.drop(columns=c, inplace=True)

    # -----------------------------
    # Fill missing values
    # -----------------------------
    # Numeric
    numeric_cols = train_df.select_dtypes(include=['int64','float64']).columns.tolist()
    if 'isFraud' in numeric_cols:
        numeric_cols.remove('isFraud')

    for col in numeric_cols:
        if col in train_df.columns:
            median = train_df[col].median()
            train_df[col].fillna(median, inplace=True)
        if col in test_df.columns:
            median = train_df[col].median() if col in train_df.columns else 0
            test_df[col].fillna(median, inplace=True)

    # Categorical
    categorical_cols = train_df.select_dtypes(include=['object','category']).columns.tolist()
    for col in categorical_cols:
        if col in train_df.columns:
            train_df[col].fillna('Unknown', inplace=True)
        if col in test_df.columns:
            test_df[col].fillna('Unknown', inplace=True)

    # -----------------------------
    # Feature creation
    # -----------------------------
    for df in [train_df, test_df]:
        if 'TransactionAmt' in df.columns:
            df['TransactionAmt_log'] = np.log1p(df['TransactionAmt'])
        if 'TransactionDT' in df.columns:
            df['Hour'] = ((df['TransactionDT'] // 3600) % 24).astype(int)
            df['DayOfWeek'] = ((df['TransactionDT'] // 3600 // 24) % 7).astype(int)

    # -----------------------------
    # Frequency encoding
    # -----------------------------
    high_card_cols = ['card1','card2','card3','card5','addr1','addr2']
    for col in high_card_cols:
        if col in train_df.columns:
            freq = train_df[col].value_counts(normalize=True)
            train_df[col+'_freq_enc'] = train_df[col].map(freq)
            if col in test_df.columns:
                test_df[col+'_freq_enc'] = test_df[col].map(freq).fillna(0)

    # -----------------------------
    # One-hot encoding
    # -----------------------------
    low_card_cols = ['ProductCD','card4','card6','P_emaildomain','R_emaildomain','DeviceType']
    for col in low_card_cols:
        if col in train_df.columns and col in test_df.columns:
            combined = pd.concat([train_df[col], test_df[col]], axis=0)
            dummies = pd.get_dummies(combined, prefix=col)
            train_dummies = dummies.iloc[:len(train_df), :]
            test_dummies = dummies.iloc[len(train_df):, :]
            train_df = pd.concat([train_df, train_dummies], axis=1)
            test_df = pd.concat([test_df, test_dummies], axis=1)
            train_df.drop(columns=[col], inplace=True)
            test_df.drop(columns=[col], inplace=True)

    # -----------------------------
    # Scaling numeric features
    # -----------------------------
    # Recompute numeric columns after drops and one-hot encoding
    train_numeric_cols = train_df.select_dtypes(include=['int64','float64']).columns.tolist()
    if 'isFraud' in train_numeric_cols:
        train_numeric_cols.remove('isFraud')

    # Only include columns present in both train & test
    scale_cols = [c for c in train_numeric_cols if c in test_df.columns]

    scaler = StandardScaler()
    train_df[scale_cols] = scaler.fit_transform(train_df[scale_cols])
    test_df[scale_cols] = scaler.transform(test_df[scale_cols])

    # -----------------------------
    # Save ready datasets
    # -----------------------------
    train_df.to_csv(os.path.join(READY_DIR,'train_ready.csv'), index=False)
    test_df.to_csv(os.path.join(READY_DIR,'test_ready.csv'), index=False)

    print("🎉 Feature engineering done! Files saved in data/ready/")

if __name__ == "__main__":
    feature_engineering()
