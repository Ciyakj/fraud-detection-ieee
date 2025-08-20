# src/train.py

import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

# ===============================
# 1. Load processed datasets
# ===============================
print("📂 Loading processed data...")
train = pd.read_csv("../data/processed/train.csv")
test = pd.read_csv("../data/processed/test.csv")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# ===============================
# 2. Split features and target
# ===============================
X = train.drop("isFraud", axis=1)
y = train["isFraud"]

# ===============================
# 3. Train/validation split
# ===============================
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===============================
# 4. Train LightGBM model
# ===============================
print("🚀 Training LightGBM model...")
model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=-1,
    num_leaves=64,
    colsample_bytree=0.8,
    subsample=0.8,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="auc",
    early_stopping_rounds=50,
    verbose=50
)

# ===============================
# 5. Evaluate on validation set
# ===============================
y_pred_proba = model.predict_proba(X_val)[:, 1]
y_pred = (y_pred_proba > 0.5).astype(int)

auc = roc_auc_score(y_val, y_pred_proba)
acc = accuracy_score(y_val, y_pred)

print(f"✅ Validation AUC: {auc:.4f}")
print(f"✅ Validation Accuracy: {acc:.4f}")

# ===============================
# 6. Train on full dataset & save model
# ===============================
print("📦 Training final model on full dataset...")
final_model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=-1,
    num_leaves=64,
    colsample_bytree=0.8,
    subsample=0.8,
    random_state=42,
    n_jobs=-1
)
final_model.fit(X, y)

import joblib
joblib.dump(final_model, "../models/fraud_lgbm.pkl")
print("🎯 Final model saved to models/fraud_lgbm.pkl")
