import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# ==========================
# 1. Load Preprocessed Data
# ==========================
print("📥 Loading train_processed.csv")
train = pd.read_csv(r"../data/processed/train_processed.csv")

print("📥 Loading test_processed.csv")
test = pd.read_csv(r"../data/processed/test_processed.csv")

# ==========================
# 2. Encode Categorical Features
# ==========================
print("🔄 Encoding categorical features...")

cat_cols = train.select_dtypes(include=["object"]).columns.tolist()
print(f"🔹 Found {len(cat_cols)} categorical columns")

for col in cat_cols:
    le = LabelEncoder()
    # Combine train & test for consistent encoding
    combined = pd.concat([train[col], test[col]], axis=0).astype(str).fillna("NA")
    le.fit(combined)
    train[col] = le.transform(train[col].astype(str).fillna("NA"))
    if col in test.columns:
        test[col] = le.transform(test[col].astype(str).fillna("NA"))
    else:
        # If the column is missing in test, create it with zeros
        test[col] = 0

# ==========================
# 3. Train / Validation Split
# ==========================
X = train.drop(["isFraud", "TransactionID"], axis=1, errors="ignore")
y = train["isFraud"]

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"🔹 Train shape: {X_train.shape}, Valid shape: {X_valid.shape}")

# ==========================
# 4. Train LightGBM Model
# ==========================
print("🚀 Training LightGBM model...")

model = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=64,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

callbacks = [
    lgb.early_stopping(stopping_rounds=50),
    lgb.log_evaluation(100)
]

model.fit(
    X_train,
    y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric="auc",
    callbacks=callbacks
)

# ==========================
# 5. Validation Score
# ==========================
y_pred = model.predict_proba(X_valid)[:, 1]
auc = roc_auc_score(y_valid, y_pred)
print(f"✅ Validation AUC: {auc:.4f}")

# ==========================
# 6. Train on Full Dataset
# ==========================
print("📦 Training final model on full dataset...")

final_model = lgb.LGBMClassifier(
    n_estimators=model.best_iteration_,
    learning_rate=0.05,
    num_leaves=64,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

final_model.fit(X, y)

# ==========================
# 7. Save Model
# ==========================
os.makedirs("../models", exist_ok=True)
joblib.dump(final_model, "../models/lightgbm_model.pkl")
print("💾 Model saved to models/lightgbm_model.pkl")

# ==========================
# 8. Generate Predictions
# ==========================
if "TransactionID" in test.columns:
    test_ids = test["TransactionID"]
    X_test = test.drop("TransactionID", axis=1, errors="ignore")
else:
    test_ids = pd.Series(range(len(test)))
    X_test = test

# Align test features with training features
X_test = X_test.reindex(columns=X.columns, fill_value=0)

# Ensure numeric
X_test = X_test.apply(pd.to_numeric, errors='coerce').fillna(0)

test_preds = final_model.predict_proba(X_test)[:, 1]

os.makedirs("../submissions", exist_ok=True)
submission = pd.DataFrame({
    "TransactionID": test_ids,
    "isFraud": test_preds
})
submission.to_csv("../submissions/lgbm_submission.csv", index=False)
print("📤 Submission saved to submissions/lgbm_submission.csv")
