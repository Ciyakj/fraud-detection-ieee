import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def reduce_memory_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype
        if pd.api.types.is_numeric_dtype(col_type):
            c_min = df[col].min()
            c_max = df[col].max()
            if pd.api.types.is_float_dtype(col_type):
                df[col] = pd.to_numeric(df[col], downcast='float')
            else:
                df[col] = pd.to_numeric(df[col], downcast='integer')
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    if verbose:
        print(f"Memory: {start_mem:.2f} MB -> {end_mem:.2f} MB ({100*(start_mem-end_mem)/start_mem:.1f}% reduction)")
    return df

def make_time_features(df: pd.DataFrame, dt_col: str = "TransactionDT") -> pd.DataFrame:
    if dt_col in df.columns:
        df["DT_day"] = (df[dt_col] // (24*60*60)).astype("float32")
        df["DT_hour"] = ((df[dt_col] % (24*60*60)) // (60*60)).astype("float32")
        df["DT_dayofweek"] = (df["DT_day"] % 7).astype("float32")
        df["DT_hour_sin"] = np.sin(2 * np.pi * df["DT_hour"] / 24).astype("float32")
        df["DT_hour_cos"] = np.cos(2 * np.pi * df["DT_hour"] / 24).astype("float32")
    return df

def basic_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    if "TransactionAmt" in df.columns:
        df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"]).astype("float32")
        df["TransactionAmt_cents"] = ((df["TransactionAmt"] - df["TransactionAmt"].astype(int)) * 100).round().astype("float32")
    return df

def fit_label_encoders(train_df: pd.DataFrame, test_df: pd.DataFrame, cat_cols: list[str]) -> dict[str, LabelEncoder]:
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([train_df[col], test_df[col]], axis=0).astype(str).fillna("nan")
        le.fit(combined)
        encoders[col] = le
    return encoders

def apply_label_encoders(df: pd.DataFrame, encoders: dict[str, LabelEncoder]) -> pd.DataFrame:
    for col, le in encoders.items():
        df[col] = df[col].astype(str).fillna("nan")
        known = set(le.classes_)
        df[col] = df[col].apply(lambda x: x if x in known else "nan")
        if "nan" not in le.classes_:
            le.classes_ = np.append(le.classes_, "nan")
        df[col] = le.transform(df[col]).astype("int32")
    return df
