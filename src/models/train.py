"""
train.py
Trains a LightGBM demand forecasting model on the filtered M5 dataset.
Compares against a naive baseline and saves model + metrics.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import json
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.data.features import get_feature_columns

PROCESSED_DIR = Path("data/processed")
MODEL_DIR     = Path("outputs/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CUTOFF  = "2016-03-27"
CAT_COLS      = ["item_id", "store_id", "cat_id", "dept_id", "state_id"]

LGBM_PARAMS = {
    "objective":               "tweedie",
    "tweedie_variance_power":  1.1,
    "metric":                  "rmse",
    "learning_rate":           0.05,
    "num_leaves":              63,
    "min_data_in_leaf":        20,
    "feature_fraction":        0.8,
    "bagging_fraction":        0.8,
    "bagging_freq":            1,
    "n_estimators":            500,
    "early_stopping_rounds":   30,
    "verbose":                 -1,
    "n_jobs":                  -1,
    "seed":                    42,
}


def encode_categoricals(df: pd.DataFrame, encoders: dict = None):
    fit = encoders is None
    if fit:
        encoders = {}
    for col in CAT_COLS:
        if col not in df.columns:
            continue
        df[col] = df[col].astype(str)
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
        else:
            le = encoders[col]
            known = set(le.classes_)
            df[col] = df[col].apply(lambda x: x if x in known else le.classes_[0])
            df[col] = le.transform(df[col])
        df[col] = df[col].astype(np.int32)
    return df, encoders


def run_training():
    print("Loading features ...")
    df = pd.read_parquet(PROCESSED_DIR / "sales_features.parquet")
    df["date"] = pd.to_datetime(df["date"])

    feature_cols = get_feature_columns()
    train = df[df["date"] <= TRAIN_CUTOFF].copy()
    val   = df[df["date"] >  TRAIN_CUTOFF].copy()
    print(f"  Train: {len(train):,}  Val: {len(val):,}")

    # Encode categoricals
    X_train, encoders = encode_categoricals(train[feature_cols].copy())
    X_val,   _        = encode_categoricals(val[feature_cols].copy(), encoders)
    y_train = train["sales"]
    y_val   = val["sales"]

    joblib.dump(encoders, MODEL_DIR / "label_encoders.pkl")

    # Train LightGBM
    print("Training LightGBM ...")
    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        categorical_feature=CAT_COLS,
        callbacks=[lgb.log_evaluation(50)],
    )

    # Predictions
    lgbm_preds  = np.clip(model.predict(X_val), 0, None)
    naive_preds = val["rolling_mean_28"].fillna(0).values

    # Metrics
    def evaluate(y_true, y_pred, name):
        mae  = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        print(f"  [{name}] MAE={mae:.4f}  RMSE={rmse:.4f}")
        return {"mae": round(mae, 4), "rmse": round(rmse, 4)}

    metrics = {
        "lightgbm":       evaluate(y_val, lgbm_preds,  "LightGBM"),
        "naive_baseline": evaluate(y_val, naive_preds, "Naive baseline"),
    }

    # Save
    joblib.dump(model, MODEL_DIR / "lgbm_model.pkl")
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    val = val.copy()
    val["lgbm_pred"]  = lgbm_preds
    val["naive_pred"] = naive_preds
    val[["id", "item_id", "store_id", "date", "sales",
         "sell_price", "lgbm_pred", "naive_pred"]].to_parquet(
        PROCESSED_DIR / "val_predictions.parquet", index=False
    )
    print("Model, metrics and predictions saved.")
    return model, metrics


if __name__ == "__main__":
    run_training()
