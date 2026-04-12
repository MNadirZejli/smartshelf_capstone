"""
predict.py
Generates a 7-day demand forecast for a given item-store pair.
Used by the Streamlit app.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from src.data.features import get_feature_columns

MODEL_DIR     = Path("outputs/models")
PROCESSED_DIR = Path("data/processed")

_model    = None
_encoders = None


def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_DIR / "lgbm_model.pkl")
    return _model


def get_encoders():
    global _encoders
    if _encoders is None:
        _encoders = joblib.load(MODEL_DIR / "label_encoders.pkl")
    return _encoders


def encode_row(df: pd.DataFrame) -> pd.DataFrame:
    encoders = get_encoders()
    for col, le in encoders.items():
        if col not in df.columns:
            continue
        df[col] = df[col].astype(str)
        known = set(le.classes_)
        df[col] = df[col].apply(lambda x: x if x in known else le.classes_[0])
        df[col] = le.transform(df[col]).astype(np.int32)
    return df


def forecast_item(item_df: pd.DataFrame, horizon: int = 7) -> pd.DataFrame:
    """
    Given the historical DataFrame for one item-store,
    generate a horizon-day forecast using the last known feature row.
    """
    model        = get_model()
    feature_cols = get_feature_columns()

    item_df = item_df.sort_values("date")
    last    = item_df.tail(1).copy()

    rows = []
    for i in range(1, horizon + 1):
        row = last.copy()
        row["date"]         = last["date"].values[0] + pd.Timedelta(days=i)
        row["day_of_week"]  = row["date"].dt.dayofweek.values[0]
        row["day_of_month"] = row["date"].dt.day.values[0]
        row["week_of_year"] = row["date"].dt.isocalendar().week.values[0]
        row["is_weekend"]   = int(row["day_of_week"].values[0] >= 5)
        rows.append(row)

    future = pd.concat(rows, ignore_index=True)
    future = encode_row(future)

    # Drop any remaining object columns not in feature_cols
    for col in future.select_dtypes(include="object").columns:
        if col in feature_cols:
            future = future.drop(columns=[col])

    X     = future[feature_cols]
    preds = np.clip(model.predict(X), 0, None)

    recent_std = item_df["sales"].tail(28).std()
    if np.isnan(recent_std):
        recent_std = 0

    return pd.DataFrame({
        "date":       future["date"].values,
        "forecast":   np.round(preds, 1),
        "lower":      np.clip(np.round(preds - recent_std, 1), 0, None),
        "upper":      np.round(preds + recent_std, 1),
        "sell_price": float(last["sell_price"].values[0]),
    })


def compute_order_quantity(
    forecast_df:      pd.DataFrame,
    current_stock:    float = 0,
    lead_time_days:   int   = 2,
    safety_stock_days: int  = 3,
) -> dict:
    daily_avg    = forecast_df["forecast"].mean()
    safety_stock = daily_avg * safety_stock_days
    order_qty    = max(0, round(daily_avg * lead_time_days + safety_stock - current_stock))
    return {
        "order_quantity":      int(order_qty),
        "expected_7d_demand":  round(float(forecast_df["forecast"].sum()), 1),
        "daily_avg":           round(float(daily_avg), 1),
        "safety_stock":        round(float(safety_stock), 1),
    }
