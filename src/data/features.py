"""
features.py
Feature engineering for the SmartShelf demand forecasting model.
Input : data/processed/sales_merged.parquet
Output: data/processed/sales_features.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
LAG_DAYS      = [7, 14, 28]
ROLL_WINDOWS  = [7, 28]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    print("Engineering features ...")
    df = df.sort_values(["id", "date"]).copy()

    # Lag features
    for lag in LAG_DAYS:
        df[f"lag_{lag}"] = df.groupby("id")["sales"].shift(lag)

    # Rolling mean and std
    for w in ROLL_WINDOWS:
        shifted = df.groupby("id")["sales"].shift(1)
        df[f"rolling_mean_{w}"] = shifted.rolling(w).mean().reset_index(level=0, drop=True)
        df[f"rolling_std_{w}"]  = shifted.rolling(w).std().reset_index(level=0, drop=True)

    # Price features
    df["price_lag_7"]       = df.groupby("id")["sell_price"].shift(7)
    df["price_change"]      = df["sell_price"] - df["price_lag_7"]
    store_avg               = df.groupby(["store_id", "wm_yr_wk"])["sell_price"].transform("mean")
    df["price_vs_avg"]      = df["sell_price"] / (store_avg + 1e-8)

    # Calendar features
    df["day_of_week"]  = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(np.int8)
    df["has_event"]    = df["event_name_1"].notna().astype(np.int8)

    # SNAP flag per state
    snap_map = {"CA": "snap_CA", "TX": "snap_TX", "WI": "snap_WI"}
    df["snap_day"] = 0
    for state, col in snap_map.items():
        mask = df["state_id"] == state
        df.loc[mask, "snap_day"] = df.loc[mask, col].astype(np.int8)

    # Demand pressure
    df["demand_7d"] = (
        df.groupby("id")["sales"]
        .transform(lambda x: x.shift(1).rolling(7).sum())
    )

    # Drop NaN rows from lag/rolling warmup period
    lag_cols = [f"lag_{l}" for l in LAG_DAYS] + [f"rolling_mean_{w}" for w in ROLL_WINDOWS]
    before = len(df)
    df = df.dropna(subset=lag_cols)
    print(f"  Dropped {before - len(df):,} warmup rows. Final shape: {df.shape}")
    return df


def get_feature_columns() -> list:
    return [
        "lag_7", "lag_14", "lag_28",
        "rolling_mean_7", "rolling_mean_28",
        "rolling_std_7",  "rolling_std_28",
        "sell_price", "price_change", "price_vs_avg",
        "day_of_week", "day_of_month", "week_of_year",
        "is_weekend", "has_event", "snap_day", "demand_7d",
        "item_id", "store_id", "cat_id", "dept_id", "state_id",
        "month", "year",
    ]


if __name__ == "__main__":
    df = pd.read_parquet(PROCESSED_DIR / "sales_merged.parquet")
    df = build_features(df)
    df.to_parquet(PROCESSED_DIR / "sales_features.parquet", index=False)
    print("Features saved.")
