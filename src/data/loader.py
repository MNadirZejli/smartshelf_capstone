"""
loader.py
Loads M5 data for 3 stores (CA_1, TX_1, WI_1) and top 100 products by sales volume.
Produces a clean parquet file of ~580k rows — runs on any machine with 4GB+ RAM.
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR       = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

STORES    = ["CA_1", "TX_1", "WI_1"]
N_PRODUCTS = 100

CAL_COLS = ["d", "date", "wm_yr_wk", "weekday", "wday", "month", "year",
            "event_name_1", "event_type_1", "snap_CA", "snap_TX", "snap_WI"]
ID_COLS  = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]


def run_pipeline() -> pd.DataFrame:
    print("Loading calendar.csv ...")
    calendar = pd.read_csv(RAW_DIR / "calendar.csv", usecols=CAL_COLS)
    calendar["date"] = pd.to_datetime(calendar["date"])

    print("Loading sell_prices.csv ...")
    prices = pd.read_csv(RAW_DIR / "sell_prices.csv",
                         dtype={"sell_price": np.float32})

    print("Loading sales data ...")
    sales = pd.read_csv(RAW_DIR / "sales_train_evaluation.csv")

    # ── Filter to 3 stores ────────────────────────────────────────────────────
    sales = sales[sales["store_id"].isin(STORES)].copy()
    print(f"  Filtered to {STORES}: {len(sales)} items")

    # ── Select top N products by total sales (across all 3 stores) ────────────
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    sales["total_sales"] = sales[day_cols].sum(axis=1)
    top_items = (
        sales.groupby("item_id")["total_sales"]
        .sum()
        .nlargest(N_PRODUCTS)
        .index.tolist()
    )
    sales = sales[sales["item_id"].isin(top_items)].drop(columns=["total_sales"])
    print(f"  Selected top {N_PRODUCTS} products: {len(sales)} rows remaining")

    # ── Melt wide -> long ─────────────────────────────────────────────────────
    print("Melting to long format ...")
    long = sales.melt(
        id_vars=ID_COLS,
        value_vars=day_cols,
        var_name="d",
        value_name="sales",
    )
    del sales
    print(f"  Long format shape: {long.shape}")

    # ── Merge calendar ────────────────────────────────────────────────────────
    print("Merging calendar ...")
    long = long.merge(calendar, on="d", how="left")

    # ── Merge prices ──────────────────────────────────────────────────────────
    print("Merging prices ...")
    prices_filtered = prices[prices["store_id"].isin(STORES)]
    long = long.merge(
        prices_filtered[["store_id", "item_id", "wm_yr_wk", "sell_price"]],
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left",
    )

    # ── Clean ─────────────────────────────────────────────────────────────────
    print("Cleaning ...")
    long = long.sort_values(["id", "date"])
    long["sell_price"] = (
        long.groupby("id")["sell_price"]
        .transform(lambda x: x.ffill().bfill())
    )
    long = long.dropna(subset=["sell_price"])
    long["sales"]      = long["sales"].astype(np.int16)
    long["sell_price"] = long["sell_price"].astype(np.float32)

    # ── Save ──────────────────────────────────────────────────────────────────
    out = PROCESSED_DIR / "sales_merged.parquet"
    long.to_parquet(out, index=False)
    print(f"Saved {len(long):,} rows to {out}")
    return long


if __name__ == "__main__":
    run_pipeline()
