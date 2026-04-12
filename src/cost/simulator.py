"""
simulator.py
Translates forecast errors into inventory costs (overstock + stockout).
Computes business value of SmartShelf vs naive baseline.
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_DIR = Path("data/processed")

# Industry standard assumptions (Silver, Pyke & Thomas, 1998)
DEFAULT_HOLDING  = 0.00068   # 25% per year = ~0.068% per day
DEFAULT_STOCKOUT = 0.75      # 75% of item value


def compute_costs(
    val:          pd.DataFrame,
    holding_rate: float = DEFAULT_HOLDING,
    stockout_rate: float = DEFAULT_STOCKOUT,
) -> pd.DataFrame:
    val = val.copy()
    price = val["sell_price"].values

    for model in ["lgbm", "naive"]:
        err  = val[f"{model}_pred"].values - val["sales"].values
        over = np.clip(err,  0, None) * price * holding_rate
        out  = np.clip(-err, 0, None) * price * stockout_rate
        val[f"{model}_overstock_cost"] = over
        val[f"{model}_stockout_cost"]  = out
        val[f"{model}_total_cost"]     = over + out

    val["savings"] = val["naive_total_cost"] - val["lgbm_total_cost"]
    return val


def summary_by_store(val: pd.DataFrame) -> pd.DataFrame:
    return (
        val.groupby("store_id")
        .agg(lgbm_cost=("lgbm_total_cost", "sum"),
             naive_cost=("naive_total_cost", "sum"),
             savings=("savings", "sum"))
        .round(2).reset_index()
        .sort_values("savings", ascending=False)
    )


def summary_by_category(val: pd.DataFrame) -> pd.DataFrame:
    v = val.copy()
    v["cat"] = v["item_id"].str.split("_").str[0]
    return (
        v.groupby("cat")
        .agg(lgbm_cost=("lgbm_total_cost", "sum"),
             naive_cost=("naive_total_cost", "sum"),
             savings=("savings", "sum"))
        .round(2).reset_index()
        .sort_values("savings", ascending=False)
    )


if __name__ == "__main__":
    val = pd.read_parquet(PROCESSED_DIR / "val_predictions.parquet")
    sim = compute_costs(val)
    print(f"Total savings: €{sim['savings'].sum():,.2f}")
    print(summary_by_store(sim))
