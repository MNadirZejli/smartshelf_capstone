"""
run_pipeline.py
Runs the full SmartShelf pipeline end to end.
Usage: python run_pipeline.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def step(n, msg):
    print(f"\n{'='*60}")
    print(f"  Step {n} — {msg}")
    print(f"{'='*60}")


def main():
    t0 = time.time()

    step("1/4", "Loading and filtering M5 data (3 stores, top 100 products)")
    from src.data.loader import run_pipeline as load_data
    df = load_data()

    step("2/4", "Engineering features")
    from src.data.features import build_features
    df = build_features(df)
    df.to_parquet("data/processed/sales_features.parquet", index=False)
    print(f"  Features saved. Shape: {df.shape}")
    del df

    step("3/4", "Training LightGBM model")
    from src.models.train import run_training
    model, metrics = run_training()
    print(f"\n  LightGBM  MAE={metrics['lightgbm']['mae']}  RMSE={metrics['lightgbm']['rmse']}")
    print(f"  Naive     MAE={metrics['naive_baseline']['mae']}  RMSE={metrics['naive_baseline']['rmse']}")

    step("4/4", "Cost simulation")
    import pandas as pd
    from src.cost.simulator import compute_costs, summary_by_store
    val = pd.read_parquet("data/processed/val_predictions.parquet")
    sim = compute_costs(val)
    print(f"\n  Total savings vs naive: €{sim['savings'].sum():,.2f}")
    print(summary_by_store(sim).to_string(index=False))

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  Done in {elapsed/60:.1f} minutes")
    print(f"  Launch the app: streamlit run app/app.py")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
