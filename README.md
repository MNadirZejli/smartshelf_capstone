# SmartShelf — AI Replenishment Assistant

> An ML-powered tool that tells store managers exactly what to reorder tomorrow, how many units, and why — reducing overstock and stockout costs vs. naive forecasting.

---

## Project overview

SmartShelf uses 5 years of real Walmart sales data (M5 dataset) to predict demand for the next 7 days per product per store. It then translates forecast errors into real € inventory costs and compares against a naive baseline to quantify business value.

**Scope:** 3 stores (CA_1, TX_1, WI_1) × top 100 products by sales volume × 1941 days.

---

## Project structure

```
smartshelf/
├── data/
│   ├── raw/                  ← Put your M5 CSV files here (not tracked by git)
│   └── processed/            ← Auto-generated parquet files (not tracked by git)
├── src/
│   ├── data/
│   │   ├── loader.py         ← Filters M5 to 3 stores + top 100 products
│   │   └── features.py       ← Feature engineering (lags, rolling, price, calendar)
│   ├── models/
│   │   ├── train.py          ← LightGBM training + naive baseline comparison
│   │   └── predict.py        ← 7-day forecast + order quantity logic
│   ├── cost/
│       └── simulator.py      ← Overstock/stockout cost engine
│  
├── app/
│   └── app.py                ← Streamlit dashboard (3 pages)
├
│           
├── outputs/
│   └── models/               ← Saved model + metrics (not tracked by git)
├── run_pipeline.py           ← Run everything in one command
└── requirements.txt
```

---

## Setup

```bash
# 1. Clone the repo

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add M5 data files to data/raw/
#    calendar.csv
#    sales_train_evaluation.csv
#    sales_train_validation.csv
#    sell_prices.csv
#    sample_submission.csv
```

---

## Run

```bash
# Run the full pipeline (loads data, engineers features, trains model, simulates costs)
python run_pipeline.py

# Launch the Streamlit app
streamlit run app/app.py
```

Pipeline runs in **~5 minutes** on a standard laptop.

---

## App pages

| Page | Description |
|---|---|
| Order Assistant | Select store + product → 7-day forecast, order recommendation, plain-language explanation |
| Cost Dashboard | Total € saved vs naive baseline, broken down by store and category |
| Model Insights | Feature importance, MAE/RMSE comparison |

---

## Cost model

| Parameter | Default | Source |
|---|---|---|
| Holding cost | 0.068%/day (25%/year) | Silver, Pyke & Thomas (1998) |
| Stockout cost | 75% of item value | ECR Europe (2003) |

Both are adjustable in the app sidebar.

---

## Dataset

M5 Forecasting Competition — Walmart sales data
- **3 stores:** CA_1 (California), TX_1 (Texas), WI_1 (Wisconsin)
- **100 products:** Top 100 by total sales volume
- **1,941 days:** January 2011 – May 2016

---
