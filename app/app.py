"""
app.py — SmartShelf Streamlit dashboard
Run: streamlit run app/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.features import get_feature_columns
from src.models.predict import forecast_item, compute_order_quantity
from src.cost.simulator import compute_costs, summary_by_store, summary_by_category

st.set_page_config(page_title="SmartShelf", page_icon="📦", layout="wide")
st.markdown("""
<style>
.big-number { font-size:2.2rem; font-weight:600; color:#1D9E75; }
.label      { font-size:0.82rem; color:#888; margin-bottom:2px; }
</style>
""", unsafe_allow_html=True)

PROCESSED = Path("data/processed")
MODELS    = Path("outputs/models")

# ── cached loaders ─────────────────────────────────────────────────────────────
@st.cache_data
def load_features():
    return pd.read_parquet(PROCESSED / "sales_features.parquet")

@st.cache_data
def load_predictions():
    return pd.read_parquet(PROCESSED / "val_predictions.parquet")

@st.cache_resource
def load_model():
    return joblib.load(MODELS / "lgbm_model.pkl")

@st.cache_resource
def load_encoders():
    return joblib.load(MODELS / "label_encoders.pkl")

@st.cache_data
def load_metrics():
    with open(MODELS / "metrics.json") as f:
        return json.load(f)

# ── sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.title("📦 SmartShelf")
st.sidebar.caption("AI-powered replenishment assistant")
st.sidebar.divider()
page = st.sidebar.radio("Navigation",
    ["🛒 Order Assistant", "💶 Cost Dashboard", "🔬 Model Insights"])
st.sidebar.divider()
st.sidebar.markdown("**Cost assumptions**")
holding_rate  = st.sidebar.slider("Holding cost (%/day)", 0.01, 0.20, 0.068, 0.001, format="%.3f%%") / 100
stockout_rate = st.sidebar.slider("Stockout cost (% of price)", 10, 150, 75, 5, format="%d%%") / 100

# ══════════════════════════════════════════════════════════════════════════════
if page == "🛒 Order Assistant":
    st.title("🛒 Order Assistant")
    st.caption("Select a store and product to get today's replenishment recommendation.")

    try:
        df = load_features()
        df["date"] = pd.to_datetime(df["date"])

        c1, c2, c3 = st.columns(3)
        store = c1.selectbox("Store", sorted(df["store_id"].unique()))
        items = sorted(df[df["store_id"] == store]["item_id"].unique())
        item  = c2.selectbox("Product", items)
        stock = c3.number_input("Current stock (units)", min_value=0, value=10, step=1)

        st.divider()

        item_df = df[(df["item_id"] == item) & (df["store_id"] == store)].copy()

        with st.spinner("Generating forecast ..."):
            forecast_df = forecast_item(item_df, horizon=7)
        order = compute_order_quantity(forecast_df, current_stock=stock)

        # ── Headline metrics ──────────────────────────────────────────────────
        ca, cb, cc, cd = st.columns(4)
        ca.markdown(f"<div class='label'>Recommended order</div><div class='big-number'>{order['order_quantity']} units</div>", unsafe_allow_html=True)
        cb.metric("7-day demand forecast", f"{order['expected_7d_demand']} units")
        cc.metric("Daily average (28d)",   f"{order['daily_avg']} units")
        cd.metric("Safety stock buffer",   f"{order['safety_stock']} units")

        # ── Why ───────────────────────────────────────────────────────────────
        st.subheader("Why this recommendation?")
        daily_avg  = order["daily_avg"]
        days_cover = stock / daily_avg if daily_avg > 0 else 99
        reasons = []
        if days_cover < 3:
            reasons.append(f"Stock critically low — only {days_cover:.1f} days of cover remaining")
        elif days_cover < 7:
            reasons.append(f"Stock covers {days_cover:.1f} days — replenishment needed soon")
        if pd.Timestamp.now().dayofweek >= 4:
            reasons.append("Weekend approaching — demand typically 15-25% higher")
        if item_df["has_event"].tail(7).sum() > 0:
            reasons.append("Special event detected nearby — demand may spike")
        snap_col = {"CA": "snap_CA", "TX": "snap_TX", "WI": "snap_WI"}.get(store.split("_")[0], None)
        if snap_col and snap_col in item_df.columns and item_df[snap_col].tail(3).sum() > 0:
            reasons.append("SNAP benefit days upcoming — historically boosts food sales")
        if not reasons:
            reasons.append("Demand pattern is stable — standard safety stock replenishment")
        for r in reasons:
            st.markdown(f"• {r}")

        # ── Forecast chart ────────────────────────────────────────────────────
        st.subheader("Sales history + 7-day forecast")
        hist = item_df.tail(30)[["date", "sales"]]
        fig  = go.Figure()
        fig.add_trace(go.Scatter(x=hist["date"], y=hist["sales"],
            mode="lines", name="Historical sales",
            line=dict(color="#888780", width=1.5)))
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast_df["date"], forecast_df["date"][::-1]]),
            y=pd.concat([forecast_df["upper"], forecast_df["lower"][::-1]]),
            fill="toself", fillcolor="rgba(29,158,117,0.1)",
            line=dict(color="rgba(0,0,0,0)"), name="Confidence band"))
        fig.add_trace(go.Scatter(x=forecast_df["date"], y=forecast_df["forecast"],
            mode="lines+markers", name="Forecast",
            line=dict(color="#1D9E75", width=2.5), marker=dict(size=7)))
        fig.update_layout(height=320, margin=dict(l=0,r=0,t=10,b=0),
            yaxis_title="Units", hovermode="x unified",
            legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, use_container_width=True)

    except FileNotFoundError:
        st.error("Run `python run_pipeline.py` first.")
    except Exception as e:
        st.error(f"Error: {e}")
        import traceback; st.code(traceback.format_exc())

# ══════════════════════════════════════════════════════════════════════════════
elif page == "💶 Cost Dashboard":
    st.title("💶 Cost Dashboard")
    st.caption("Business impact of SmartShelf vs. naive 'order same as last month' baseline.")

    try:
        val = load_predictions()
        sim = compute_costs(val, holding_rate, stockout_rate)

        total_naive   = sim["naive_total_cost"].sum()
        total_lgbm    = sim["lgbm_total_cost"].sum()
        total_savings = sim["savings"].sum()
        pct           = total_savings / total_naive * 100 if total_naive > 0 else 0

        ca, cb, cc, cd = st.columns(4)
        ca.metric("Naive baseline cost", f"€{total_naive:,.0f}")
        cb.metric("SmartShelf cost",     f"€{total_lgbm:,.0f}")
        cc.metric("Total savings",       f"€{total_savings:,.0f}", delta=f"{pct:.1f}% less")
        cd.metric("Days evaluated",      str(sim["date"].nunique()))

        st.divider()
        cl, cr = st.columns(2)

        with cl:
            st.subheader("Savings by store")
            sdf = summary_by_store(sim)
            fig = px.bar(sdf, x="store_id", y="savings",
                color="savings", color_continuous_scale=["#9FE1CB","#085041"],
                labels={"savings":"€ saved","store_id":"Store"})
            fig.update_layout(height=300, margin=dict(l=0,r=0,t=10,b=0),
                coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        with cr:
            st.subheader("Savings by category")
            cdf = summary_by_category(sim)
            fig2 = px.pie(cdf, names="cat", values="savings",
                color_discrete_sequence=["#1D9E75","#5DCAA5","#9FE1CB"])
            fig2.update_layout(height=300, margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Daily cost over time")
        sim["date"] = pd.to_datetime(sim["date"])
        daily = sim.groupby("date")[["naive_total_cost","lgbm_total_cost"]].sum().reset_index()
        fig3  = go.Figure()
        fig3.add_trace(go.Scatter(x=daily["date"], y=daily["naive_total_cost"],
            name="Naive", line=dict(color="#E24B4A", width=1.5)))
        fig3.add_trace(go.Scatter(x=daily["date"], y=daily["lgbm_total_cost"],
            name="SmartShelf", line=dict(color="#1D9E75", width=2)))
        fig3.update_layout(height=300, margin=dict(l=0,r=0,t=10,b=0),
            yaxis_title="Daily cost (€)", hovermode="x unified",
            legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig3, use_container_width=True)

    except FileNotFoundError:
        st.error("Run `python run_pipeline.py` first.")
    except Exception as e:
        st.error(f"Error: {e}")

# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Model Insights":
    st.title("🔬 Model Insights")

    try:
        metrics = load_metrics()
        st.subheader("Performance vs. naive baseline")
        ca, cb, cc, cd = st.columns(4)
        ca.metric("SmartShelf MAE",  metrics["lightgbm"]["mae"])
        cb.metric("SmartShelf RMSE", metrics["lightgbm"]["rmse"])
        cc.metric("Naive MAE",  metrics["naive_baseline"]["mae"],
            delta=f"{metrics['lightgbm']['mae']-metrics['naive_baseline']['mae']:.4f}",
            delta_color="inverse")
        cd.metric("Naive RMSE", metrics["naive_baseline"]["rmse"],
            delta=f"{metrics['lightgbm']['rmse']-metrics['naive_baseline']['rmse']:.4f}",
            delta_color="inverse")

        st.divider()
        st.subheader("Feature importance")
        model        = load_model()
        feature_cols = get_feature_columns()
        fi = pd.DataFrame({
            "feature":    feature_cols,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False).head(15)
        fig = px.bar(fi, x="importance", y="feature", orientation="h",
            color="importance", color_continuous_scale=["#9FE1CB","#085041"])
        fig.update_layout(height=420, margin=dict(l=0,r=0,t=10,b=0),
            yaxis=dict(autorange="reversed"), coloraxis_showscale=False,
            xaxis_title="Importance", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    except FileNotFoundError:
        st.error("Run `python run_pipeline.py` first.")
    except Exception as e:
        st.error(f"Error: {e}")
