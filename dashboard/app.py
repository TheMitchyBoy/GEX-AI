"""Streamlit dashboard for GEX prediction and analytics."""

from __future__ import annotations

import time
from datetime import datetime, timezone

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import config
from db.connection import get_connection
from db.features import enrich_snapshot_metrics, safe_float
from db.loader import load_snapshot_history
from db.queries import fetch_calibration_stats, fetch_daily_insights, fetch_intraday_timeline, fetch_snapshot_strikes, get_latest_ts
from models.backtest import run_backtest
from models.llm_predict import generate_llm_forecast
from models.multi_horizon import predict_multi_horizon
from models.predict import predict_next_snapshot, similar_setups

st.set_page_config(page_title="GEX Analytics", layout="wide", page_icon="📊")
REFRESH_SEC = config.FORECAST_POLL_SEC


@st.cache_data(ttl=REFRESH_SEC)
def cached_history(ticker: str, lookback_days: int) -> list[dict]:
    return load_snapshot_history(ticker, lookback_days=lookback_days)


def render_compare(knn: dict | None, llm: dict | None) -> None:
    st.subheader("KNN vs LLM")
    c1, c2, c3 = st.columns(3)
    if knn:
        c1.metric("KNN ΔGEX", f"{knn.get('predicted_delta_gex', 0):.3f}", knn.get("predicted_regime"))
    if llm:
        c2.metric("LLM ΔGEX", f"{llm.get('predicted_delta_gex_bn', 0):.3f}", llm.get("predicted_regime"))
        c2.caption(f"bias: {llm.get('spot_bias')} · conf {llm.get('confidence', 0):.0%}")
    if knn and llm:
        agree = knn.get("predicted_regime") == llm.get("predicted_regime")
        c3.metric("Regime agreement", "Yes" if agree else "No")


def plot_intraday(timeline: pd.DataFrame, forecast: dict | None, enriched: dict) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=timeline["ts"], y=timeline["spot"], name="Spot", line=dict(color="#2563eb")), secondary_y=False)
    if "gamma_flip" in timeline.columns and timeline["gamma_flip"].notna().any():
        fig.add_trace(go.Scatter(x=timeline["ts"], y=timeline["gamma_flip"], name="Gamma flip", line=dict(color="#f59e0b", dash="dash")), secondary_y=False)
    fig.add_trace(go.Scatter(x=timeline["ts"], y=timeline["total_gex"], name="Total GEX", line=dict(color="#10b981")), secondary_y=True)
    fig.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20))
    fig.update_yaxes(title_text="Spot", secondary_y=False)
    fig.update_yaxes(title_text="GEX (Bn$/1%)", secondary_y=True)
    return fig


def plot_strike_heatmap(strikes_df: pd.DataFrame, spot: float) -> go.Figure:
    colors = ["#10b981" if v >= 0 else "#ef4444" for v in strikes_df["gex_bn_per_pct"]]
    fig = go.Figure(go.Bar(x=strikes_df["strike"], y=strikes_df["gex_bn_per_pct"], marker_color=colors))
    fig.add_vline(x=spot, line_dash="dash", line_color="#2563eb", annotation_text="Spot")
    fig.update_layout(height=360, xaxis_title="Strike", yaxis_title="GEX Bn$/1%")
    return fig


@st.fragment(run_every=REFRESH_SEC)
def live_panel(ticker: str, lookback: int) -> tuple[list, dict, dict | None, dict | None]:
    history = cached_history(ticker, lookback)
    enriched = enrich_snapshot_metrics(history[-1].copy()) if history else {}
    forecast = predict_next_snapshot(history, lookback_days=lookback) if len(history) >= config.MIN_KNN_SNAPSHOTS else None
    llm = generate_llm_forecast(history, lookback_days=lookback, persist=False) if len(history) >= config.MIN_KNN_SNAPSHOTS else None
    return history, enriched, forecast, llm


def main() -> None:
    st.title("GEX Prediction & Analytics")
    if not config.DATABASE_URL:
        st.error("DATABASE_URL is not configured.")
        st.stop()

    ticker = st.sidebar.selectbox("Ticker", config.SUPPORTED_TICKERS, index=config.SUPPORTED_TICKERS.index(config.DEFAULT_TICKER) if config.DEFAULT_TICKER in config.SUPPORTED_TICKERS else 0)
    lookback = st.sidebar.slider("Lookback days", 7, 180, config.LOOKBACK_DAYS)
    st.sidebar.caption(f"Auto-refresh every {REFRESH_SEC}s")

    try:
        with get_connection() as conn:
            latest_ts = get_latest_ts(conn, ticker)
        history, enriched, forecast, llm = live_panel(ticker, lookback)
    except Exception as exc:
        st.error(f"Database error: {exc}")
        st.stop()

    st.sidebar.write(f"Latest ts: `{latest_ts}` · {len(history)} snapshots")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Spot", f"{safe_float(enriched.get('spot')):,.2f}")
    c2.metric("Total GEX", f"{safe_float(enriched.get('total_gex')):.3f}")
    c3.metric("Regime", enriched.get("regime") or "—")
    c4.metric("Gamma Flip", f"{safe_float(enriched.get('gamma_flip')):,.2f}")
    c5.metric("Forecast ΔGEX", f"{forecast['predicted_delta_gex']:.3f}" if forecast else "—")

    render_compare(forecast, llm)

    if forecast:
        st.progress(min(max(forecast["confidence"], 0.0), 1.0))
        horizons = predict_multi_horizon(history, lookback_days=lookback)
        if len(horizons) > 1:
            st.caption("Multi-horizon: " + " · ".join(f"h{h}={v['predicted_delta_gex']:.3f}" for h, v in horizons.items()))

    market_date = enriched.get("market_date") or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with get_connection() as conn:
        timeline = fetch_intraday_timeline(conn, ticker, market_date)
        strikes_df = fetch_snapshot_strikes(conn, ticker, enriched["ts"]) if history else pd.DataFrame()
        cal = fetch_calibration_stats(conn, ticker)
        insights = fetch_daily_insights(conn, ticker, limit=3)

    if not timeline.empty:
        timeline["gamma_flip"] = timeline["summary_json"].apply(lambda s: safe_float((s or {}).get("gamma_flip")) if isinstance(s, dict) else 0.0)

    tabs = st.tabs(["Intraday", "Strikes", "Similar", "Backtest", "LLM", "Accuracy"])
    with tabs[0]:
        if not timeline.empty:
            st.plotly_chart(plot_intraday(timeline, forecast, enriched), use_container_width=True)
    with tabs[1]:
        if not strikes_df.empty:
            st.plotly_chart(plot_strike_heatmap(strikes_df, safe_float(enriched.get("spot"))), use_container_width=True)
    with tabs[2]:
        setups = similar_setups(history, lookback_days=lookback) if len(history) >= 3 else []
        if setups:
            st.dataframe(pd.DataFrame(setups), use_container_width=True, hide_index=True)
    with tabs[3]:
        if len(history) >= config.MIN_KNN_SNAPSHOTS + 5:
            st.json(run_backtest(history, lookback_days=min(lookback, 30)).to_dict())
    with tabs[4]:
        if llm:
            st.write(llm.get("reasoning", ""))
            for p in llm.get("predictions", []):
                st.markdown(f"- {p}")
            st.json(llm)
    with tabs[5]:
        st.json(cal)
        if insights:
            st.subheader("Daily insights")
            for row in insights:
                st.json(row.get("payload_json"))


if __name__ == "__main__":
    main()
