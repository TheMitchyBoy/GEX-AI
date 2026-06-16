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
from db.queries import fetch_intraday_timeline, fetch_snapshot_strikes, get_latest_ts
from models.backtest import run_backtest
from models.predict import predict_next_snapshot, similar_setups

st.set_page_config(page_title="GEX Analytics", layout="wide", page_icon="📊")

REFRESH_SEC = config.FORECAST_POLL_SEC


@st.cache_data(ttl=REFRESH_SEC)
def cached_history(ticker: str, lookback_days: int) -> list[dict]:
    return load_snapshot_history(ticker, lookback_days=lookback_days)


def render_metric_row(latest: dict, forecast: dict | None) -> None:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Spot", f"{safe_float(latest.get('spot')):,.2f}")
    c2.metric("Total GEX", f"{safe_float(latest.get('total_gex')):.3f} Bn$/1%")
    c3.metric("Regime", latest.get("regime") or "—")
    c4.metric("Gamma Flip", f"{safe_float(latest.get('gamma_flip')):,.2f}")
    if forecast:
        c5.metric(
            "Forecast ΔGEX",
            f"{forecast['predicted_delta_gex']:.3f}",
            delta=f"conf {forecast['confidence']:.0%}",
        )
    else:
        c5.metric("Forecast", "Insufficient data")


def render_forecast_card(forecast: dict) -> None:
    st.subheader("Next Snapshot Forecast (~10 min)")
    cols = st.columns(4)
    cols[0].write(f"**Predicted total GEX:** {forecast['predicted_total_gex']:.3f}")
    cols[1].write(f"**Predicted regime:** {forecast['predicted_regime']}")
    cols[2].write(f"**Predicted flip:** {forecast['predicted_flip']:,.2f}")
    cols[3].write(f"**Spot bias:** {forecast['spot_bias']}")

    interval = forecast.get("prediction_interval", {})
    st.progress(min(max(forecast["confidence"], 0.0), 1.0))
    st.caption(
        f"ΔGEX interval [{interval.get('low', 0):.3f}, {interval.get('high', 0):.3f}] · "
        f"Regime flip P={forecast.get('regime_flip_probability', 0):.0%}"
    )


def plot_intraday(timeline: pd.DataFrame, forecast: dict | None, enriched: dict) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=timeline["ts"], y=timeline["spot"], name="Spot", line=dict(color="#2563eb")),
        secondary_y=False,
    )
    if "gamma_flip" in timeline.columns and timeline["gamma_flip"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=timeline["ts"],
                y=timeline["gamma_flip"],
                name="Gamma flip",
                line=dict(color="#f59e0b", dash="dash"),
            ),
            secondary_y=False,
        )
    fig.add_trace(
        go.Scatter(
            x=timeline["ts"],
            y=timeline["total_gex"],
            name="Total GEX",
            line=dict(color="#10b981"),
        ),
        secondary_y=True,
    )
    if forecast and timeline["ts"].iloc[-1]:
        last_ts = timeline["ts"].iloc[-1]
        fig.add_trace(
            go.Scatter(
                x=[last_ts, f"{last_ts}*"],
                y=[timeline["spot"].iloc[-1], safe_float(enriched.get("spot"))],
                name="Predicted path (spot hold)",
                line=dict(color="#ef4444", dash="dot"),
                mode="lines+markers",
            ),
            secondary_y=False,
        )
    fig.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20), legend=dict(orientation="h"))
    fig.update_yaxes(title_text="Spot", secondary_y=False)
    fig.update_yaxes(title_text="GEX (Bn$/1%)", secondary_y=True)
    return fig


def plot_strike_heatmap(strikes_df: pd.DataFrame, spot: float) -> go.Figure:
    fig = go.Figure()
    colors = ["#10b981" if v >= 0 else "#ef4444" for v in strikes_df["gex_bn_per_pct"]]
    fig.add_trace(
        go.Bar(
            x=strikes_df["strike"],
            y=strikes_df["gex_bn_per_pct"],
            marker_color=colors,
            name="GEX / strike",
        )
    )
    fig.add_vline(x=spot, line_dash="dash", line_color="#2563eb", annotation_text="Spot")
    fig.update_layout(height=360, margin=dict(l=20, r=20, t=30, b=20), xaxis_title="Strike", yaxis_title="GEX Bn$/1%")
    return fig


def main() -> None:
    st.title("GEX Prediction & Analytics")
    st.caption("Read-only consumer of Railway Postgres GEX snapshots")

    if not config.DATABASE_URL:
        st.error("DATABASE_URL is not configured. Set it in `.env` and restart.")
        st.stop()

    ticker = st.sidebar.text_input("Ticker", value=config.DEFAULT_TICKER).upper()
    lookback = st.sidebar.slider("Lookback days", 7, 180, config.LOOKBACK_DAYS)
    auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True)

    try:
        with get_connection() as conn:
            latest_ts = get_latest_ts(conn, ticker)
        history = cached_history(ticker, lookback)
    except Exception as exc:
        st.error(f"Database error: {exc}")
        st.stop()

    if len(history) < config.MIN_KNN_SNAPSHOTS:
        st.warning(f"Only {len(history)} snapshots — need at least {config.MIN_KNN_SNAPSHOTS} for forecasts.")

    enriched_latest = enrich_snapshot_metrics(history[-1].copy()) if history else {}
    forecast = predict_next_snapshot(history, lookback_days=lookback) if len(history) >= config.MIN_KNN_SNAPSHOTS else None

    st.sidebar.write(f"Latest ts: `{latest_ts}`")
    st.sidebar.write(f"Snapshots loaded: {len(history)}")

    render_metric_row(enriched_latest, forecast)
    if forecast:
        render_forecast_card(forecast)

    market_date = enriched_latest.get("market_date") or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with get_connection() as conn:
        timeline = fetch_intraday_timeline(conn, ticker, market_date)
        strikes_df = fetch_snapshot_strikes(conn, ticker, enriched_latest["ts"]) if history else pd.DataFrame()

    if not timeline.empty:
        timeline["gamma_flip"] = timeline["summary_json"].apply(
            lambda s: safe_float((s or {}).get("gamma_flip")) if isinstance(s, dict) else 0.0
        )
    else:
        timeline = pd.DataFrame(
            [
                {
                    "ts": h["ts"],
                    "spot": h["spot"],
                    "total_gex": h["total_gex"],
                    "gamma_flip": safe_float(h.get("gamma_flip")),
                }
                for h in history
                if h.get("market_date") == market_date
            ]
        )

    tab1, tab2, tab3, tab4 = st.tabs(["Intraday", "Strike Heatmap", "Similar Setups", "Backtest"])

    with tab1:
        if not timeline.empty:
            st.plotly_chart(plot_intraday(timeline, forecast, enriched_latest), use_container_width=True)
        else:
            st.info("No intraday timeline for selected market date.")

    with tab2:
        if not strikes_df.empty:
            st.plotly_chart(
                plot_strike_heatmap(strikes_df, safe_float(enriched_latest.get("spot"))),
                use_container_width=True,
            )
            call_wall = enriched_latest.get("call_wall")
            put_wall = enriched_latest.get("put_wall")
            st.write(f"Call wall: **{call_wall:,.0f}** · Put wall: **{put_wall:,.0f}**")
        else:
            st.info("No strike profile for latest snapshot.")

    with tab3:
        setups = similar_setups(history, lookback_days=lookback) if len(history) >= 3 else []
        if setups:
            st.dataframe(pd.DataFrame(setups), use_container_width=True, hide_index=True)
        else:
            st.info("Not enough history for similarity search.")

    with tab4:
        if len(history) >= config.MIN_KNN_SNAPSHOTS + 5:
            report = run_backtest(history, lookback_days=min(lookback, 30))
            st.json(report.to_dict())
            if report.rows:
                st.dataframe(pd.DataFrame(report.rows).tail(20), use_container_width=True, hide_index=True)
        else:
            st.info("Need more snapshots for backtest.")

    if forecast and forecast.get("last_move_attribution"):
        with st.expander("Structural attribution (last move)"):
            st.json(forecast["last_move_attribution"])

    if auto_refresh:
        time.sleep(REFRESH_SEC)
        st.rerun()


if __name__ == "__main__":
    main()
