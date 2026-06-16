"""Synthetic snapshot data for local testing without Postgres."""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd


def generate_synthetic_history(
    *,
    ticker: str = "SPX",
    n_snapshots: int = 120,
    base_spot: float = 5500.0,
    interval_minutes: int = 10,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Build in-memory history mimicking processor output."""
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    start = datetime(2026, 5, 1, 14, 30, 0)
    history: list[dict[str, Any]] = []
    spot = base_spot
    total_gex = -0.2

    for i in range(n_snapshots):
        ts_dt = start + timedelta(minutes=interval_minutes * i)
        if ts_dt.hour >= 21:
            start = ts_dt.replace(hour=14, minute=30) + timedelta(days=1)
            ts_dt = start + timedelta(minutes=interval_minutes * (i % 40))
        ts = ts_dt.strftime("%Y-%m-%d_%H%M%S")
        market_date = ts_dt.strftime("%Y-%m-%d")

        spot += np_rng.normal(0, 3.0)
        total_gex += np_rng.normal(0, 0.02)
        regime = "LONG gamma" if total_gex >= 0 else "SHORT gamma"
        flip = spot - 30 + math.sin(i / 8) * 20

        strikes = np.arange(int(spot - 150), int(spot + 155), 5, dtype=float)
        gex = np_rng.normal(0, 0.01, size=len(strikes))
        peak_idx = int(np.clip(len(strikes) // 2 + rng.randint(-5, 5), 0, len(strikes) - 1))
        gex[peak_idx] += 0.08 if total_gex >= 0 else -0.08
        cumulative = np.cumsum(gex)
        strike = pd.Series(gex, index=strikes)
        cum = pd.Series(cumulative, index=strikes)

        exp_dates = [
            (ts_dt + timedelta(days=d)).strftime("%Y-%m-%d")
            for d in (0, 1, 7, 14, 30)
        ]
        exp = {d: float(np_rng.normal(total_gex / 5, 0.01)) for d in exp_dates}

        summary = {
            "spot": spot,
            "total_gex_bn_per_pct": total_gex,
            "net_gamma_regime": regime,
            "gamma_flip": flip,
            "gamma_oi_bn": total_gex,
            "gamma_vol_bn": total_gex * 0.01,
            "net_charm_bn": float(np_rng.normal(500000, 50000)),
            "net_vanna_bn": float(np_rng.normal(-900000, 80000)),
            "net_delta_bn": float(np_rng.normal(450000, 60000)),
            "flow_event_count": rng.randint(0, 8),
            "flow_buy_ratio": rng.random(),
            "flow_aggressiveness": rng.randint(10, 80),
            "event_risk_score": round(rng.random() * 0.5, 2),
            "is_fomc_week": 1 if i % 50 < 5 else 0,
            "interval_minutes": interval_minutes,
        }

        history.append(
            {
                "ticker": ticker,
                "ts": ts,
                "market_date": market_date,
                "spot": spot,
                "total_gex": total_gex,
                "regime": regime,
                "strike": strike,
                "cumulative": cum,
                "summary": summary,
                "expiration_json": exp,
                "term_total_gex_bn": sum(exp.values()),
                "zero_dte_gex_bn": exp[exp_dates[0]],
                "zero_dte_ratio": 0.2,
                "near_term_gex_bn": sum(list(exp.values())[:2]),
                "near_term_ratio": 0.45,
                "back_term_gex_bn": sum(list(exp.values())[-2:]),
                "back_term_ratio": 0.35,
                "term_curvature": 0.05,
                "expiration_count": float(len(exp)),
                "gamma_flip": flip,
            }
        )
    return history
