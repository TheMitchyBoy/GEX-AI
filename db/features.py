"""Feature engineering from Postgres snapshot rows."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

import config


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_timestamp(ts: str) -> datetime:
    return datetime.strptime(ts, "%Y-%m-%d_%H%M%S")


def parse_gamma_flip_value(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, dict):
        flip = safe_float(value.get("flip_strike"), 0.0)
        return flip if flip > 0 else None
    try:
        flip = float(value)
    except (TypeError, ValueError):
        return None
    return flip if flip > 0 else None


def estimate_gamma_flip(cumulative: pd.Series) -> float | None:
    if cumulative.empty:
        return None
    values = cumulative.astype(float).values
    idx = pd.to_numeric(cumulative.index, errors="coerce").to_numpy(dtype=float)
    valid = ~np.isnan(idx)
    if int(np.sum(valid)) < 2:
        return None
    x = idx[valid]
    y = values[valid]
    signs = np.sign(y)
    for i in range(len(signs) - 1):
        if signs[i] == signs[i + 1]:
            continue
        x0, x1 = float(x[i]), float(x[i + 1])
        y0, y1 = float(y[i]), float(y[i + 1])
        if y1 == y0:
            return x0
        return x0 - y0 * (x1 - x0) / (y1 - y0)
    return None


def select_atm_strike_series(
    series: pd.Series,
    spot: float | None,
    *,
    window_pct: float = 0.04,
    min_strikes: int = 5,
) -> pd.Series:
    if series is None or series.empty:
        return pd.Series(dtype=float)
    cleaned = pd.Series(
        pd.to_numeric(series, errors="coerce"),
        index=pd.to_numeric(series.index, errors="coerce"),
    ).dropna()
    cleaned = cleaned[~cleaned.index.isna()].sort_index()
    if cleaned.index.duplicated().any():
        cleaned = cleaned.groupby(level=0).sum()

    spot_val = safe_float(spot, 0.0)
    if spot_val > 0:
        lo, hi = spot_val * (1 - window_pct), spot_val * (1 + window_pct)
        window = cleaned.loc[(cleaned.index >= lo) & (cleaned.index <= hi)]
        if len(window) < min_strikes:
            distances = pd.Series(
                np.abs(cleaned.index.astype(float) - spot_val),
                index=cleaned.index,
            )
            window = cleaned.loc[distances.nsmallest(min(len(cleaned), min_strikes)).index]
        cleaned = window.sort_index()
    return cleaned


def gamma_flip_from_profile(strike_series: pd.Series | None, spot: float | None) -> float | None:
    if strike_series is None or strike_series.empty:
        return None
    series = pd.Series(
        pd.to_numeric(strike_series, errors="coerce"),
        index=pd.to_numeric(strike_series.index, errors="coerce"),
    ).dropna()
    series = series[~series.index.isna()].sort_index()
    if series.index.duplicated().any():
        series = series.groupby(level=0).sum()
    if len(series) < 2:
        return None
    local = select_atm_strike_series(series, spot) if safe_float(spot, 0.0) > 0 else series
    return estimate_gamma_flip(local.cumsum())


def strike_center_of_mass(strike: pd.Series, spot: float | None = None) -> float:
    if strike.empty:
        return safe_float(spot, 0.0)
    weights = strike.abs().values
    total = weights.sum()
    if total <= 0:
        return safe_float(spot, 0.0)
    return float(np.average(strike.index.astype(float), weights=weights))


def top_strike_concentration(strike: pd.Series, top_n: int = 5) -> float:
    if strike.empty:
        return 0.0
    total_abs = strike.abs().sum()
    if total_abs == 0:
        return 0.0
    top = strike.abs().sort_values(ascending=False).head(top_n).sum()
    return float(top / total_abs)


def term_structure_breakdown(
    expirations: pd.Series,
    *,
    snapshot_date: pd.Timestamp | None = None,
    near_term_buckets: int = 3,
) -> dict[str, float]:
    if expirations is None or expirations.empty:
        return {
            "term_total_gex_bn": 0.0,
            "zero_dte_gex_bn": 0.0,
            "zero_dte_ratio": 0.0,
            "near_term_gex_bn": 0.0,
            "near_term_ratio": 0.0,
            "back_term_gex_bn": 0.0,
            "back_term_ratio": 0.0,
            "term_curvature": 0.0,
            "expiration_count": 0.0,
            "front_term_gex_bn": 0.0,
            "front_term_ratio": 0.0,
        }

    exp = pd.Series(expirations, dtype=float).sort_index()
    total = float(exp.sum())
    abs_total = float(exp.abs().sum())
    zero_dte = 0.0
    if snapshot_date is not None:
        idx_dates = pd.to_datetime(exp.index, errors="coerce")
        valid_dates = pd.Series(idx_dates, index=exp.index).dt.date
        snap_date = pd.Timestamp(snapshot_date).date()
        same_day = exp.loc[valid_dates == snap_date]
        if not same_day.empty:
            zero_dte = float(same_day.sum())
    if zero_dte == 0.0 and not exp.empty:
        zero_dte = float(exp.iloc[0])

    near = float(exp.head(max(1, near_term_buckets)).sum())
    back = float(exp.tail(max(1, near_term_buckets)).sum())
    denom = total if total != 0 else abs_total
    front_term = float(exp.iloc[0]) if not exp.empty else 0.0

    return {
        "term_total_gex_bn": total,
        "zero_dte_gex_bn": zero_dte,
        "zero_dte_ratio": zero_dte / denom if denom else 0.0,
        "front_term_gex_bn": front_term,
        "front_term_ratio": front_term / denom if denom else 0.0,
        "near_term_gex_bn": near,
        "near_term_ratio": near / denom if denom else 0.0,
        "back_term_gex_bn": back,
        "back_term_ratio": back / denom if denom else 0.0,
        "term_curvature": near - back,
        "expiration_count": float(len(exp)),
    }


def cumulative_slope_at_spot(cumulative: pd.Series, spot: float) -> float:
    if cumulative.empty or spot <= 0:
        return 0.0
    idx = pd.to_numeric(cumulative.index, errors="coerce").astype(float)
    vals = cumulative.astype(float).values
    valid = ~np.isnan(idx)
    if int(np.sum(valid)) < 2:
        return 0.0
    x = idx[valid].values
    y = vals[valid]
    order = np.argsort(x)
    x, y = x[order], y[order]
    pos = np.searchsorted(x, spot)
    if pos <= 0:
        return float((y[1] - y[0]) / max(x[1] - x[0], 1e-9))
    if pos >= len(x):
        return float((y[-1] - y[-2]) / max(x[-1] - x[-2], 1e-9))
    x0, x1 = float(x[pos - 1]), float(x[pos])
    y0, y1 = float(y[pos - 1]), float(y[pos])
    return float((y1 - y0) / max(x1 - x0, 1e-9))


def extract_surface_vector(
    strike: pd.Series,
    spot: float | None = None,
    window_pct: float = 0.05,
    n_bins: int | None = None,
) -> np.ndarray:
    n_bins = n_bins or config.SURFACE_BINS
    if strike.empty:
        return np.zeros(n_bins, dtype=float)
    spot = safe_float(spot, float(np.median(strike.index.astype(float))) if len(strike) else 0.0)
    near = select_atm_strike_series(strike, spot, window_pct=window_pct, min_strikes=5)
    if near.empty:
        near = strike
    lower = spot * (1 - window_pct)
    upper = spot * (1 + window_pct)
    edges = np.linspace(lower, upper, n_bins + 1)
    bins = np.zeros(n_bins, dtype=float)
    for s, v in zip(near.index.astype(float).values, near.values.astype(float)):
        bi = int(np.clip(np.searchsorted(edges, s, side="right") - 1, 0, n_bins - 1))
        bins[bi] += v
    norm = np.linalg.norm(bins)
    return bins / norm if norm > 1e-12 else bins


def derive_walls(strike: pd.Series) -> tuple[float, float, float]:
    """Return call_wall, put_wall, max_positive_magnet."""
    if strike.empty:
        return 0.0, 0.0, 0.0
    vals = strike.astype(float)
    call_wall = float(vals.idxmax()) if len(vals) else 0.0
    put_wall = float(vals.idxmin()) if len(vals) else 0.0
    positive = vals[vals > 0]
    magnet = float(positive.idxmax()) if not positive.empty else call_wall
    return call_wall, put_wall, magnet


def expiration_series_from_json(expiration_json: Any) -> pd.Series:
    if not expiration_json:
        return pd.Series(dtype=float)
    if isinstance(expiration_json, dict):
        return pd.Series(expiration_json, dtype=float)
    return pd.Series(dtype=float)


def strike_series_from_strikes_df(strikes_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    if strikes_df.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    strike = strikes_df.set_index("strike")["gex_bn_per_pct"].astype(float)
    cumulative = strikes_df.set_index("strike")["cumulative_gex_bn_per_pct"].astype(float)
    return strike.sort_index(), cumulative.sort_index()


def summary_scalar(summary: dict[str, Any] | None, key: str, default: float = 0.0) -> float:
    if not summary:
        return default
    return safe_float(summary.get(key), default)


def extended_feature_names() -> list[str]:
    return [
        "flow_event_count",
        "flow_buy_ratio",
        "flow_aggressiveness",
        "event_risk_score",
        "is_fomc_week",
        "is_opex_week",
        "net_charm_bn",
        "net_vanna_bn",
        "net_delta_bn",
        "gamma_oi_bn",
        "gamma_vol_bn",
        "vix_level",
        "spy_return",
        "realized_vol",
        "spot_return",
    ]


def apply_summary_fields(metrics: dict[str, Any], summary: dict[str, Any] | None) -> None:
    if not summary:
        return
    for key in extended_feature_names():
        if key in summary and summary[key] is not None:
            metrics[key] = safe_float(summary[key], 0.0)
    if "gamma_flip" in summary:
        metrics.setdefault("gamma_flip", parse_gamma_flip_value(summary.get("gamma_flip")))
    if "net_gamma_regime" in summary and not metrics.get("regime"):
        metrics["regime"] = summary["net_gamma_regime"]
    if "total_gex_bn_per_pct" in summary:
        metrics.setdefault("total_gex", safe_float(summary["total_gex_bn_per_pct"]))


def enrich_snapshot_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    strike = metrics.get("strike", pd.Series(dtype=float))
    cumulative = metrics.get("cumulative", pd.Series(dtype=float))
    spot = safe_float(metrics.get("spot"), float(np.median(strike.index.astype(float))) if len(strike) else 0.0)

    if len(strike):
        call_wall, put_wall, magnet = derive_walls(strike)
        metrics.setdefault("call_wall", call_wall)
        metrics.setdefault("put_wall", put_wall)
        metrics.setdefault("max_positive_magnet", magnet)
        metrics["pos_gex"] = float(strike[strike > 0].sum())
        metrics["neg_gex"] = float(strike[strike < 0].sum())
        metrics["gex_std"] = float(strike.std()) if len(strike) > 1 else 0.0
        metrics["abs_mean"] = float(strike.abs().mean()) if len(strike) else 0.0
    else:
        metrics.setdefault("call_wall", 0.0)
        metrics.setdefault("put_wall", 0.0)
        metrics.setdefault("max_positive_magnet", 0.0)
        metrics["pos_gex"] = max(safe_float(metrics.get("total_gex")), 0.0)
        metrics["neg_gex"] = min(safe_float(metrics.get("total_gex")), 0.0)
        metrics["gex_std"] = 0.0
        metrics["abs_mean"] = abs(safe_float(metrics.get("total_gex")))

    gamma_flip = parse_gamma_flip_value(metrics.get("gamma_flip"))
    if gamma_flip is None and len(cumulative):
        gamma_flip = estimate_gamma_flip(cumulative)
    if gamma_flip is None and len(strike):
        gamma_flip = gamma_flip_from_profile(strike, spot)

    metrics["gamma_flip"] = gamma_flip
    metrics["wall_spread"] = safe_float(metrics.get("call_wall")) - safe_float(metrics.get("put_wall"))
    metrics["gex_concentration"] = top_strike_concentration(strike) if len(strike) else 0.0
    metrics["gex_com"] = strike_center_of_mass(strike, spot) if len(strike) else spot
    metrics["flip_distance_pct"] = (
        (safe_float(gamma_flip) - spot) / spot if gamma_flip is not None and spot > 0 else 0.0
    )
    metrics["cum_slope_at_spot"] = cumulative_slope_at_spot(cumulative, spot) if len(cumulative) and spot > 0 else 0.0
    metrics["surface_vector"] = extract_surface_vector(strike, spot)
    metrics["surface_peak"] = float(strike.abs().max()) if len(strike) else 0.0
    metrics["spot"] = spot
    metrics["ts_label"] = parse_timestamp(metrics["ts"]).strftime("%Y-%m-%d %H:%M:%S")

    # term structure defaults
    for key in (
        "term_total_gex_bn",
        "zero_dte_gex_bn",
        "zero_dte_ratio",
        "front_term_gex_bn",
        "front_term_ratio",
        "near_term_gex_bn",
        "near_term_ratio",
        "back_term_gex_bn",
        "back_term_ratio",
        "term_curvature",
        "expiration_count",
    ):
        metrics[key] = safe_float(metrics.get(key), 0.0)

    for name in extended_feature_names():
        metrics.setdefault(name, 0.0)

    return metrics


def snapshot_feature_vector(row: dict[str, Any]) -> np.ndarray:
    base = [
        row["total_gex"],
        row["pos_gex"],
        row["neg_gex"],
        row["gex_std"],
        row["near_term_ratio"],
        row.get("surface_peak", 0.0),
        safe_float(row.get("call_wall"), 0.0),
        safe_float(row.get("put_wall"), 0.0),
        safe_float(row.get("gamma_flip"), 0.0),
        safe_float(row.get("wall_spread"), 0.0),
        safe_float(row.get("flip_distance_pct"), 0.0),
        safe_float(row.get("total_gex_momentum"), 0.0),
        safe_float(row.get("flip_velocity"), 0.0),
        safe_float(row.get("gex_concentration"), 0.0),
        safe_float(row.get("cum_slope_at_spot"), 0.0),
        safe_float(row.get("zero_dte_ratio"), 0.0),
        safe_float(row.get("back_term_ratio"), 0.0),
        safe_float(row.get("term_curvature"), 0.0),
        safe_float(row.get("expiration_count"), 0.0),
        safe_float(row.get("realized_vol"), 0.0),
        safe_float(row.get("spot_return"), 0.0),
        safe_float(row.get("front_term_ratio"), 0.0),
    ]
    extended = [safe_float(row.get(name), 0.0) for name in extended_feature_names()]
    return np.array(base + extended, dtype=float)


def attach_market_features(enriched: list[dict[str, Any]]) -> None:
    """Compute simple realized vol / spot return from consecutive spots."""
    for i, row in enumerate(enriched):
        if i == 0:
            row["spot_return"] = 0.0
            row["realized_vol"] = 0.0
            continue
        prev = enriched[i - 1]
        prev_spot = safe_float(prev.get("spot"), 0.0)
        spot = safe_float(row.get("spot"), 0.0)
        if prev_spot > 0:
            ret = (spot - prev_spot) / prev_spot
            row["spot_return"] = ret
            row["realized_vol"] = abs(ret)
        else:
            row["spot_return"] = 0.0
            row["realized_vol"] = 0.0


def compute_spot_bias(
    *,
    spot: float,
    predicted_flip: float,
    call_wall: float,
    put_wall: float,
    magnet: float,
    predicted_regime: str,
) -> str:
    targets: list[tuple[float, str]] = []
    if predicted_flip > 0:
        targets.append((predicted_flip, "flip"))
    if call_wall > 0:
        targets.append((call_wall, "call_wall"))
    if put_wall > 0:
        targets.append((put_wall, "put_wall"))
    if magnet > 0:
        targets.append((magnet, "magnet"))

    if not targets or spot <= 0:
        return "neutral"

    nearest = min(targets, key=lambda t: abs(t[0] - spot))
    dist_pct = (nearest[0] - spot) / spot
    if abs(dist_pct) < 0.001:
        return "neutral"
    if predicted_regime.startswith("LONG"):
        return "up" if dist_pct > 0 else "down"
    return "down" if dist_pct > 0 else "up"
