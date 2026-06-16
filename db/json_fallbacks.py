"""Parse surface_json and greek_exposure_json when snapshot_strikes is sparse."""

from __future__ import annotations

from typing import Any

import pandas as pd

from db.features import safe_float


def _rows_from_json_blob(blob: Any) -> list[dict[str, Any]]:
    if not blob:
        return []
    if isinstance(blob, str):
        import json

        try:
            blob = json.loads(blob)
        except json.JSONDecodeError:
            return []
    if isinstance(blob, list):
        return [r for r in blob if isinstance(r, dict)]
    if isinstance(blob, dict):
        if "rows" in blob and isinstance(blob["rows"], list):
            return [r for r in blob["rows"] if isinstance(r, dict)]
        return [blob]
    return []


def strike_series_from_surface_json(surface_json: Any) -> pd.Series:
    rows = _rows_from_json_blob(surface_json)
    if not rows:
        return pd.Series(dtype=float)
    values: dict[float, float] = {}
    for row in rows:
        strike = safe_float(row.get("strike") or row.get("Strike"), 0.0)
        if strike <= 0:
            continue
        gex = row.get("net_gex") or row.get("GEX") or row.get("gex_bn_per_pct")
        if gex is None and "call_gex" in row and "put_gex" in row:
            gex = safe_float(row.get("call_gex")) + safe_float(row.get("put_gex"))
        if gex is not None:
            values[strike] = values.get(strike, 0.0) + safe_float(gex)
    if not values:
        return pd.Series(dtype=float)
    return pd.Series(values, dtype=float).sort_index()


def strike_series_from_greek_json(greek_json: Any) -> pd.Series:
    rows = _rows_from_json_blob(greek_json)
    if not rows:
        return pd.Series(dtype=float)
    values: dict[float, float] = {}
    for row in rows:
        strike = safe_float(row.get("strike"), 0.0)
        if strike <= 0:
            continue
        for key in ("net_gex", "GEX", "gamma", "gex_bn_per_pct"):
            if key in row and row[key] is not None:
                values[strike] = values.get(strike, 0.0) + safe_float(row[key])
                break
        else:
            if "call_gex" in row or "put_gex" in row:
                values[strike] = values.get(strike, 0.0) + safe_float(row.get("call_gex")) + safe_float(
                    row.get("put_gex")
                )
    if not values:
        return pd.Series(dtype=float)
    return pd.Series(values, dtype=float).sort_index()


def resolve_strike_series(
    strikes_df: pd.DataFrame | None,
    row: dict[str, Any],
) -> tuple[pd.Series, pd.Series]:
    """Prefer snapshot_strikes; fall back to surface/greek JSON columns."""
    if strikes_df is not None and not strikes_df.empty:
        strike = strikes_df.set_index("strike")["gex_bn_per_pct"].astype(float).sort_index()
        cumulative = strikes_df.set_index("strike")["cumulative_gex_bn_per_pct"].astype(float).sort_index()
        return strike, cumulative

    strike = strike_series_from_surface_json(row.get("surface_json"))
    if strike.empty:
        strike = strike_series_from_greek_json(row.get("greek_exposure_json"))

    if strike.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    cumulative = strike.cumsum()
    return strike, cumulative
