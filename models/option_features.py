"""Feature vectors for option price movement learning (GEX DB + UW quotes)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from db.features import safe_float

FEATURE_KEYS = [
    "mid_price",
    "spread_pct",
    "implied_volatility",
    "volume",
    "open_interest",
    "moneyness",
    "dte",
    "gex_at_strike",
    "total_gex",
    "gamma_flip",
    "flip_distance_pct",
    "spot",
    "flow_buy_ratio",
    "flow_event_count",
    "flow_aggressiveness",
    "iv_x_moneyness",
    "gex_x_moneyness",
]


def _days_to_expiry(expiry: str, ref_date: str | None = None) -> int:
    try:
        exp = datetime.strptime(expiry[:10], "%Y-%m-%d").date()
        ref = datetime.strptime((ref_date or datetime.utcnow().strftime("%Y-%m-%d"))[:10], "%Y-%m-%d").date()
        return max((exp - ref).days, 0)
    except ValueError:
        return 0


def build_quote_row(
    *,
    ticker: str,
    snapshot: dict[str, Any],
    uw_ticker: str,
    slot: str,
    contract: dict[str, Any],
    parsed: dict[str, Any],
    mid: float,
    gex_strike: float | None,
    quote_ts: str,
    flow_features: dict[str, Any] | None = None,
) -> dict[str, Any]:
    spot = safe_float(snapshot.get("spot"))
    strike = safe_float(parsed.get("strike"))
    summary = snapshot.get("summary_json") or {}
    if isinstance(summary, str):
        import json
        try:
            summary = json.loads(summary)
        except json.JSONDecodeError:
            summary = {}
    bid = safe_float(contract.get("nbbo_bid"))
    ask = safe_float(contract.get("nbbo_ask"))
    spread_pct = (ask - bid) / mid if mid > 0 and ask > bid else 0.0
    expiry = parsed.get("expiry") or ""
    market_date = snapshot.get("market_date") or quote_ts[:10]
    return {
        "ticker": ticker.upper(),
        "snapshot_ts": snapshot.get("ts"),
        "quote_ts": quote_ts,
        "slot": slot,
        "uw_ticker": uw_ticker,
        "option_symbol": contract.get("option_symbol"),
        "option_type": parsed.get("option_type"),
        "expiry": expiry,
        "strike": strike,
        "spot": spot,
        "mid_price": mid,
        "last_price": safe_float(contract.get("last_price")),
        "nbbo_bid": bid,
        "nbbo_ask": ask,
        "implied_volatility": safe_float(contract.get("implied_volatility")),
        "volume": int(safe_float(contract.get("volume"))),
        "open_interest": int(safe_float(contract.get("open_interest"))),
        "moneyness": strike / spot if spot > 0 else 1.0,
        "dte": _days_to_expiry(expiry, market_date),
        "gex_at_strike": safe_float(gex_strike),
        "total_gex": safe_float(snapshot.get("total_gex")),
        "gamma_flip": safe_float(summary.get("gamma_flip") or snapshot.get("gamma_flip")),
        "flow_features": flow_features or {},
        "source": "unusual_whales",
    }


def feature_dict(quote: dict[str, Any]) -> dict[str, float]:
    spot = safe_float(quote.get("spot"))
    flip = safe_float(quote.get("gamma_flip"))
    mid = safe_float(quote.get("mid_price"))
    bid = safe_float(quote.get("nbbo_bid"))
    ask = safe_float(quote.get("nbbo_ask"))
    spread_pct = (ask - bid) / mid if mid > 0 and ask > bid else safe_float(quote.get("spread_pct"))
    moneyness = safe_float(quote.get("moneyness"), 1.0)
    iv = safe_float(quote.get("implied_volatility"))
    flow = quote.get("flow_features") or {}
    if isinstance(flow, str):
        import json
        try:
            flow = json.loads(flow)
        except json.JSONDecodeError:
            flow = {}
    raw = {
        "mid_price": mid,
        "spread_pct": spread_pct,
        "implied_volatility": iv,
        "volume": safe_float(quote.get("volume")),
        "open_interest": safe_float(quote.get("open_interest")),
        "moneyness": moneyness,
        "dte": safe_float(quote.get("dte")),
        "gex_at_strike": safe_float(quote.get("gex_at_strike")),
        "total_gex": safe_float(quote.get("total_gex")),
        "gamma_flip": flip,
        "flip_distance_pct": abs(spot - flip) / spot if spot > 0 and flip > 0 else 0.0,
        "spot": spot,
        "flow_buy_ratio": safe_float(flow.get("flow_buy_ratio") or flow.get("buy_ratio")),
        "flow_event_count": safe_float(flow.get("flow_event_count")),
        "flow_aggressiveness": safe_float(flow.get("flow_aggressiveness")),
        "iv_x_moneyness": iv * moneyness,
        "gex_x_moneyness": safe_float(quote.get("gex_at_strike")) * moneyness,
    }
    return {k: safe_float(raw.get(k)) for k in FEATURE_KEYS}
