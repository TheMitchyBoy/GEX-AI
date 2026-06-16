"""Ingest UW option quotes + learn/predict price movements from DB GEX context."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import config
from db.connection import get_connection
from db.option_queries import (
    fetch_latest_option_quotes,
    fetch_option_quotes,
    gex_at_strike,
    insert_option_prediction,
    upsert_option_quote,
    utc_now_iso,
)
from db.queries import ensure_extensions, fetch_latest_snapshot
from integrations.uw_client import (
    UnusualWhalesClient,
    contract_mid,
    is_configured,
    nearest_expiry,
    parse_option_symbol,
    pick_atm_contracts,
    uw_ticker_for,
)
from models.option_features import build_quote_row
from models.option_learn import SLOTS, ensure_bootstrapped, maybe_learn_latest, model_status, predict_delta_mid

logger = logging.getLogger(__name__)

SLOT_MAP = {"call": "atm_call", "put": "atm_put"}


def _flow_from_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    summary = snapshot.get("summary_json") or {}
    if not isinstance(summary, dict):
        return {}
    return {
        "flow_buy_ratio": summary.get("flow_buy_ratio"),
        "flow_event_count": summary.get("flow_event_count"),
        "flow_aggressiveness": summary.get("flow_aggressiveness"),
        "event_risk_score": summary.get("event_risk_score"),
    }


def ingest_uw_quotes(ticker: str, *, client: UnusualWhalesClient | None = None) -> dict[str, Any]:
    """Pull ATM option quotes from UW and store with GEX snapshot context."""
    if not is_configured():
        return {"ok": False, "error": "UW_API_KEY is not set"}
    ticker = ticker.upper()
    uw_ticker = uw_ticker_for(ticker)
    client = client or UnusualWhalesClient()
    quote_ts = utc_now_iso()

    with get_connection() as conn:
        ensure_extensions(conn)
        snapshot = fetch_latest_snapshot(conn, ticker)
        if not snapshot:
            return {"ok": False, "error": f"No GEX snapshots for {ticker}"}

        expiry = nearest_expiry(snapshot.get("expiration_json"), snapshot.get("market_date"))
        if not expiry:
            return {"ok": False, "error": "No expiration_json on latest snapshot"}

        try:
            contracts = client.option_contracts(uw_ticker, expiry=expiry)
        except Exception as exc:
            logger.exception("UW option-contracts failed for %s", uw_ticker)
            return {"ok": False, "error": str(exc)}

        spot = float(snapshot.get("spot") or 0)
        picked = pick_atm_contracts(contracts, spot)
        if not picked:
            return {"ok": False, "error": "No ATM contracts returned from UW", "expiry": expiry}

        flow = _flow_from_snapshot(snapshot)
        stored: list[dict[str, Any]] = []
        for otype, contract in picked.items():
            symbol = contract.get("option_symbol") or ""
            parsed = parse_option_symbol(symbol)
            if not parsed:
                continue
            mid = contract_mid(contract)
            if mid is None or mid <= 0:
                continue
            slot = SLOT_MAP[otype]
            strike_gex = gex_at_strike(conn, ticker, snapshot["ts"], parsed["strike"])
            row = build_quote_row(
                ticker=ticker,
                snapshot=snapshot,
                uw_ticker=uw_ticker,
                slot=slot,
                contract=contract,
                parsed=parsed,
                mid=mid,
                gex_strike=strike_gex,
                quote_ts=quote_ts,
                flow_features=flow,
            )
            upsert_option_quote(conn, row)
            stored.append({"slot": slot, "option_symbol": symbol, "mid_price": mid})

        conn.commit()

    return {
        "ok": True,
        "ticker": ticker,
        "uw_ticker": uw_ticker,
        "expiry": expiry,
        "snapshot_ts": snapshot["ts"],
        "quote_ts": quote_ts,
        "stored": stored,
    }


def learn_from_db(ticker: str, slot: str = "atm_call") -> dict[str, Any]:
    """Bootstrap / incrementally learn option Δmid from stored quotes."""
    if not config.OPTION_LEARN_ENABLED:
        return {"ok": False, "error": "OPTION_LEARN_ENABLED=0"}
    ticker = ticker.upper()
    with get_connection() as conn:
        ensure_extensions(conn)
        quotes = fetch_option_quotes(conn, ticker, slot=slot, limit=1000)
    if len(quotes) < 2:
        return {"ok": False, "error": "Need at least 2 option quotes — run ingest first", "n": len(quotes)}

    boot = ensure_bootstrapped(quotes, ticker, slot)
    learned = maybe_learn_latest(quotes, ticker, slot)
    status = model_status(ticker, slot)
    return {"ok": True, "bootstrap": boot, "learned_pairs": learned, "status": status}


def predict_option_moves(ticker: str) -> dict[str, Any]:
    """Predict next-interval Δmid for latest ATM call/put quotes."""
    ticker = ticker.upper()
    with get_connection() as conn:
        ensure_extensions(conn)
        latest = fetch_latest_option_quotes(conn, ticker)
        snapshot = fetch_latest_snapshot(conn, ticker)
        if not latest:
            return {"ok": False, "error": "No option quotes — ingest from UW first"}

        predictions: list[dict[str, Any]] = []
        for quote in latest:
            slot = quote["slot"]
            delta = predict_delta_mid(quote, ticker, slot)
            mid = float(quote.get("mid_price") or 0)
            pct = (delta / mid) if delta is not None and mid > 0 else None
            conf = min(1.0, model_status(ticker, slot)["n_updates"] / max(config.OPTION_MIN_UPDATES * 2, 1))
            pred_row = {
                "ticker": ticker,
                "snapshot_ts": snapshot["ts"] if snapshot else quote.get("snapshot_ts"),
                "slot": slot,
                "option_symbol": quote.get("option_symbol"),
                "predicted_delta_mid": delta,
                "predicted_pct_change": pct,
                "confidence": conf if delta is not None else 0.0,
                "model": "river_option_delta_mid",
                "features_json": {"current_mid": mid, "spot": quote.get("spot")},
                "created_at": utc_now_iso(),
            }
            if delta is not None and pred_row["snapshot_ts"]:
                insert_option_prediction(conn, pred_row)
            predictions.append(
                {
                    "slot": slot,
                    "option_symbol": quote.get("option_symbol"),
                    "current_mid": mid,
                    "predicted_delta_mid": delta,
                    "predicted_pct_change": pct,
                    "confidence": pred_row["confidence"],
                    "expiry": quote.get("expiry"),
                    "strike": quote.get("strike"),
                    "ready": delta is not None,
                }
            )
        conn.commit()

    return {
        "ok": True,
        "ticker": ticker,
        "snapshot_ts": snapshot["ts"] if snapshot else None,
        "predictions": predictions,
    }


def run_option_cycle(ticker: str | None = None) -> dict[str, Any]:
    """Full cycle: ingest UW → learn → predict."""
    ticker = (ticker or config.DEFAULT_TICKER).upper()
    ingest = ingest_uw_quotes(ticker)
    out: dict[str, Any] = {"ticker": ticker, "ingest": ingest}
    if not ingest.get("ok"):
        return out
    learn_results = {}
    predict_results = {}
    for slot in SLOTS:
        learn_results[slot] = learn_from_db(ticker, slot=slot)
    predict_results = predict_option_moves(ticker)
    out["learn"] = learn_results
    out["predict"] = predict_results
    return out
