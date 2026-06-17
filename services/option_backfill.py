"""Backfill option_quotes from GEX snapshot history + UW intraday/historic API."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Any

import config
from db.connection import get_connection
from db.option_queries import ensure_option_schema, fetch_option_quotes, gex_at_strike, upsert_option_quote
from db.queries import fetch_snapshots
from integrations.uw_client import (
    UnusualWhalesClient,
    contract_mid,
    historic_mid_on_date,
    is_configured,
    mid_at_snapshot_time,
    nearest_expiry,
    parse_option_symbol,
    pick_atm_symbols,
    snapshot_ts_to_quote_iso,
    synthetic_atm_mid,
    uw_ticker_for,
)
from models.option_features import build_quote_row
from models.option_learn import SLOTS, warm_start

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


def _filter_expiry_symbols(symbols: list[str], expiry: str) -> list[str]:
    exp = expiry.replace("-", "")[2:]  # YYMMDD from YYYY-MM-DD
    return [s for s in symbols if exp in s.upper()]


def _resolve_mid(
    client: UnusualWhalesClient,
    symbol: str,
    snapshot: dict[str, Any],
    intraday_cache: dict[str, list[dict[str, Any]]],
    historic_cache: dict[str, list[dict[str, Any]]],
    option_type: str,
) -> tuple[float | None, str]:
    ts = snapshot["ts"]
    mdate = snapshot.get("market_date") or ts[:10]

    if symbol not in intraday_cache:
        try:
            intraday_cache[symbol] = client.option_intraday(symbol, market_date=mdate)
            time.sleep(config.UW_BACKFILL_SLEEP_SEC)
        except Exception:
            logger.debug("Intraday fetch failed for %s on %s", symbol, mdate, exc_info=True)
            intraday_cache[symbol] = []

    mid = mid_at_snapshot_time(intraday_cache[symbol], ts)
    if mid and mid > 0:
        return mid, "uw_intraday"

    if symbol not in historic_cache:
        try:
            historic_cache[symbol] = client.option_historic(symbol, limit=120)
            time.sleep(config.UW_BACKFILL_SLEEP_SEC)
        except Exception:
            historic_cache[symbol] = []

    mid = historic_mid_on_date(historic_cache[symbol], mdate)
    if mid and mid > 0:
        return mid, "uw_historic"

    if config.OPTION_BACKFILL_GEX_PROXY:
        spot = float(snapshot.get("spot") or 0)
        parsed = parse_option_symbol(symbol)
        dte = 7
        if parsed and parsed.get("expiry"):
            from datetime import datetime

            try:
                exp = datetime.strptime(parsed["expiry"], "%Y-%m-%d").date()
                ref = datetime.strptime(mdate[:10], "%Y-%m-%d").date()
                dte = max((exp - ref).days, 0)
            except ValueError:
                pass
        return synthetic_atm_mid(spot, option_type, dte=dte), "gex_proxy"

    return None, "missing"


def backfill_option_quotes(
    ticker: str,
    *,
    lookback_days: int | None = None,
    step: int | None = None,
    train: bool = True,
    client: UnusualWhalesClient | None = None,
) -> dict[str, Any]:
    """Align historical GEX snapshots with UW option mids; optionally warm-start models."""
    if not is_configured():
        return {"ok": False, "error": "UW_API_KEY is not set"}

    ticker = ticker.upper()
    lookback_days = lookback_days if lookback_days is not None else config.OPTION_BACKFILL_LOOKBACK_DAYS
    step = max(1, step if step is not None else config.OPTION_BACKFILL_STEP)
    uw_ticker = uw_ticker_for(ticker)
    client = client or UnusualWhalesClient()

    with get_connection() as conn:
        ensure_option_schema(conn)
        snapshots = fetch_snapshots(conn, ticker, lookback_days=lookback_days)
        if not snapshots:
            return {"ok": False, "error": f"No snapshots in last {lookback_days} days"}

        if step > 1:
            snapshots = snapshots[::step]

        by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for snap in snapshots:
            mdate = snap.get("market_date") or snap["ts"][:10]
            by_date[mdate].append(snap)

        stored = 0
        skipped = 0
        sources: dict[str, int] = defaultdict(int)
        days_ok = 0
        days_failed = 0
        intraday_cache: dict[str, list[dict[str, Any]]] = {}
        historic_cache: dict[str, list[dict[str, Any]]] = {}
        chain_cache: dict[str, list[str]] = {}

        for mdate in sorted(by_date.keys()):
            day_snaps = by_date[mdate]
            try:
                if mdate not in chain_cache:
                    chain_cache[mdate] = client.option_chains(uw_ticker, market_date=mdate)
                    time.sleep(config.UW_BACKFILL_SLEEP_SEC)
                symbols = chain_cache[mdate]
                if not symbols:
                    days_failed += 1
                    skipped += len(day_snaps) * 2
                    continue
                days_ok += 1
            except Exception as exc:
                logger.warning("Chain fetch failed %s %s: %s", uw_ticker, mdate, exc)
                days_failed += 1
                skipped += len(day_snaps) * 2
                continue

            for snap in day_snaps:
                spot = float(snap.get("spot") or 0)
                if spot <= 0:
                    skipped += 2
                    continue
                expiry = nearest_expiry(snap.get("expiration_json"), mdate)
                day_symbols = _filter_expiry_symbols(symbols, expiry) if expiry else symbols
                if not day_symbols:
                    day_symbols = symbols
                atm = pick_atm_symbols(day_symbols, spot)
                flow = _flow_from_snapshot(snap)
                quote_ts = snapshot_ts_to_quote_iso(snap["ts"])

                for otype, slot_key in (("call", "atm_call"), ("put", "atm_put")):
                    symbol = atm.get(otype)
                    if not symbol:
                        skipped += 1
                        continue
                    parsed = parse_option_symbol(symbol)
                    if not parsed:
                        skipped += 1
                        continue
                    mid, source = _resolve_mid(
                        client, symbol, snap, intraday_cache, historic_cache, otype
                    )
                    if mid is None or mid <= 0:
                        skipped += 1
                        continue
                    contract = {
                        "option_symbol": symbol,
                        "nbbo_bid": str(mid * 0.98),
                        "nbbo_ask": str(mid * 1.02),
                        "last_price": str(mid),
                        "implied_volatility": "0",
                        "volume": 0,
                        "open_interest": 0,
                    }
                    strike_gex = gex_at_strike(conn, ticker, snap["ts"], parsed["strike"])
                    row = build_quote_row(
                        ticker=ticker,
                        snapshot=snap,
                        uw_ticker=uw_ticker,
                        slot=slot_key,
                        contract=contract,
                        parsed=parsed,
                        mid=mid,
                        gex_strike=strike_gex,
                        quote_ts=quote_ts,
                        flow_features={**flow, "backfill_source": source},
                    )
                    upsert_option_quote(conn, row)
                    stored += 1
                    sources[source] += 1

            conn.commit()

    train_results: dict[str, Any] = {}
    if train and stored > 0:
        for slot in SLOTS:
            with get_connection() as conn:
                quotes = fetch_option_quotes(conn, ticker, slot=slot, limit=20000)
            train_results[slot] = warm_start(quotes, ticker, slot) if len(quotes) >= 2 else {"ok": False, "n": len(quotes)}

    return {
        "ok": True,
        "ticker": ticker,
        "lookback_days": lookback_days,
        "snapshots_processed": len(snapshots),
        "trading_days": len(by_date),
        "days_with_chains": days_ok,
        "days_failed": days_failed,
        "quotes_stored": stored,
        "quotes_skipped": skipped,
        "sources": dict(sources),
        "train": train_results,
    }
