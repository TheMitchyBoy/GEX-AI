"""Backfill option_quotes from GEX snapshot history + UW intraday/historic API."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from datetime import datetime
from typing import Any

import config
from db.connection import get_connection
from db.option_queries import (
    ensure_option_schema,
    fetch_existing_quote_ts,
    fetch_option_quotes,
    gex_at_strike,
    upsert_option_quote,
)
from db.queries import fetch_snapshots
from integrations.uw_client import (
    UnusualWhalesClient,
    atm_strike_for_spot,
    build_synthetic_occ_symbol,
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

SLOT_TYPES = (("call", "atm_call"), ("put", "atm_put"))


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
    exp = expiry.replace("-", "")[2:]
    return [s for s in symbols if exp in s.upper()]


def _dte(expiry: str, market_date: str) -> int:
    try:
        exp = datetime.strptime(expiry[:10], "%Y-%m-%d").date()
        ref = datetime.strptime(market_date[:10], "%Y-%m-%d").date()
        return max((exp - ref).days, 0)
    except ValueError:
        return 7


def _store_quote(
    conn,
    *,
    ticker: str,
    uw_ticker: str,
    snapshot: dict[str, Any],
    slot: str,
    symbol: str,
    parsed: dict[str, Any],
    mid: float,
    source: str,
    quote_ts: str,
    flow: dict[str, Any],
) -> None:
    contract = {
        "option_symbol": symbol,
        "nbbo_bid": str(mid * 0.98),
        "nbbo_ask": str(mid * 1.02),
        "last_price": str(mid),
        "implied_volatility": "0",
        "volume": 0,
        "open_interest": 0,
    }
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
        flow_features={**flow, "backfill_source": source},
    )
    upsert_option_quote(conn, row)


def _resolve_mid_uw(
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
        dte = _dte(parsed.get("expiry", ""), mdate) if parsed else 7
        return synthetic_atm_mid(spot, option_type, dte=dte), "gex_proxy"

    return None, "missing"


def _process_day_gex_only(
    conn,
    *,
    ticker: str,
    uw_ticker: str,
    day_snaps: list[dict[str, Any]],
    existing_ts: set[str],
    symbol_root: str,
) -> tuple[int, int, dict[str, int]]:
    stored, skipped = 0, 0
    sources: dict[str, int] = defaultdict(int)
    for snap in day_snaps:
        spot = float(snap.get("spot") or 0)
        if spot <= 0:
            skipped += 2
            continue
        mdate = snap.get("market_date") or snap["ts"][:10]
        expiry = nearest_expiry(snap.get("expiration_json"), mdate) or mdate
        strike = atm_strike_for_spot(spot)
        flow = _flow_from_snapshot(snap)
        quote_ts = snapshot_ts_to_quote_iso(snap["ts"])
        if quote_ts in existing_ts:
            skipped += 2
            continue
        dte = _dte(expiry, mdate)
        for otype, slot in SLOT_TYPES:
            mid = synthetic_atm_mid(spot, otype, dte=dte)
            symbol = build_synthetic_occ_symbol(symbol_root, expiry, strike, otype)
            parsed = parse_option_symbol(symbol) or {
                "expiry": expiry,
                "strike": strike,
                "option_type": otype,
            }
            _store_quote(
                conn,
                ticker=ticker,
                uw_ticker=uw_ticker,
                snapshot=snap,
                slot=slot,
                symbol=symbol,
                parsed=parsed,
                mid=mid,
                source="gex_only",
                quote_ts=quote_ts,
                flow=flow,
            )
            stored += 1
            sources["gex_only"] += 1
        existing_ts.add(quote_ts)
    return stored, skipped, dict(sources)


def backfill_option_quotes(
    ticker: str,
    *,
    lookback_days: int | None = None,
    step: int | None = None,
    train: bool = True,
    gex_only: bool = False,
    resume: bool = True,
    client: UnusualWhalesClient | None = None,
) -> dict[str, Any]:
    """Align historical GEX snapshots with option mids; optionally warm-start models."""
    if not gex_only and not is_configured():
        return {"ok": False, "error": "UW_API_KEY is not set (or use --gex-only)"}

    ticker = ticker.upper()
    lookback_days = lookback_days if lookback_days is not None else config.OPTION_BACKFILL_LOOKBACK_DAYS
    step = max(1, step if step is not None else config.OPTION_BACKFILL_STEP)
    uw_ticker = uw_ticker_for(ticker)
    symbol_root = "SPXW" if ticker == "SPX" else uw_ticker
    client = client or UnusualWhalesClient()

    with get_connection() as conn:
        ensure_option_schema(conn)
        snapshots = fetch_snapshots(conn, ticker, lookback_days=lookback_days)
        if not snapshots:
            return {"ok": False, "error": f"No snapshots in last {lookback_days} days"}

        if step > 1:
            snapshots = snapshots[::step]

        existing_ts = fetch_existing_quote_ts(conn, ticker) if resume else set()

        by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for snap in snapshots:
            mdate = snap.get("market_date") or snap["ts"][:10]
            by_date[mdate].append(snap)

        stored = 0
        skipped = 0
        resumed = 0
        sources: dict[str, int] = defaultdict(int)
        days_ok = 0
        days_gex_fallback = 0
        days_failed = 0
        uw_days_used = 0
        intraday_cache: dict[str, list[dict[str, Any]]] = {}
        historic_cache: dict[str, list[dict[str, Any]]] = {}
        chain_cache: dict[str, list[str]] = {}
        max_uw_days = config.OPTION_BACKFILL_MAX_UW_DAYS

        for mdate in sorted(by_date.keys()):
            day_snaps = by_date[mdate]

            if gex_only:
                s, sk, src = _process_day_gex_only(
                    conn, ticker=ticker, uw_ticker=uw_ticker, day_snaps=day_snaps,
                    existing_ts=existing_ts, symbol_root=symbol_root,
                )
                stored += s
                skipped += sk
                for k, v in src.items():
                    sources[k] += v
                days_gex_fallback += 1
                conn.commit()
                continue

            if max_uw_days > 0 and uw_days_used >= max_uw_days:
                s, sk, src = _process_day_gex_only(
                    conn, ticker=ticker, uw_ticker=uw_ticker, day_snaps=day_snaps,
                    existing_ts=existing_ts, symbol_root=symbol_root,
                )
                stored += s
                skipped += sk
                for k, v in src.items():
                    sources[k] += v
                days_gex_fallback += 1
                conn.commit()
                continue

            symbols: list[str] = []
            uw_failed = False
            try:
                if mdate not in chain_cache:
                    chain_cache[mdate] = client.option_chains(uw_ticker, market_date=mdate)
                    time.sleep(config.UW_BACKFILL_SLEEP_SEC)
                    uw_days_used += 1
                symbols = chain_cache[mdate]
                if not symbols:
                    uw_failed = True
                else:
                    days_ok += 1
            except Exception as exc:
                logger.warning("Chain fetch failed %s %s: %s", uw_ticker, mdate, exc)
                uw_failed = True

            if uw_failed:
                s, sk, src = _process_day_gex_only(
                    conn, ticker=ticker, uw_ticker=uw_ticker, day_snaps=day_snaps,
                    existing_ts=existing_ts, symbol_root=symbol_root,
                )
                stored += s
                skipped += sk
                for k, v in src.items():
                    sources[k] += v
                days_gex_fallback += 1
                conn.commit()
                continue

            for snap in day_snaps:
                spot = float(snap.get("spot") or 0)
                if spot <= 0:
                    skipped += 2
                    continue
                quote_ts = snapshot_ts_to_quote_iso(snap["ts"])
                if quote_ts in existing_ts:
                    resumed += 2
                    continue

                expiry = nearest_expiry(snap.get("expiration_json"), mdate)
                day_symbols = _filter_expiry_symbols(symbols, expiry) if expiry else symbols
                if not day_symbols:
                    day_symbols = symbols
                atm = pick_atm_symbols(day_symbols, spot)
                flow = _flow_from_snapshot(snap)

                for otype, slot_key in SLOT_TYPES:
                    symbol = atm.get(otype)
                    if not symbol:
                        skipped += 1
                        continue
                    parsed = parse_option_symbol(symbol)
                    if not parsed:
                        skipped += 1
                        continue
                    mid, source = _resolve_mid_uw(
                        client, symbol, snap, intraday_cache, historic_cache, otype
                    )
                    if mid is None or mid <= 0:
                        skipped += 1
                        continue
                    _store_quote(
                        conn,
                        ticker=ticker,
                        uw_ticker=uw_ticker,
                        snapshot=snap,
                        slot=slot_key,
                        symbol=symbol,
                        parsed=parsed,
                        mid=mid,
                        source=source,
                        quote_ts=quote_ts,
                        flow=flow,
                    )
                    stored += 1
                    sources[source] += 1
                existing_ts.add(quote_ts)

            conn.commit()

    train_results: dict[str, Any] = {}
    if train:
        for slot in SLOTS:
            with get_connection() as conn:
                quotes = fetch_option_quotes(conn, ticker, slot=slot, limit=20000)
            train_results[slot] = (
                warm_start(quotes, ticker, slot) if len(quotes) >= 2 else {"ok": False, "n": len(quotes)}
            )

    return {
        "ok": True,
        "ticker": ticker,
        "lookback_days": lookback_days,
        "gex_only": gex_only,
        "snapshots_processed": len(snapshots),
        "trading_days": len(by_date),
        "days_with_uw_chains": days_ok,
        "days_gex_fallback": days_gex_fallback,
        "days_failed": days_failed,
        "uw_api_days_called": uw_days_used,
        "quotes_stored": stored,
        "quotes_skipped": skipped,
        "quotes_resumed_skipped": resumed,
        "sources": dict(sources),
        "train": train_results,
    }
