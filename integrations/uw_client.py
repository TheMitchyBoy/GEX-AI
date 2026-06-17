"""Unusual Whales REST API client (https://api.unusualwhales.com)."""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any

import httpx

import config

logger = logging.getLogger(__name__)

_OCC_RE = re.compile(r"^([A-Z]+)(\d{6})([CP])(\d{8})$")


def is_configured() -> bool:
    return bool(config.UW_API_KEY)


def uw_ticker_for(gex_ticker: str) -> str:
    return config.UW_TICKER_MAP.get(gex_ticker.upper(), gex_ticker.upper())


def parse_option_symbol(symbol: str) -> dict[str, Any] | None:
    """Parse OCC option symbol into root, expiry, type, strike."""
    m = _OCC_RE.match(symbol.upper())
    if not m:
        return None
    root, yymmdd, cp, strike_raw = m.groups()
    yy, mm, dd = int(yymmdd[:2]), int(yymmdd[2:4]), int(yymmdd[4:6])
    year = 2000 + yy
    expiry = f"{year:04d}-{mm:02d}-{dd:02d}"
    strike = int(strike_raw) / 1000.0
    return {
        "root": root,
        "expiry": expiry,
        "option_type": "call" if cp == "C" else "put",
        "strike": strike,
    }


def contract_mid(row: dict[str, Any]) -> float | None:
    bid = _safe_float(row.get("nbbo_bid"))
    ask = _safe_float(row.get("nbbo_ask"))
    if bid > 0 and ask > 0:
        return (bid + ask) / 2.0
    last = _safe_float(row.get("last_price") or row.get("close"))
    if last > 0:
        return last
    avg = _safe_float(row.get("avg_price"))
    return avg if avg > 0 else None


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


class UnusualWhalesClient:
    def __init__(self, api_key: str | None = None, base_url: str | None = None, timeout: float | None = None):
        self.api_key = api_key or config.UW_API_KEY
        self.base_url = (base_url or config.UW_BASE_URL).rstrip("/")
        self.timeout = timeout or config.UW_TIMEOUT_SEC

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("UW_API_KEY is not set")
        url = f"{self.base_url}{path}"
        with httpx.Client(timeout=self.timeout) as client:
            r = client.get(url, headers=self._headers(), params=params)
            r.raise_for_status()
            return r.json()

    def option_contracts(
        self,
        ticker: str,
        *,
        expiry: str | None = None,
        option_type: str | None = None,
        limit: int = 500,
        exclude_zero_oi_chains: bool = True,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"limit": min(max(limit, 1), 500)}
        if expiry:
            params["expiry"] = expiry
        if option_type:
            params["option_type"] = option_type
        if exclude_zero_oi_chains:
            params["exclude_zero_oi_chains"] = "true"
        payload = self._get(f"/api/stock/{ticker.upper()}/option-contracts", params)
        return list(payload.get("data") or [])

    def atm_chains(self, ticker: str, expirations: list[str]) -> list[dict[str, Any]]:
        if not expirations:
            return []
        params: list[tuple[str, str]] = [("expirations[]", d) for d in expirations[:5]]
        with httpx.Client(timeout=self.timeout) as client:
            r = client.get(
                f"{self.base_url}/api/stock/{ticker.upper()}/atm-chains",
                headers=self._headers(),
                params=params,
            )
            r.raise_for_status()
            return list(r.json().get("data") or [])

    def stock_price_levels(self, ticker: str, market_date: str | None = None) -> list[dict[str, Any]]:
        params: dict[str, str] = {}
        if market_date:
            params["date"] = market_date
        payload = self._get(f"/api/stock/{ticker.upper()}/option/stock-price-levels", params)
        return list(payload.get("data") or [])


def pick_atm_contracts(
    contracts: list[dict[str, Any]],
    spot: float,
    *,
    option_types: tuple[str, ...] = ("call", "put"),
) -> dict[str, dict[str, Any]]:
    """Pick nearest-to-spot call and put from UW contract rows."""
    by_type: dict[str, list[tuple[float, dict[str, Any]]]] = {t: [] for t in option_types}
    for row in contracts:
        symbol = row.get("option_symbol") or ""
        parsed = parse_option_symbol(symbol)
        if not parsed:
            continue
        otype = parsed["option_type"]
        if otype not in by_type:
            continue
        dist = abs(parsed["strike"] - spot)
        by_type[otype].append((dist, row))
    out: dict[str, dict[str, Any]] = {}
    for otype in option_types:
        candidates = sorted(by_type.get(otype, []), key=lambda x: x[0])
        if candidates:
            out[otype] = candidates[0][1]
    return out


def _normalize_date_key(key: Any) -> str:
    """Normalize expiration_json keys to YYYY-MM-DD for UW API."""
    s = str(key).strip()
    if " " in s:
        s = s.split(" ", 1)[0]
    if "T" in s:
        s = s.split("T", 1)[0]
    return s[:10]


def nearest_expiry(expiration_json: dict[str, Any] | None, market_date: str | None = None) -> str | None:
    if not expiration_json:
        return None
    dates = sorted({_normalize_date_key(k) for k in expiration_json.keys()})
    if not dates:
        return None
    ref = _normalize_date_key(market_date or datetime.utcnow().strftime("%Y-%m-%d"))
    future = [d for d in dates if d >= ref]
    return future[0] if future else dates[-1]
