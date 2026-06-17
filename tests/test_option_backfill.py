"""Tests for option quote backfill."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from integrations.uw_client import mid_at_snapshot_time, pick_atm_symbols, snapshot_ts_to_quote_iso
from services.option_backfill import backfill_option_quotes


def test_snapshot_ts_to_quote_iso():
    assert snapshot_ts_to_quote_iso("2026-06-16_143000") == "2026-06-16T14:30:00Z"


def test_mid_at_snapshot_time():
    bars = [
        {"start_time": "2026-06-16T14:29:00.000000Z", "close": "1.50", "nbbo_bid": "1.4", "nbbo_ask": "1.6"},
        {"start_time": "2026-06-16T14:30:00.000000Z", "close": "1.55", "nbbo_bid": "1.5", "nbbo_ask": "1.6"},
    ]
    mid = mid_at_snapshot_time(bars, "2026-06-16_143000")
    assert mid == 1.55


def test_pick_atm_symbols():
    syms = ["SPXW260616C07500000", "SPXW260616C07510000", "SPXW260616P07510000"]
    atm = pick_atm_symbols(syms, 7510.0)
    assert atm["call"] == "SPXW260616C07510000"


def test_backfill_no_snapshots():
    mock_conn = MagicMock()
    with patch("services.option_backfill.get_connection") as gc:
        gc.return_value.__enter__.return_value = mock_conn
        with patch("services.option_backfill.fetch_snapshots", return_value=[]):
            with patch("services.option_backfill.is_configured", return_value=True):
                result = backfill_option_quotes("SPX", lookback_days=7, train=False)
    assert result["ok"] is False
