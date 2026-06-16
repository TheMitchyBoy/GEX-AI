"""Tests for option price learning pipeline."""

from __future__ import annotations

import config
from models.option_features import build_quote_row, feature_dict
from models.option_learn import learn_pair, predict_delta_mid, warm_start
from tests.synthetic_data import generate_synthetic_history


def _sample_quote(mid: float, quote_ts: str, snapshot: dict) -> dict:
    contract = {
        "option_symbol": "SPX260620C05500000",
        "nbbo_bid": str(mid - 0.05),
        "nbbo_ask": str(mid + 0.05),
        "last_price": str(mid),
        "implied_volatility": "0.18",
        "volume": 1000,
        "open_interest": 5000,
    }
    parsed = {"expiry": "2026-06-20", "option_type": "call", "strike": 5500.0}
    row = build_quote_row(
        ticker="SPX",
        snapshot=snapshot,
        uw_ticker="SPX",
        slot="atm_call",
        contract=contract,
        parsed=parsed,
        mid=mid,
        gex_strike=0.02,
        quote_ts=quote_ts,
    )
    return row


def _clean_option_artifacts(ticker: str = "SPX", slot: str = "atm_call") -> None:
    for suffix in (".river.pkl", ".json"):
        p = config.MODELS_DIR / f"{ticker.upper()}_option_{slot}{suffix}"
        if p.exists():
            p.unlink()


def test_feature_dict_shapes():
    snap = generate_synthetic_history(n_snapshots=1)[0]
    q = _sample_quote(12.5, "2026-06-15T15:00:00Z", snap)
    feats = feature_dict(q)
    assert feats["mid_price"] == 12.5
    assert "gex_at_strike" in feats
    assert "implied_volatility" in feats


def test_warm_start_and_predict_option_delta():
    _clean_option_artifacts()
    snap = generate_synthetic_history(n_snapshots=1)[0]
    quotes = [_sample_quote(10.0 + i * 0.15, f"2026-06-15T15:{i:02d}:00Z", snap) for i in range(20)]
    result = warm_start(quotes, "SPX", "atm_call")
    assert result["ok"] is True
    assert result["n_updates"] >= 12
    pred = predict_delta_mid(quotes[-1], "SPX", "atm_call")
    assert pred is not None
    assert isinstance(pred, float)
    _clean_option_artifacts()


def test_learn_pair_incremental():
    _clean_option_artifacts()
    snap = generate_synthetic_history(n_snapshots=1)[0]
    q1 = _sample_quote(10.0, "2026-06-15T15:00:00Z", snap)
    q2 = _sample_quote(10.4, "2026-06-15T15:10:00Z", snap)
    assert learn_pair(q1, q2, "SPX", "atm_call") is True
    _clean_option_artifacts()


def test_predict_after_enough_updates(monkeypatch):
    _clean_option_artifacts()
    monkeypatch.setattr(config, "OPTION_MIN_UPDATES", 5)
    snap = generate_synthetic_history(n_snapshots=1)[0]
    quotes = [_sample_quote(8.0 + i * 0.2, f"2026-06-15T16:{i:02d}:00Z", snap) for i in range(15)]
    warm_start(quotes, "SPX", "atm_call")
    pred = predict_delta_mid(quotes[-1], "SPX", "atm_call")
    assert pred is not None
    assert isinstance(pred, float)
    _clean_option_artifacts()
