"""Tests for Unusual Whales client helpers."""

from integrations.uw_client import (
    contract_mid,
    nearest_expiry,
    parse_option_symbol,
    pick_atm_contracts,
)


def test_parse_option_symbol():
    parsed = parse_option_symbol("AAPL240202C00185000")
    assert parsed is not None
    assert parsed["expiry"] == "2024-02-02"
    assert parsed["option_type"] == "call"
    assert parsed["strike"] == 185.0


def test_contract_mid_from_nbbo():
    mid = contract_mid({"nbbo_bid": "1.00", "nbbo_ask": "1.20"})
    assert mid == 1.1


def test_pick_atm_contracts():
    spot = 5500.0
    contracts = [
        {"option_symbol": "SPX240315C05400000", "nbbo_bid": "10", "nbbo_ask": "12"},
        {"option_symbol": "SPX240315C05500000", "nbbo_bid": "8", "nbbo_ask": "9"},
        {"option_symbol": "SPX240315P05500000", "nbbo_bid": "7", "nbbo_ask": "8"},
    ]
    picked = pick_atm_contracts(contracts, spot)
    assert "call" in picked
    assert "put" in picked
    assert parse_option_symbol(picked["call"]["option_symbol"])["strike"] == 5500.0


def test_nearest_expiry():
    exp = {"2026-06-10": 1.0, "2026-06-17": 2.0, "2026-06-20": 3.0}
    assert nearest_expiry(exp, "2026-06-15") == "2026-06-17"


def test_nearest_expiry_datetime_keys():
    exp = {"2026-06-16 00:00:00": 1.0, "2026-06-20 00:00:00": 2.0}
    assert nearest_expiry(exp, "2026-06-16") == "2026-06-16"
