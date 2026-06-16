"""Tests for option API routes."""

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_options_status():
    r = client.get("/options/status")
    assert r.status_code == 200
    body = r.json()
    assert "uw_configured" in body
    assert "learn_enabled" in body


def test_ingest_without_uw_key():
    import config
    old = config.UW_API_KEY
    config.UW_API_KEY = ""
    try:
        r = client.post("/options/ingest/SPX")
        assert r.status_code == 503
        assert "UW_API_KEY" in r.json()["detail"]
    finally:
        config.UW_API_KEY = old


def test_forecast_without_quotes():
    import config
    old_url = config.DATABASE_URL
    if not old_url:
        return
    r = client.get("/options/forecast/SPX")
    # May be 422 if no quotes, or 200 if quotes exist in DB
    assert r.status_code in (200, 422, 503)
