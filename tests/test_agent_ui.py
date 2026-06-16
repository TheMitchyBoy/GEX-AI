"""Tests for embedded agent UI routes."""

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_home_page_served():
    r = client.get("/")
    assert r.status_code == 200
    assert "GEX Hub" in r.text
    assert "Live Dashboard" in r.text
    assert "Market Agent" in r.text


def test_agent_redirects_home():
    r = client.get("/agent", follow_redirects=False)
    assert r.status_code == 302
    assert r.headers["location"] == "/"


def test_agent_legacy_page():
    r = client.get("/agent/legacy")
    assert r.status_code == 200
    assert "GEX Market Agent" in r.text


def test_chat_redirect():
    r = client.get("/chat", follow_redirects=False)
    assert r.status_code == 302
    assert r.headers["location"] == "/"


def test_api_root():
    r = client.get("/api")
    assert r.status_code == 200
    assert r.json()["home"] == "/"
    assert r.json()["health"] == "/health"


def test_chat_without_database_url():
    import config
    old = config.DATABASE_URL
    config.DATABASE_URL = ""
    try:
        r = client.post("/llm/chat/SPX", json={
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            "mode": "fast",
        })
        assert r.status_code == 503
        assert "DATABASE_URL" in r.json()["detail"]
    finally:
        config.DATABASE_URL = old
