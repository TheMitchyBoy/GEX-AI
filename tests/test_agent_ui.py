"""Tests for embedded agent UI routes."""

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_agent_page_served():
    r = client.get("/agent")
    assert r.status_code == 200
    assert "GEX Market Agent" in r.text


def test_chat_redirect():
    r = client.get("/chat", follow_redirects=False)
    assert r.status_code == 302
    assert r.headers["location"] == "/agent"


def test_root_links_agent():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["agent_ui"] == "/agent"


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
