"""Tests for option schema migration."""

from unittest.mock import MagicMock, patch

from db.option_queries import _option_schema_statements, ensure_option_schema


def test_option_schema_statements_skip_comments():
    stmts = _option_schema_statements()
    assert len(stmts) >= 2
    assert all("option_quotes" in s or "option_price_predictions" in s or "idx_" in s for s in stmts)
    assert not any(s.strip().startswith("--") for s in stmts)


def test_ensure_option_schema_creates_when_missing():
    conn = MagicMock()
    exists_results = [None, ("option_quotes",)]

    def execute_side_effect(sql, *args, **kwargs):
        result = MagicMock()
        if "to_regclass" in str(sql):
            val = exists_results.pop(0)
            result.fetchone.return_value = (val,) if val else (None,)
        return result

    conn.execute.side_effect = execute_side_effect

    with patch("db.option_queries._option_tables_exist", side_effect=[False, True]):
        ensure_option_schema(conn)

    assert conn.commit.called
