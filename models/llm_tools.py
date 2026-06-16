"""Agent tool definitions and execution against existing forecast APIs."""

from __future__ import annotations

import json
from typing import Any, Callable

import config
from db.features import enrich_snapshot_metrics, safe_float
from models.backtest import run_backtest
from models.gboost import predict_gboost_delta
from models.llm_predict import generate_llm_forecast
from models.multi_horizon import predict_multi_horizon
from models.predict import predict_next_snapshot, similar_setups

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_forecast",
            "description": "Get KNN quant forecast and multi-horizon projections for the ticker.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_similar_setups",
            "description": "Find nearest historical snapshot analogs and their next-snapshot outcomes.",
            "parameters": {
                "type": "object",
                "properties": {"top_n": {"type": "integer", "default": 5}},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_strikes_near_atm",
            "description": "Get strike-level GEX profile near ATM from latest snapshot.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_backtest_summary",
            "description": "Walk-forward backtest metrics for KNN forecasts over recent history.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_knn_llm",
            "description": "Compare KNN baseline forecast vs LLM structured forecast.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
]


class AgentToolRunner:
    def __init__(self, history: list[dict[str, Any]], *, lookback_days: int | None = None):
        self.history = history
        self.lookback_days = lookback_days or config.LOOKBACK_DAYS
        self.ticker = history[-1].get("ticker", config.DEFAULT_TICKER) if history else config.DEFAULT_TICKER

    def execute(self, name: str, arguments: str | dict | None) -> str:
        args: dict[str, Any] = {}
        if isinstance(arguments, str) and arguments.strip():
            try:
                args = json.loads(arguments)
            except json.JSONDecodeError:
                args = {}
        elif isinstance(arguments, dict):
            args = arguments

        handlers: dict[str, Callable[..., Any]] = {
            "get_forecast": self._get_forecast,
            "get_similar_setups": self._get_similar_setups,
            "get_strikes_near_atm": self._get_strikes_near_atm,
            "get_backtest_summary": self._get_backtest_summary,
            "compare_knn_llm": self._compare_knn_llm,
        }
        fn = handlers.get(name)
        if not fn:
            return json.dumps({"error": f"unknown tool {name}"})
        try:
            return json.dumps(fn(**args), default=str)
        except TypeError:
            return json.dumps(fn(), default=str)

    def _get_forecast(self, **_kwargs) -> dict[str, Any]:
        knn = predict_next_snapshot(self.history, lookback_days=self.lookback_days)
        gboost = predict_gboost_delta(self.history, self.ticker)
        horizons = predict_multi_horizon(self.history, lookback_days=self.lookback_days)
        return {
            "knn": knn,
            "gboost_delta_gex_bn": gboost,
            "horizons": {str(k): v for k, v in horizons.items()},
        }

    def _get_similar_setups(self, top_n: int = 5, **_kwargs) -> dict[str, Any]:
        return {"similar_setups": similar_setups(self.history, top_n=top_n, lookback_days=self.lookback_days)}

    def _get_strikes_near_atm(self, **_kwargs) -> dict[str, Any]:
        cur = enrich_snapshot_metrics(self.history[-1].copy())
        strike = cur.get("strike")
        spot = safe_float(cur.get("spot"))
        rows = []
        if strike is not None and not getattr(strike, "empty", True):
            for k, v in strike.items():
                sk = float(k)
                if spot > 0 and abs(sk - spot) / spot <= 0.03:
                    rows.append({"strike": sk, "gex_bn_per_pct": float(v)})
            rows.sort(key=lambda r: r["strike"])
        return {"spot": spot, "strikes_near_atm": rows[:25]}

    def _get_backtest_summary(self, **_kwargs) -> dict[str, Any]:
        report = run_backtest(self.history, lookback_days=min(30, self.lookback_days))
        return report.to_dict()

    def _compare_knn_llm(self, **_kwargs) -> dict[str, Any]:
        knn = predict_next_snapshot(self.history, lookback_days=self.lookback_days)
        llm = generate_llm_forecast(self.history, lookback_days=self.lookback_days, persist=False)
        agree = knn and llm and knn.get("predicted_regime") == llm.get("predicted_regime")
        return {
            "knn": knn,
            "llm": llm,
            "regime_agreement": agree,
            "delta_gex_diff": abs(
                safe_float(knn.get("predicted_delta_gex")) - safe_float(llm.get("predicted_delta_gex_bn"))
            )
            if knn and llm
            else None,
        }
