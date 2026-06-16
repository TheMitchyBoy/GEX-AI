"""Online option mid-price movement learner (River + DB GEX + UW quotes)."""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import config
from models.option_features import FEATURE_KEYS, feature_dict

logger = logging.getLogger(__name__)

SLOTS = ("atm_call", "atm_put")


def _make_model() -> Any:
    from river import compose, linear_model, optim, preprocessing

    return compose.Pipeline(
        preprocessing.StandardScaler(),
        linear_model.LinearRegression(optimizer=optim.SGD(lr=0.015)),
    )


def model_path(ticker: str, slot: str) -> Path:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return config.MODELS_DIR / f"{ticker.upper()}_option_{slot}.river.pkl"


def meta_path(ticker: str, slot: str) -> Path:
    return model_path(ticker, slot).with_suffix(".json")


def _load(ticker: str, slot: str) -> tuple[Any | None, dict[str, Any]]:
    path = model_path(ticker, slot)
    meta_file = meta_path(ticker, slot)
    if not path.exists():
        return None, {"n_updates": 0, "last_learned_ts": None, "slot": slot}
    try:
        model = pickle.loads(path.read_bytes())
        meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
        meta.setdefault("n_updates", 0)
        meta.setdefault("slot", slot)
        return model, meta
    except Exception:
        logger.warning("Failed to load option model %s %s", ticker, slot, exc_info=True)
        return None, {"n_updates": 0, "last_learned_ts": None, "slot": slot}


def _save(ticker: str, slot: str, model: Any, meta: dict[str, Any]) -> None:
    model_path(ticker, slot).write_bytes(pickle.dumps(model))
    meta_path(ticker, slot).write_text(
        json.dumps(
            {
                "ticker": ticker.upper(),
                "slot": slot,
                "target": "delta_mid",
                "library": "river",
                **meta,
            },
            indent=2,
        )
    )


def learn_pair(before: dict[str, Any], after: dict[str, Any], ticker: str, slot: str) -> bool:
    if not config.OPTION_LEARN_ENABLED:
        return False
    before_mid = float(before.get("mid_price") or 0)
    after_mid = float(after.get("mid_price") or 0)
    if before_mid <= 0 or after_mid <= 0:
        return False
    target = after_mid - before_mid
    x = feature_dict(before)
    model, meta = _load(ticker, slot)
    if model is None:
        model = _make_model()
    model.learn_one(x, target)
    meta["n_updates"] = int(meta.get("n_updates", 0)) + 1
    meta["last_learned_ts"] = after.get("quote_ts")
    _save(ticker, slot, model, meta)
    return True


def warm_start(quotes: list[dict[str, Any]], ticker: str, slot: str) -> dict[str, Any]:
    if len(quotes) < config.OPTION_MIN_TRAIN_ROWS:
        return {"ok": False, "reason": "insufficient quotes", "n": len(quotes)}
    model = _make_model()
    n = 0
    for i in range(len(quotes) - 1):
        before, after = quotes[i], quotes[i + 1]
        before_mid = float(before.get("mid_price") or 0)
        after_mid = float(after.get("mid_price") or 0)
        if before_mid <= 0 or after_mid <= 0:
            continue
        model.learn_one(feature_dict(before), after_mid - before_mid)
        n += 1
    if n < 3:
        return {"ok": False, "reason": "insufficient valid pairs", "n": n}
    meta = {"n_updates": n, "last_learned_ts": quotes[-1].get("quote_ts"), "slot": slot}
    _save(ticker, slot, model, meta)
    return {"ok": True, "n_updates": n, "slot": slot}


def predict_delta_mid(quote: dict[str, Any], ticker: str, slot: str) -> float | None:
    model, meta = _load(ticker, slot)
    if model is None or int(meta.get("n_updates", 0)) < config.OPTION_MIN_UPDATES:
        return None
    try:
        return float(model.predict_one(feature_dict(quote)))
    except Exception:
        logger.warning("Option predict failed %s %s", ticker, slot, exc_info=True)
        return None


def maybe_learn_latest(quotes: list[dict[str, Any]], ticker: str, slot: str) -> int:
    if len(quotes) < 2:
        return 0
    before, after = quotes[-2], quotes[-1]
    _, meta = _load(ticker, slot)
    if meta.get("last_learned_ts") == after.get("quote_ts"):
        return 0
    return 1 if learn_pair(before, after, ticker, slot) else 0


def ensure_bootstrapped(quotes: list[dict[str, Any]], ticker: str, slot: str) -> dict[str, Any]:
    if not config.OPTION_LEARN_ENABLED:
        return {"ok": False, "reason": "disabled"}
    _, meta = _load(ticker, slot)
    if int(meta.get("n_updates", 0)) >= config.OPTION_MIN_UPDATES:
        return {"ok": True, "bootstrapped": False, "n_updates": meta["n_updates"]}
    return warm_start(quotes, ticker, slot)


def model_status(ticker: str, slot: str) -> dict[str, Any]:
    _, meta = _load(ticker, slot)
    return {
        "ticker": ticker.upper(),
        "slot": slot,
        "enabled": config.OPTION_LEARN_ENABLED,
        "n_updates": int(meta.get("n_updates", 0)),
        "last_learned_ts": meta.get("last_learned_ts"),
        "ready": int(meta.get("n_updates", 0)) >= config.OPTION_MIN_UPDATES,
        "feature_keys": FEATURE_KEYS,
    }
