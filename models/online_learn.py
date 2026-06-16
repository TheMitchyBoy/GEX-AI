"""Online ΔGEX learner using River (https://github.com/online-ml/river).

Learns incrementally from each new snapshot pair — no full batch retrain.
Model state is persisted per ticker under MODELS_DIR.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import config
from db.features import enrich_snapshot_metrics, safe_float

logger = logging.getLogger(__name__)

FEATURE_KEYS = [
    "total_gex",
    "pos_gex",
    "neg_gex",
    "gex_std",
    "near_term_ratio",
    "gamma_flip",
    "flip_distance_pct",
    "wall_spread",
    "zero_dte_ratio",
    "term_curvature",
    "total_gex_momentum",
    "flip_velocity",
    "gex_concentration",
    "realized_vol",
    "spot_return",
]


def _make_model() -> Any:
    from river import compose, linear_model, optim, preprocessing

    return compose.Pipeline(
        preprocessing.StandardScaler(),
        linear_model.LinearRegression(optimizer=optim.SGD(lr=0.02)),
    )


def model_path(ticker: str) -> Path:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return config.MODELS_DIR / f"{ticker.upper()}_online_gex.river.pkl"


def meta_path(ticker: str) -> Path:
    return model_path(ticker).with_suffix(".json")


def _feature_dict(enriched: dict[str, Any]) -> dict[str, float]:
    return {k: safe_float(enriched.get(k)) for k in FEATURE_KEYS}


def _load(ticker: str) -> tuple[Any | None, dict[str, Any]]:
    path = model_path(ticker)
    meta_file = meta_path(ticker)
    if not path.exists():
        return None, {"n_updates": 0, "last_learned_ts": None}
    try:
        model = pickle.loads(path.read_bytes())
        meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
        meta.setdefault("n_updates", 0)
        return model, meta
    except Exception:
        logger.warning("Failed to load online model for %s", ticker, exc_info=True)
        return None, {"n_updates": 0, "last_learned_ts": None}


def _save(ticker: str, model: Any, meta: dict[str, Any]) -> None:
    path = model_path(ticker)
    path.write_bytes(pickle.dumps(model))
    meta_path(ticker).write_text(
        json.dumps(
            {
                "ticker": ticker.upper(),
                "library": "river",
                "repo": "https://github.com/online-ml/river",
                **meta,
            },
            indent=2,
        )
    )


def learn_pair(before: dict[str, Any], after: dict[str, Any], ticker: str) -> bool:
    """Update online model from one snapshot transition."""
    if not config.ONLINE_LEARNING_ENABLED:
        return False
    before_e = enrich_snapshot_metrics(before.copy())
    after_e = enrich_snapshot_metrics(after.copy())
    target = safe_float(after_e.get("total_gex")) - safe_float(before_e.get("total_gex"))
    x = _feature_dict(before_e)
    model, meta = _load(ticker)
    if model is None:
        model = _make_model()
    model.learn_one(x, target)
    meta["n_updates"] = int(meta.get("n_updates", 0)) + 1
    meta["last_learned_ts"] = after_e.get("ts")
    _save(ticker, model, meta)
    return True


def warm_start(history: list[dict[str, Any]], ticker: str) -> dict[str, Any]:
    """Bootstrap online model from historical snapshot pairs."""
    if len(history) < 3:
        return {"ok": False, "reason": "insufficient history"}
    model = _make_model()
    n = 0
    enriched = [enrich_snapshot_metrics(h.copy()) for h in history]
    for i in range(len(enriched) - 1):
        x = _feature_dict(enriched[i])
        y = safe_float(enriched[i + 1].get("total_gex")) - safe_float(enriched[i].get("total_gex"))
        model.learn_one(x, y)
        n += 1
    meta = {"n_updates": n, "last_learned_ts": enriched[-1].get("ts"), "bootstrapped": True}
    _save(ticker, model, meta)
    return {"ok": True, "n_updates": n, "last_learned_ts": meta["last_learned_ts"]}


def ensure_bootstrapped(history: list[dict[str, Any]], ticker: str) -> bool:
    """Warm-start once if no persisted model exists."""
    if not config.ONLINE_LEARNING_ENABLED or not config.ONLINE_AUTO_BOOTSTRAP:
        return False
    if model_path(ticker).exists():
        return False
    result = warm_start(history, ticker)
    if result.get("ok"):
        logger.info("Online model bootstrapped for %s (%s updates)", ticker, result["n_updates"])
    return bool(result.get("ok"))


def maybe_learn_latest(history: list[dict[str, Any]], ticker: str) -> int:
    """Learn from the latest snapshot pair if not already processed."""
    if not config.ONLINE_LEARNING_ENABLED or len(history) < 2:
        return 0
    before, after = history[-2], history[-1]
    after_ts = after.get("ts")
    _, meta = _load(ticker)
    if meta.get("last_learned_ts") == after_ts:
        return 0
    learn_pair(before, after, ticker)
    return 1


def predict_online_delta(history: list[dict[str, Any]], ticker: str) -> float | None:
    """Predict ΔGEX from the latest snapshot using the online model."""
    if not config.ONLINE_LEARNING_ENABLED:
        return None
    model, meta = _load(ticker)
    if model is None:
        return None
    if int(meta.get("n_updates", 0)) < config.ONLINE_MIN_UPDATES:
        return None
    if not history:
        return None
    current = enrich_snapshot_metrics(history[-1].copy())
    x = _feature_dict(current)
    try:
        pred = model.predict_one(x)
        return float(pred) if pred is not None else None
    except Exception:
        logger.warning("Online prediction failed for %s", ticker, exc_info=True)
        return None


def model_status(ticker: str) -> dict[str, Any]:
    _, meta = _load(ticker)
    return {
        "enabled": config.ONLINE_LEARNING_ENABLED,
        "library": "river",
        "repo": "https://github.com/online-ml/river",
        "model_exists": model_path(ticker).exists(),
        "n_updates": int(meta.get("n_updates", 0)),
        "last_learned_ts": meta.get("last_learned_ts"),
        "min_updates_for_blend": config.ONLINE_MIN_UPDATES,
        "blend_weight": config.ONLINE_BLEND_WEIGHT,
    }
