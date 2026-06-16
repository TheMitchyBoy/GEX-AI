"""Event-aware prompt snippets injected by market conditions."""

from __future__ import annotations

from typing import Any

from db.features import safe_float


PLAYBOOK_0DTE = """## Active playbook: 0DTE-heavy session
- Pin risk into close is elevated; charm decay accelerates final hour.
- Dealers may defend strikes with largest positive gamma near spot.
- Reduce trend-follow conviction; widen confidence intervals."""

PLAYBOOK_FOMC = """## Active playbook: FOMC / high event risk
- Cite event_risk_score and is_fomc_week in your answer.
- Widen uncertainty; avoid high-confidence directional calls pre-event.
- Flip and walls may reposition quickly — note what would invalidate the view."""

PLAYBOOK_OPEX = """## Active playbook: OPEX week
- Pin toward strikes with large open interest / gamma magnets.
- Wall levels are more binding; mean-reversion bias increases near spot."""

PLAYBOOK_NEAR_FLIP = """## Active playbook: spot near gamma flip
- flip_distance_pct is tight — small spot moves can change dealer hedging regime.
- Emphasize flip as magnet/resistance; alternate scenario = clean break through flip."""


def event_prompt_snippets(summary: dict[str, Any]) -> list[str]:
    snippets: list[str] = []
    if safe_float(summary.get("zero_dte_ratio")) >= 0.35:
        snippets.append(PLAYBOOK_0DTE)
    if summary.get("is_fomc_week") or safe_float(summary.get("event_risk_score")) >= 0.5:
        snippets.append(PLAYBOOK_FOMC)
    if summary.get("is_opex_week"):
        snippets.append(PLAYBOOK_OPEX)
    if abs(safe_float(summary.get("flip_distance_pct"))) <= 0.004:
        snippets.append(PLAYBOOK_NEAR_FLIP)
    return snippets


def build_event_system_addon(summary: dict[str, Any]) -> str:
    parts = event_prompt_snippets(summary)
    return "\n\n".join(parts)
