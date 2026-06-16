"""Interactive GEX LLM agent chat dashboard."""

from __future__ import annotations

import streamlit as st

import config
from db.connection import get_connection
from db.features import enrich_snapshot_metrics, safe_float
from db.loader import load_snapshot_history
from db.queries import get_latest_ts
from models.llm_agent import SUGGESTED_PROMPTS, build_agent_context, chat_with_agent
from models.llm_client import is_llm_configured

st.set_page_config(page_title="GEX Agent", layout="wide", page_icon="🤖")

st.markdown(
    """
    <style>
    .stChatMessage { border-radius: 12px; }
    div[data-testid="stSidebar"] { background: #0f172a; }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_session() -> None:
    if "agent_messages" not in st.session_state:
        st.session_state.agent_messages = []
    if "agent_context" not in st.session_state:
        st.session_state.agent_context = None
    if "agent_ticker" not in st.session_state:
        st.session_state.agent_ticker = config.DEFAULT_TICKER


def render_context_panel(ctx: dict | None) -> None:
    st.subheader("Live context")
    if not ctx or not ctx.get("bundle"):
        st.caption("No context loaded")
        return
    summary = ctx["bundle"].get("summary") or {}
    c1, c2 = st.columns(2)
    c1.metric("Spot", f"{safe_float(summary.get('spot')):,.2f}")
    c2.metric("Total GEX", f"{safe_float(summary.get('total_gex_bn_per_pct')):.3f}")
    st.caption(
        f"**{summary.get('net_gamma_regime')}** · flip {safe_float(summary.get('gamma_flip')):,.0f} · "
        f"ts `{ctx['bundle'].get('snapshot_ts')}`"
    )
    if ctx.get("llm_forecast"):
        lf = ctx["llm_forecast"]
        st.info(
            f"LLM: {lf.get('predicted_regime')} · ΔGEX {safe_float(lf.get('predicted_delta_gex_bn')):.3f} · "
            f"bias {lf.get('spot_bias')} · conf {safe_float(lf.get('confidence')):.0%}"
        )
    st.caption(f"~{ctx.get('estimated_tokens', 0)} context tokens")


def main() -> None:
    init_session()

    st.title("GEX Market Agent")
    st.caption("Chat with an LLM analyst grounded in live Postgres GEX snapshots")

    if not config.DATABASE_URL:
        st.error("DATABASE_URL is not configured.")
        st.stop()

    if not is_llm_configured():
        st.warning("Set `OPENAI_API_KEY` in `.env` to enable the agent.")

    with st.sidebar:
        ticker = st.selectbox(
            "Ticker",
            config.SUPPORTED_TICKERS,
            index=config.SUPPORTED_TICKERS.index(st.session_state.agent_ticker)
            if st.session_state.agent_ticker in config.SUPPORTED_TICKERS
            else 0,
        )
        lookback = st.slider("Context lookback (days)", 7, 180, config.LOOKBACK_DAYS)
        st.session_state.agent_ticker = ticker

        try:
            with get_connection() as conn:
                latest_ts = get_latest_ts(conn, ticker)
            st.caption(f"Latest snapshot: `{latest_ts}`")
        except Exception as exc:
            st.error(str(exc))
            st.stop()

        if st.button("Refresh market context", use_container_width=True):
            try:
                history = load_snapshot_history(ticker, lookback_days=lookback)
                st.session_state.agent_context = build_agent_context(history, lookback_days=lookback)
                st.success("Context refreshed")
            except Exception as exc:
                st.error(f"Context load failed: {exc}")

        if st.button("Clear chat", use_container_width=True):
            st.session_state.agent_messages = []
            st.rerun()

        st.divider()
        st.subheader("Suggested prompts")
        for prompt in SUGGESTED_PROMPTS:
            if st.button(prompt, key=f"prompt_{prompt[:24]}", use_container_width=True):
                st.session_state.pending_prompt = prompt

    col_chat, col_ctx = st.columns([2, 1])

    with col_ctx:
        render_context_panel(st.session_state.agent_context)
        if st.session_state.agent_context and st.checkbox("Show raw context", value=False):
            st.json(st.session_state.agent_context.get("bundle"))

    with col_chat:
        for msg in st.session_state.agent_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        pending = st.session_state.pop("pending_prompt", None)
        user_input = st.chat_input("Ask about regime, flip, walls, forecasts, trade setups…")
        prompt = pending or user_input

        if prompt:
            st.session_state.agent_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Analyzing GEX data…"):
                    try:
                        history = load_snapshot_history(ticker, lookback_days=lookback)
                        if st.session_state.agent_context is None:
                            st.session_state.agent_context = build_agent_context(history, lookback_days=lookback)

                        result = chat_with_agent(
                            history,
                            st.session_state.agent_messages,
                            lookback_days=lookback,
                            refresh_context=False,
                            context=st.session_state.agent_context,
                        )
                        if result.get("error") and not result.get("reply"):
                            st.error(result["error"])
                            reply = "I couldn't process that request. Check your API key and database connection."
                        else:
                            reply = result.get("reply") or "No response."
                            st.session_state.agent_context = result.get("context") or st.session_state.agent_context
                        st.markdown(reply)
                        st.session_state.agent_messages.append({"role": "assistant", "content": reply})
                    except Exception as exc:
                        st.error(str(exc))


if __name__ == "__main__":
    main()
