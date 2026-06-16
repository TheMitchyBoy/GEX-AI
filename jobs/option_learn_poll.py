"""Poll UW for option quotes and learn price movements from DB + GEX context."""

from __future__ import annotations

import logging
import time

import config
from services.option_pipeline import run_option_cycle

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def poll_loop() -> None:
    tickers = config.SUPPORTED_TICKERS or [config.DEFAULT_TICKER]
    logger.info("Starting option learn poller for %s (every %ss)", tickers, config.OPTION_POLL_SEC)
    while True:
        for ticker in tickers:
            try:
                result = run_option_cycle(ticker)
                if result.get("ingest", {}).get("ok"):
                    n = len(result["ingest"].get("stored") or [])
                    logger.info("Option cycle %s: stored %s quotes", ticker, n)
                else:
                    err = result.get("ingest", {}).get("error", "unknown")
                    logger.warning("Option cycle %s skipped: %s", ticker, err)
            except Exception:
                logger.exception("Option cycle failed for %s", ticker)
        time.sleep(config.OPTION_POLL_SEC)


if __name__ == "__main__":
    poll_loop()
