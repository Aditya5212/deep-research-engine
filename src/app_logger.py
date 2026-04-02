import logging
import os

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Root application logger
logger = logging.getLogger("deep_research_engine")

# Avoid adding duplicate handlers if module imported multiple times
if not logger.handlers:
    handler = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    handler.setFormatter(fmt)
    logger.addHandler(handler)

logger.setLevel(LOG_LEVEL)


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child logger under the app logger namespace for convenience.

    Example: `get_logger("a2a.agent")` -> logger named `deep_research_engine.a2a.agent`.
    """
    if name:
        return logger.getChild(name)
    return logger
