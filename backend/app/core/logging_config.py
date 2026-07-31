"""Logging setup shared by the API, the ETL scripts and the training pipeline.

The ETL needs to *justify* every row it drops, so logging is treated as a first
class concern rather than an afterthought. ``configure_logging`` is idempotent -
calling it from several entry points will not duplicate handlers.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

_CONFIGURED = False

# ANSI colours, used only when stdout is an interactive terminal.
_COLOURS = {
    "DEBUG": "\033[36m",
    "INFO": "\033[32m",
    "WARNING": "\033[33m",
    "ERROR": "\033[31m",
    "CRITICAL": "\033[1;31m",
}
_RESET = "\033[0m"


class _ConsoleFormatter(logging.Formatter):
    """Readable console output, with colour when attached to a TTY."""

    def __init__(self, use_colour: bool) -> None:
        super().__init__(
            fmt="%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s",
            datefmt="%H:%M:%S",
        )
        self.use_colour = use_colour

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        if self.use_colour:
            colour = _COLOURS.get(record.levelname)
            if colour:
                text = f"{colour}{text}{_RESET}"
        return text


def configure_logging(level: str | int | None = None, *, force: bool = False) -> None:
    """Install the console handler on the root logger.

    Parameters
    ----------
    level:
        Log level name or number. Defaults to ``settings.log_level``.
    force:
        Reconfigure even if logging was already set up.
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    # Imported lazily so this module stays usable if config validation fails.
    if level is None:
        try:
            from app.core.config import settings

            level = settings.log_level
        except Exception:  # pragma: no cover - fallback for broken env
            level = "INFO"

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_ConsoleFormatter(use_colour=sys.stdout.isatty()))
    root.addHandler(handler)
    root.setLevel(level)

    # These libraries are chatty at INFO and drown out our own messages.
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger (configuring logging on first use)."""
    configure_logging()
    return logging.getLogger(name)


def log_section(logger: logging.Logger, title: str, **fields: Any) -> None:
    """Log a visually distinct section header.

    The ETL and training pipelines print a lot of output; these banners make
    the phase boundaries easy to spot when scanning a terminal.
    """
    logger.info("=" * 74)
    logger.info(title.upper())
    if fields:
        for key, value in fields.items():
            logger.info("  %-24s %s", f"{key}:", value)
    logger.info("=" * 74)


__all__ = ["configure_logging", "get_logger", "log_section"]
