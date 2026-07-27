"""Structured logging with a correlation ID that follows an event across services.

The value here is operational: when a quote looks wrong on the dashboard you
grab its ``correlation_id`` and get every log line from ingestion through fill,
across all eight processes, with one grep.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

import structlog

_correlation_id: ContextVar[str | None] = ContextVar("amme_correlation_id", default=None)
_configured = False


def set_correlation_id(cid: str | None) -> None:
    _correlation_id.set(cid)


def get_correlation_id() -> str | None:
    return _correlation_id.get()


@contextmanager
def correlation_scope(cid: str | None) -> Iterator[None]:
    """Bind ``cid`` for the duration of the block, then restore the previous value."""
    token = _correlation_id.set(cid)
    try:
        yield
    finally:
        _correlation_id.reset(token)


def _inject_correlation_id(
    _logger: Any, _method: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    cid = _correlation_id.get()
    if cid is not None:
        event_dict.setdefault("correlation_id", cid)
    return event_dict


def configure_logging(
    service: str, level: str = "INFO", json_output: bool = True
) -> structlog.stdlib.BoundLogger:
    """Idempotently configure structlog and return a logger bound to ``service``."""
    global _configured

    if not _configured:
        logging.basicConfig(
            format="%(message)s",
            stream=sys.stdout,
            level=getattr(logging, level.upper(), logging.INFO),
            force=True,
        )
        processors: list[Any] = [
            structlog.contextvars.merge_contextvars,
            _inject_correlation_id,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
        ]
        processors.append(
            structlog.processors.JSONRenderer()
            if json_output
            else structlog.dev.ConsoleRenderer(colors=False)
        )
        structlog.configure(
            processors=processors,
            wrapper_class=structlog.make_filtering_bound_logger(
                getattr(logging, level.upper(), logging.INFO)
            ),
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
        _configured = True

    logger: structlog.stdlib.BoundLogger = structlog.get_logger(service).bind(service=service)
    return logger


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name)
    return logger


__all__ = [
    "configure_logging",
    "correlation_scope",
    "get_correlation_id",
    "get_logger",
    "set_correlation_id",
]
