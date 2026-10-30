"""
cosmo_mind.utils.logging
========================

Lightweight JSON-structured logging used across the package.  Every log
record carries a ``service`` tag so records can be filtered downstream in
Qdrant/Neo4j/log aggregators.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "service": getattr(record, "service", "cosmo_mind"),
            "msg": record.getMessage(),
        }
        for key in ("module", "funcName", "lineno"):
            payload[key] = getattr(record, key, None)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def get_logger(name: str = "cosmo_mind") -> logging.Logger:
    """Return a configured JSON logger; safe to call from anywhere."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def now_iso() -> str:
    """ISO-8601 UTC timestamp utility."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


__all__ = ["get_logger", "now_iso"]
