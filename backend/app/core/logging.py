"""
Logging Configuration Module.

Configures structured logging for the application using Loguru.
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Any

from loguru import logger
from backend.app.core.config import settings


class JSONFormatter:
    """Custom JSON formatter for structured logging."""

    def __call__(self, record: dict) -> str:
        """Format a log record as JSON.

        Args:
            record: Log record dictionary.

        Returns:
            str: JSON formatted log string.
        """
        log_entry: dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record["level"].name,
            "module": record["name"],
            "function": record["function"],
            "line": record["line"],
            "message": record["message"],
        }

        # Add exception info if present
        if record.get("exception"):
            log_entry["exception"] = str(record["exception"])

        # Add extra fields if present
        if record.get("extra"):
            log_entry["extra"] = record["extra"]

        return json.dumps(log_entry) + "\n"


def setup_logging() -> None:
    """Configure application logging with Loguru."""
    # Remove default handler
    logger.remove()

    # Console handler with colorized output
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    log_level = settings.LOG_LEVEL.upper()

    # Add console handler
    logger.add(
        sys.stdout,
        format=console_format,
        level=log_level,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # Add file handler with JSON formatting
    log_file = Path(settings.LOG_FILE)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger.add(
        str(log_file),
        format=JSONFormatter(),
        level=log_level,
        rotation="10 MB",
        retention="30 days",
        compression="zip",
        backtrace=True,
        diagnose=True,
    )

    # Add error file handler
    error_log_file = log_file.parent / "error.log"
    logger.add(
        str(error_log_file),
        format=JSONFormatter(),
        level="ERROR",
        rotation="10 MB",
        retention="30 days",
        compression="zip",
        backtrace=True,
        diagnose=True,
    )

    logger.info(f"Logging configured. Level: {log_level}, File: {log_file}")


def get_logger(name: str):
    """Get a logger instance for a specific module.

    Args:
        name: Module name.

    Returns:
        Logger: Configured logger instance.
    """
    return logger.bind(module=name)