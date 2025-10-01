"""Centralised logging configuration for the MINIMINIMOON project."""

from __future__ import annotations

import logging
import os
from typing import Optional

LOGGER = logging.getLogger(__name__)


def configure_logging(log_level: Optional[str] = None) -> None:
    """Configure logging once for the entire application.

    The configuration honours the ``LOG_LEVEL`` environment variable unless an
    explicit ``log_level`` argument is provided. Subsequent calls simply update
    the log level instead of installing duplicate handlers, keeping test output
    tidy while allowing scripts to opt-in to more verbose logging.

    Args:
        log_level: Optional override for the desired log level.
    """

    level_string = log_level or os.getenv("LOG_LEVEL", "INFO")
    level = get_log_level(level_string)

    root_logger = logging.getLogger()
    if root_logger.handlers:
        root_logger.setLevel(level)
        for handler in root_logger.handlers:
            handler.setLevel(level)
    else:
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    LOGGER.debug(
        "Logging configured with level: %s (%s)",
        logging.getLevelName(level),
        level,
    )

    if level_string.upper() != logging.getLevelName(level):
        LOGGER.warning(
            "Invalid LOG_LEVEL value '%s' - falling back to %s",
            level_string,
            logging.getLevelName(level),
        )


def get_log_level(level_string: str) -> int:
    """
    Convert log level string to logging constant with validation.

    Args:
        level_string: Log level as string (e.g., 'DEBUG', 'INFO', 'WARNING')

    Returns:
        Logging level constant (int)
    """
    # Normalize to uppercase
    level_string = level_string.upper()

    # Valid log levels
    valid_levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "WARN": logging.WARNING,  # Common alias
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
        "FATAL": logging.CRITICAL,  # Common alias
    }

    return valid_levels.get(level_string, logging.INFO)


# Configure logging on module import
configure_logging()
