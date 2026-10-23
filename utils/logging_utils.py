"""Logging Utilities for Q-Nav.

Provides configurable console and file logging with standardized formatting.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from config import DEFAULT_CONFIG


def setup_logger(
    name: str = "qnav",
    log_file: Optional[Path] = None,
    level: str = "INFO",
    to_console: bool = True,
    to_file: bool = True,
) -> logging.Logger:
    """Set up and return a configured logger instance.

    Args:
        name: Name of the logger.
        log_file: Path to the log file. If None, uses default from config.
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR').
        to_console: If True, adds a StreamHandler for stdout.
        to_file: If True, adds a FileHandler.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)

    # Avoid adding duplicate handlers if logger was already configured
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if to_file:
        file_path = log_file or DEFAULT_CONFIG.logging.log_file
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "qnav") -> logging.Logger:
    """Retrieve an existing logger or create one with default settings."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(
            name=name,
            log_file=DEFAULT_CONFIG.logging.log_file,
            level=DEFAULT_CONFIG.logging.log_level,
            to_console=DEFAULT_CONFIG.logging.log_to_console,
            to_file=DEFAULT_CONFIG.logging.log_to_file,
        )
    return logger
