"""Centralized logging configuration for iBERT.

This module provides a singleton logging infrastructure with:
- Console output with color formatting
- Rotating file logs
- Per-module loggers
- Performance tracking utilities
"""

import functools
import logging
import logging.handlers
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional


class ColoredFormatter(logging.Formatter):
    """Formatter that adds color to console output."""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with color."""
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
        return super().format(record)


class IBERTLogger:
    """Singleton logger for iBERT with structured logging."""

    _instance: Optional["IBERTLogger"] = None
    _initialized: bool = False

    def __new__(cls) -> "IBERTLogger":
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize logging infrastructure."""
        if self._initialized:
            return

        # Create root logger for ibert
        self._logger = logging.getLogger("ibert")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False

        # Console handler with color formatting
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(logging.INFO)
        console_formatter = ColoredFormatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(console_formatter)
        self._logger.addHandler(console_handler)

        # File handler with detailed formatting
        log_dir = Path(".logs")
        log_dir.mkdir(exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "ibert.log",
            maxBytes=10_000_000,  # 10MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(funcName)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        self._logger.addHandler(file_handler)

        self._initialized = True
        self._logger.info("iBERT logging system initialized")

    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger for a specific module.

        Parameters
        ----------
        name : str
            Module name (e.g., 'models.mistral', 'tasks.completion')

        Returns
        -------
        logging.Logger
            Configured logger for the module
        """
        return logging.getLogger(f"ibert.{name}")

    def set_level(self, level: str) -> None:
        """Set global logging level.

        Parameters
        ----------
        level : str
            Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL
        """
        numeric_level = getattr(logging, level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {level}")
        self._logger.setLevel(numeric_level)


# Singleton instance
_logger_instance = IBERTLogger()


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a module.

    Parameters
    ----------
    name : str
        Module name

    Returns
    -------
    logging.Logger
        Configured logger

    Examples
    --------
    >>> from ibert.logging import get_logger
    >>> log = get_logger("models.mistral")
    >>> log.info("Model loaded successfully")
    """
    return _logger_instance.get_logger(name)


def set_log_level(level: str) -> None:
    """Set global logging level.

    Parameters
    ----------
    level : str
        Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL

    Examples
    --------
    >>> from ibert.logging import set_log_level
    >>> set_log_level("DEBUG")
    """
    _logger_instance.set_level(level)


def log_performance(func: Callable) -> Callable:
    """Decorator to log function performance.

    Parameters
    ----------
    func : Callable
        Function to monitor

    Returns
    -------
    Callable
        Wrapped function with performance logging

    Examples
    --------
    >>> @log_performance
    ... def generate_data():
    ...     # Long-running operation
    ...     pass
    """
    logger = get_logger(func.__module__)

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        func_name = func.__qualname__

        logger.debug("Starting %s", func_name)

        try:
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start_time

            logger.info("Completed %s in %.2f seconds", func_name, elapsed)

            return result

        except Exception as e:
            elapsed = time.perf_counter() - start_time

            logger.error(
                "Failed %s after %.2f seconds: %s", func_name, elapsed, str(e), exc_info=True
            )
            raise

    return wrapper


class LogContext:
    """Context manager for scoped logging with indentation.

    Examples
    --------
    >>> with LogContext("models.mistral", "Loading model"):
    ...     # Operations here will be logged with context
    ...     pass
    """

    def __init__(self, logger_name: str, operation: str):
        """Initialize log context.

        Parameters
        ----------
        logger_name : str
            Logger name
        operation : str
            Operation description
        """
        self.logger = get_logger(logger_name)
        self.operation = operation
        self.start_time: float | None = None

    def __enter__(self) -> "LogContext":
        """Enter context."""
        self.start_time = time.perf_counter()
        self.logger.info("→ %s started", self.operation)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context."""
        if self.start_time is not None:
            elapsed = time.perf_counter() - self.start_time

            if exc_type is None:
                self.logger.info("✓ %s completed in %.2f seconds", self.operation, elapsed)
            else:
                self.logger.error(
                    "✗ %s failed after %.2f seconds: %s", self.operation, elapsed, str(exc_val)
                )
