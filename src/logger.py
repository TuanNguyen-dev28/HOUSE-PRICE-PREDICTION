"""
Centralized logging configuration for the House Price Prediction project.
Replaces scattered print() calls with structured, leveled logging.
Supports simultaneous console + file output with UTF-8 encoding.
"""
import logging
import os
import sys
from datetime import datetime


# ─── Constants ────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logger(
    name: str = "house_price",
    level: int = logging.INFO,
    log_to_file: bool = True,
    log_filename: str = None,
) -> logging.Logger:
    """
    Create and configure a logger instance.

    Args:
        name: Logger name (typically module name like 'train', 'predict').
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_to_file: Whether to also write logs to a file.
        log_filename: Custom log filename. Defaults to '{name}_{date}.log'.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if logger already configured
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    # ── Console handler (UTF-8 safe) ──
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    # Force UTF-8 encoding on Windows
    if sys.platform == "win32":
        try:
            import io
            console_handler.stream = io.TextIOWrapper(
                sys.stdout.buffer, encoding="utf-8", errors="replace"
            )
        except Exception:
            pass  # Fallback to default stdout

    logger.addHandler(console_handler)

    # ── File handler ──
    if log_to_file:
        os.makedirs(LOG_DIR, exist_ok=True)
        if log_filename is None:
            date_str = datetime.now().strftime("%Y%m%d")
            log_filename = f"{name}_{date_str}.log"
        log_path = os.path.join(LOG_DIR, log_filename)

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)  # File always captures DEBUG+
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get an existing logger or create a new one with default settings.
    Convenience wrapper for modules that just need a simple logger.

    Args:
        name: Logger name (e.g., 'train', 'predict', 'app').

    Returns:
        logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger
