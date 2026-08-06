"""
=====================================================================
XAI Attack and Defense Framework - Logging Utility
Author : Mithin Sagar S  (https://github.com/mithinsagar)
=====================================================================
"""

from __future__ import annotations

import logging
import os
from pathlib import Path


_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(
    name: str,
    log_file: str | os.PathLike | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Return a configured logger. If `log_file` is provided, a rotating-friendly
    file handler is attached in addition to the console handler.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    logger.propagate = False
    return logger
