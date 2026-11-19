"""Run-scoped console and UTF-8 file logging."""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def run_logger(path: Path) -> Iterator[logging.Logger]:
    """Open isolated handlers and release them when the run finishes."""
    logger = logging.Logger("clinical_privacy_study", level=logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    console = logging.StreamHandler()
    file_handler = logging.FileHandler(path, encoding="utf-8")
    for handler in (console, file_handler):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    try:
        yield logger
    finally:
        for handler in (console, file_handler):
            logger.removeHandler(handler)
            handler.close()
