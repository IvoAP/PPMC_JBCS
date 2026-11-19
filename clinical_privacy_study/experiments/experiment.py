"""Shared execution contract for study stages."""

import logging
from dataclasses import dataclass
from pathlib import Path

from config import StudyConfig


@dataclass(frozen=True)
class Experiment:
    """A named stage; override run when its scientific protocol is ready."""

    name: str

    def run(
        self, config: StudyConfig, run_directory: Path, logger: logging.Logger,
    ) -> str:
        """Report a placeholder without reading data or fabricating outputs."""
        logger.info("%s: skipped (not implemented)", self.name)
        return "not_implemented"
