"""Check independent stage execution and run isolation."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from config import Paths, StudyConfig
from main import STAGES, run
from utils.file_manager import load_metadata


class ExecutionTests(unittest.TestCase):
    def test_each_stage_runs_alone_without_inputs(self) -> None:
        with TemporaryDirectory() as workspace:
            config = StudyConfig(paths=Paths(Path(workspace)))
            runs = set()
            placeholders = STAGES.keys() - {"preprocessing"}
            for name in placeholders:
                with self.subTest(stage=name):
                    directory = run(name, config)
                    runs.add(directory)
                    self.assertEqual(load_metadata(directory)["stages"],
                                     {name: "not_implemented"})
                    self.assertIn(name, (directory / "run.log").read_text("utf-8"))
            self.assertEqual(len(runs), len(placeholders))
            self.assertFalse(config.paths.dataset.exists())
