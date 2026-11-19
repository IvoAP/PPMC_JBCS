"""Check reproducibility utilities used by future anonymizers."""

import random
import unittest
from unittest.mock import Mock

import numpy as np

from utils.reproducibility import seed_generators


class ReproducibilityTests(unittest.TestCase):
    def test_seed_replays_streams_and_calls_framework_hook(self) -> None:
        hook = Mock()
        rng = seed_generators(123, (hook,))
        first = (random.random(), np.random.random(), rng.random())
        rng = seed_generators(123)
        self.assertEqual(first, (random.random(), np.random.random(), rng.random()))
        hook.assert_called_once_with(123)
