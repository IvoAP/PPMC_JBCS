"""Seed the active random-number generators without importing ML stacks."""

import random
from collections.abc import Callable, Iterable

import numpy as np


def seed_generators(
    seed: int, framework_seeders: Iterable[Callable[[int], object]] = (),
) -> np.random.Generator:
    """Seed Python, legacy NumPy and optional framework hooks; return a local RNG."""
    random.seed(seed)
    np.random.seed(seed)
    for set_seed in framework_seeders:
        set_seed(seed)
    return np.random.default_rng(seed)
