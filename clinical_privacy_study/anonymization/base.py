"""Contract for train-fitted tabular anonymizers."""

from typing import Protocol, TypeVar

Table = TypeVar("Table")


class Anonymizer(Protocol[Table]):
    """Fit on training features and transform without refitting."""

    def fit(self, training_features: Table) -> None:
        """Learn transformation parameters from training data only."""
        ...

    def transform(self, features: Table) -> Table:
        """Apply previously fitted parameters to a feature table."""
        ...
