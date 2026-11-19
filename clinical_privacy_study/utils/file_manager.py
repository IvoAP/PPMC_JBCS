"""Filesystem and artifact helpers. CSV values are loaded as text."""

import csv
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path


def create_directories(paths: Iterable[Path]) -> None:
    """Create missing directories without modifying existing contents."""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: Mapping[str, object]) -> None:
    """Save an indented UTF-8 JSON object, rejecting nonfinite numbers."""
    text = json.dumps(payload, indent=2, allow_nan=False, ensure_ascii=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, object]:
    """Read a JSON object; reject unexpected top-level types."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path.name}.")
    return value


def save_csv(
    path: Path, columns: Sequence[str], rows: Iterable[Mapping[str, object]],
) -> None:
    """Write a table with an explicit, stable column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def load_csv(path: Path) -> list[dict[str, str]]:
    """Load a rectangular CSV with named, unique columns."""
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        header = next(reader, [])
        if not all(header) or not header or len(header) != len(set(header)):
            raise ValueError(f"Invalid column names in {path.name}.")
        return [dict(zip(header, row, strict=True)) for row in reader]


def save_metadata(directory: Path, metadata: Mapping[str, object]) -> None:
    """Persist the manifest in its run-specific directory."""
    save_json(directory / "metadata.json", metadata)


def load_metadata(directory: Path) -> dict[str, object]:
    """Load the manifest for a single execution."""
    return load_json(directory / "metadata.json")
