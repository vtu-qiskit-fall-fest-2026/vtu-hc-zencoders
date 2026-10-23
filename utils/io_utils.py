"""I/O Utilities for Q-Nav.

Provides robust JSON and CSV loading and saving functionality,
directory creation helpers, and numpy-compatible JSON serialization.
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np


class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy scalars, arrays, and Path objects."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure that the given directory or parent of given file exists.

    Args:
        path: Directory path or file path.

    Returns:
        Path object of the directory.
    """
    p = Path(path)
    if p.suffix:  # It's a file
        p.parent.mkdir(parents=True, exist_ok=True)
        return p.parent
    else:  # It's a directory
        p.mkdir(parents=True, exist_ok=True)
        return p


def save_json(
    data: Any,
    filepath: Union[str, Path],
    indent: int = 2,
    cls: type = NumpyJSONEncoder,
) -> Path:
    """Save data structure to a JSON file.

    Args:
        data: Data to serialize.
        filepath: Destination file path.
        indent: JSON indentation.
        cls: Custom JSON encoder class.

    Returns:
        Path to the written file.
    """
    target = Path(filepath)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, cls=cls)
    return target


def load_json(filepath: Union[str, Path]) -> Any:
    """Load JSON data from a file.

    Args:
        filepath: Path to the JSON file.

    Returns:
        Deserialized Python object.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If JSON syntax is invalid.
    """
    target = Path(filepath)
    if not target.exists():
        raise FileNotFoundError(f"JSON file not found: {target.resolve()}")
    with open(target, "r", encoding="utf-8") as f:
        return json.load(f)


def save_csv(
    data: List[Dict[str, Any]],
    filepath: Union[str, Path],
    fieldnames: Optional[List[str]] = None,
    append: bool = False,
) -> Path:
    """Save list of dictionaries as a CSV file.

    Args:
        data: List of dictionary records.
        filepath: Destination CSV path.
        fieldnames: Column headers. If None, derived from first record keys.
        append: If True, append to existing file without writing headers again.

    Returns:
        Path to the saved CSV file.
    """
    target = Path(filepath)
    target.parent.mkdir(parents=True, exist_ok=True)

    if not data and not fieldnames:
        # Create empty file
        target.touch(exist_ok=True)
        return target

    if fieldnames is None and data:
        fieldnames = list(data[0].keys())

    file_exists = target.exists() and target.stat().st_size > 0
    mode = "a" if append else "w"

    with open(target, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames or [])
        if not append or not file_exists:
            writer.writeheader()
        for row in data:
            # Convert any numpy types in row
            cleaned_row = {}
            for k, v in row.items():
                if isinstance(v, (np.integer, np.floating)):
                    cleaned_row[k] = v.item()
                elif isinstance(v, np.ndarray):
                    cleaned_row[k] = v.tolist()
                else:
                    cleaned_row[k] = v
            writer.writerow(cleaned_row)

    return target


def load_csv(filepath: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load records from a CSV file into a list of dictionaries.

    Args:
        filepath: Path to the CSV file.

    Returns:
        List of row dictionaries.
    """
    target = Path(filepath)
    if not target.exists():
        raise FileNotFoundError(f"CSV file not found: {target.resolve()}")

    records = []
    with open(target, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(dict(row))
    return records
