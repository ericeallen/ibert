"""I/O utilities for JSONL, Parquet, NDJSON formats."""

import json
from pathlib import Path
from typing import Any, Dict, Generator, List

import pandas as pd


def read_jsonl(path: Path) -> Generator[Dict[str, Any], None, None]:
    """Read JSONL file line by line.

    Parameters
    ----------
    path : Path
        Path to JSONL file

    Yields
    ------
    dict
        Parsed JSON object from each line
    """
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(data: List[Dict[str, Any]], path: Path, pretty: bool = False) -> None:
    """Write list of dicts to JSONL file.

    Parameters
    ----------
    data : list of dict
        Records to write
    path : Path
        Output path
    pretty : bool
        If True, use indented JSON per line (not standard JSONL)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in data:
            if pretty:
                f.write(json.dumps(record, indent=2) + "\n")
            else:
                f.write(json.dumps(record) + "\n")


def append_jsonl(record: Dict[str, Any], path: Path) -> None:
    """Append single record to JSONL file.

    Parameters
    ----------
    record : dict
        Record to append
    path : Path
        Output path
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def read_parquet(path: Path) -> pd.DataFrame:
    """Read Parquet file to DataFrame."""
    return pd.read_parquet(path)


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write DataFrame to Parquet file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
