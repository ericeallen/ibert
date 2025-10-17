"""Mine SQL→Ibis examples from GitHub repositories."""

import re
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json


class GitHubMiner:
    """Mine code examples from GitHub repositories."""

    def __init__(self, cache_dir: Path = Path("data/mining/repos")):
        """Initialize miner.

        Parameters
        ----------
        cache_dir : Path
            Directory to cache cloned repositories
        """
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def clone_repo(self, repo_url: str, repo_name: str) -> Path:
        """Clone a GitHub repository.

        Parameters
        ----------
        repo_url : str
            GitHub repository URL
        repo_name : str
            Name for the local directory

        Returns
        -------
        Path
            Path to cloned repository
        """
        repo_path = self.cache_dir / repo_name

        if repo_path.exists():
            print(f"Repository already cloned: {repo_path}")
            return repo_path

        print(f"Cloning {repo_url}...")
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(repo_path)],
            check=True,
            capture_output=True
        )

        return repo_path

    def extract_python_files(self, repo_path: Path) -> List[Path]:
        """Find all Python files in a repository.

        Parameters
        ----------
        repo_path : Path
            Path to repository

        Returns
        -------
        list of Path
            Python files
        """
        return list(repo_path.rglob("*.py"))

    def find_sql_conversions(self, python_file: Path) -> List[Dict[str, Any]]:
        """Find SQL→Ibis conversions in a Python file.

        Parameters
        ----------
        python_file : Path
            Python file to analyze

        Returns
        -------
        list of dict
            Found examples with SQL and Ibis code
        """
        try:
            content = python_file.read_text()
        except Exception:
            return []

        examples = []

        # Pattern 1: Look for con.sql() calls
        # This captures SQL strings passed to Ibis
        sql_pattern = r'(?:con|connection|backend)\.sql\(["\'](.+?)["\']\)'
        for match in re.finditer(sql_pattern, content, re.DOTALL):
            sql = match.group(1)
            examples.append({
                "source": "con.sql()",
                "file": str(python_file),
                "sql": sql,
                "ibis": None,  # We'll try to extract this separately
            })

        # Pattern 2: Look for SQL in test assertions or comparisons
        # e.g., ibis.to_sql(expr) == "SELECT ..."
        test_pattern = r'ibis\.to_sql\((.+?)\)\s*==\s*["\'](.+?)["\']\s*$'
        for match in re.finditer(test_pattern, content, re.MULTILINE):
            ibis_expr = match.group(1).strip()
            sql = match.group(2)
            examples.append({
                "source": "test_assertion",
                "file": str(python_file),
                "sql": sql,
                "ibis": ibis_expr,
            })

        # Pattern 3: Look for SQL in docstrings with Examples section
        docstring_pattern = r'"""(.+?)"""'
        for match in re.finditer(docstring_pattern, content, re.DOTALL):
            docstring = match.group(1)
            if "SELECT" in docstring.upper() and ">>>" in docstring:
                # This is an example with SQL - extract it
                examples.append({
                    "source": "docstring",
                    "file": str(python_file),
                    "docstring": docstring,
                })

        return examples


def mine_ibis_repo() -> List[Dict[str, Any]]:
    """Mine examples from the official Ibis repository.

    Returns
    -------
    list of dict
        Extracted examples
    """
    miner = GitHubMiner()

    # Clone Ibis repository
    repo_path = miner.clone_repo(
        "https://github.com/ibis-project/ibis.git",
        "ibis"
    )

    # Focus on test files and documentation examples
    test_dirs = [
        repo_path / "ibis" / "tests",
        repo_path / "docs" / "examples",
    ]

    all_examples = []

    for test_dir in test_dirs:
        if not test_dir.exists():
            continue

        python_files = list(test_dir.rglob("*.py"))
        print(f"Scanning {len(python_files)} files in {test_dir.name}...")

        for py_file in python_files:
            examples = miner.find_sql_conversions(py_file)
            all_examples.extend(examples)

    print(f"\nFound {len(all_examples)} potential examples")
    return all_examples


def save_mined_examples(examples: List[Dict[str, Any]], output_path: Path):
    """Save mined examples to JSON.

    Parameters
    ----------
    examples : list of dict
        Mined examples
    output_path : Path
        Output file path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")

    print(f"Saved {len(examples)} examples to {output_path}")


if __name__ == "__main__":
    examples = mine_ibis_repo()

    output_path = Path("data/mining/ibis_mined.jsonl")
    save_mined_examples(examples, output_path)
