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

        # Pattern 1: Table.sql() or Backend.sql() - THE MAIN SQL→Ibis PATTERN
        # e.g., expr = t.sql("SELECT x FROM t WHERE x > 0")
        # or:   expr = con.sql("SELECT x FROM t WHERE x > 0")
        sql_method_pattern = r'(\w+)\s*=\s*(?:\w+)\.sql\(\s*["\'](.+?)["\']\s*\)'
        for match in re.finditer(sql_method_pattern, content, re.DOTALL):
            var_name = match.group(1).strip()
            sql = match.group(2).strip()

            if "SELECT" in sql.upper():  # Filter for actual SQL queries
                examples.append({
                    "source": "table.sql()",
                    "file": str(python_file),
                    "sql": sql,
                    "ibis_var": var_name,
                })

        # Pattern 2: Direct SQL strings in function calls
        # con.sql("SELECT ...")
        direct_sql_pattern = r'\.sql\(\s*["\'](.+?)["\']\s*\)'
        for match in re.finditer(direct_sql_pattern, content, re.DOTALL):
            sql = match.group(1).strip()
            if "SELECT" in sql.upper():
                examples.append({
                    "source": "direct_sql",
                    "file": str(python_file),
                    "sql": sql,
                })

        # Pattern 3: SQL strings in multi-line format
        # sql = """
        # SELECT ...
        # """
        multiline_sql_pattern = r'sql\s*=\s*"""(.+?)"""'
        for match in re.finditer(multiline_sql_pattern, content, re.DOTALL):
            sql = match.group(1).strip()
            if "SELECT" in sql.upper():
                examples.append({
                    "source": "multiline_sql",
                    "file": str(python_file),
                    "sql": sql,
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
        repo_path / "docs" / "tutorials",
        repo_path / "docs" / "how-to",
    ]

    all_examples = []

    for test_dir in test_dirs:
        if not test_dir.exists():
            print(f"Directory not found: {test_dir}")
            continue

        python_files = list(test_dir.rglob("*.py"))
        print(f"Scanning {len(python_files)} Python files in {test_dir.name}...")

        for py_file in python_files:
            examples = miner.find_sql_conversions(py_file)
            if examples:
                print(f"  Found {len(examples)} in {py_file.name}")
            all_examples.extend(examples)

    print(f"\nFound {len(all_examples)} potential examples from Ibis repo")
    return all_examples


def mine_ibis_tutorial() -> List[Dict[str, Any]]:
    """Mine examples from the Ibis tutorial repository.

    Returns
    -------
    list of dict
        Extracted examples
    """
    miner = GitHubMiner()

    # Clone Ibis tutorial repository
    repo_path = miner.clone_repo(
        "https://github.com/ibis-project/ibis-tutorial.git",
        "ibis-tutorial"
    )

    all_examples = []

    # Search all Python files in the tutorial
    python_files = list(repo_path.rglob("*.py"))
    print(f"\nScanning {len(python_files)} Python files in ibis-tutorial...")

    for py_file in python_files:
        examples = miner.find_sql_conversions(py_file)
        if examples:
            print(f"  Found {len(examples)} in {py_file.name}")
        all_examples.extend(examples)

    # Also search notebooks
    notebooks = list(repo_path.rglob("*.ipynb"))
    print(f"Scanning {len(notebooks)} Jupyter notebooks in ibis-tutorial...")

    for notebook in notebooks:
        # Use the doc extractor for notebooks
        from src.datagen.mining.ibis_doc_extractor import extract_from_jupyter
        examples = extract_from_jupyter(notebook)
        if examples:
            print(f"  Found {len(examples)} in {notebook.name}")
        all_examples.extend(examples)

    print(f"\nFound {len(all_examples)} potential examples from ibis-tutorial")
    return all_examples


def mine_ibis_examples() -> List[Dict[str, Any]]:
    """Mine examples from the Ibis examples repository.

    Returns
    -------
    list of dict
        Extracted examples
    """
    miner = GitHubMiner()

    # Clone Ibis examples repository
    try:
        repo_path = miner.clone_repo(
            "https://github.com/ibis-project/ibis-examples.git",
            "ibis-examples"
        )
    except Exception as e:
        print(f"Could not clone ibis-examples: {e}")
        return []

    all_examples = []

    # Search all Python files
    python_files = list(repo_path.rglob("*.py"))
    print(f"\nScanning {len(python_files)} Python files in ibis-examples...")

    for py_file in python_files:
        examples = miner.find_sql_conversions(py_file)
        if examples:
            print(f"  Found {len(examples)} in {py_file.name}")
        all_examples.extend(examples)

    print(f"\nFound {len(all_examples)} potential examples from ibis-examples")
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
    print("="*60)
    print("Mining SQL→Ibis examples from GitHub repositories")
    print("="*60)

    all_examples = []

    # Mine from main Ibis repository
    print("\n[1/3] Mining from ibis-project/ibis...")
    examples = mine_ibis_repo()
    all_examples.extend(examples)

    # Mine from Ibis tutorial repository
    print("\n[2/3] Mining from ibis-project/ibis-tutorial...")
    examples = mine_ibis_tutorial()
    all_examples.extend(examples)

    # Mine from Ibis examples repository
    print("\n[3/3] Mining from ibis-project/ibis-examples...")
    examples = mine_ibis_examples()
    all_examples.extend(examples)

    print("\n" + "="*60)
    print(f"TOTAL: Found {len(all_examples)} examples across all repositories")
    print("="*60)

    output_path = Path("data/mining/ibis_mined.jsonl")
    save_mined_examples(all_examples, output_path)
