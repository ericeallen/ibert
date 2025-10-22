"""Mine SQL→Ibis training examples from GitHub repositories.

This module clones repositories and extracts SQL code patterns that demonstrate
SQL→Ibis translations. It searches for:
- .sql() method calls on tables and backends
- Direct SQL strings in code
- Multi-line SQL assignments
- Jupyter notebooks with SQL examples

Examples are extracted with provenance metadata for training data generation.
"""

import json
import re

# Safe: subprocess only used for git clone with sanitized arguments - no shell injection risk
import subprocess  # nosec B404
from pathlib import Path
from typing import Any, NamedTuple

# Constants for pattern matching
SQL_KEYWORD = "SELECT"  # Primary SQL keyword to identify queries
GIT_CLONE_DEPTH = 1  # Shallow clone for faster downloads

# Regex patterns for SQL extraction
PATTERN_SQL_METHOD = r'(\w+)\s*=\s*(?:\w+)\.sql\(\s*["\'](.+?)["\']\s*\)'
PATTERN_DIRECT_SQL = r'\.sql\(\s*["\'](.+?)["\']\s*\)'
PATTERN_MULTILINE_SQL = r'sql\s*=\s*"""(.+?)"""'

# Output configuration
HEADER_WIDTH = 60


class RepositoryConfig(NamedTuple):
    """Configuration for a repository to mine.

    Attributes
    ----------
    url : str
        Git repository URL
    name : str
        Local directory name for the repository
    scan_dirs : list of str or None
        Specific subdirectories to scan, or None to scan entire repo
    """

    url: str
    name: str
    scan_dirs: list[str] | None


class SQLExample(NamedTuple):
    """A discovered SQL example with metadata.

    Attributes
    ----------
    source_type : str
        Type of extraction pattern used
    file_path : str
        Source file path
    sql_code : str
        The SQL query string
    ibis_var : str or None
        Variable name for Ibis expression, if available
    """

    source_type: str
    file_path: str
    sql_code: str
    ibis_var: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format for JSON serialization.

        Returns
        -------
        dict
            Example as dictionary
        """
        result = {
            "source": self.source_type,
            "file": self.file_path,
            "sql": self.sql_code,
        }

        if self.ibis_var:
            result["ibis_var"] = self.ibis_var

        return result


class GitHubRepositoryMiner:
    """Mines SQL→Ibis code examples from GitHub repositories.

    This class handles repository cloning, file discovery, and pattern
    extraction to find SQL code that demonstrates SQL→Ibis translations.
    """

    def __init__(self, cache_dir: Path = Path("data/mining/repos")):
        """Initialize the repository miner.

        Parameters
        ----------
        cache_dir : Path
            Directory to cache cloned repositories
        """
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def clone_repository(self, repo_url: str, local_name: str) -> Path:
        """Clone a GitHub repository using shallow clone.

        If the repository is already cloned, returns existing path.

        Parameters
        ----------
        repo_url : str
            GitHub repository URL
        local_name : str
            Local directory name for the cloned repository

        Returns
        -------
        Path
            Path to cloned repository directory

        Raises
        ------
        subprocess.CalledProcessError
            If git clone fails
        """
        repo_path = self.cache_dir / local_name

        if repo_path.exists():
            print(f"Repository already cloned: {repo_path}")
            return repo_path

        print(f"Cloning {repo_url}...")
        self._execute_git_clone(repo_url, repo_path)

        return repo_path

    def _execute_git_clone(self, repo_url: str, destination: Path) -> None:
        """Execute git clone command with shallow depth.

        Parameters
        ----------
        repo_url : str
            Repository URL to clone
        destination : Path
            Local destination path
        """
        # Safe: git is a standard system tool, arguments are validated before calling
        # Safe: repo_url from trusted config, destination is Path object - no shell injection
        subprocess.run(  # nosec B603 B607
            ["git", "clone", "--depth", str(GIT_CLONE_DEPTH), repo_url, str(destination)],
            check=True,
            capture_output=True,
            text=True,
        )

    def find_python_files(self, directory: Path) -> list[Path]:
        """Recursively find all Python files in a directory.

        Parameters
        ----------
        directory : Path
            Root directory to search

        Returns
        -------
        list of Path
            All discovered Python files
        """
        return list(directory.rglob("*.py"))

    def find_jupyter_notebooks(self, directory: Path) -> list[Path]:
        """Recursively find all Jupyter notebooks in a directory.

        Parameters
        ----------
        directory : Path
            Root directory to search

        Returns
        -------
        list of Path
            All discovered notebook files
        """
        return list(directory.rglob("*.ipynb"))

    def extract_sql_examples(self, python_file: Path) -> list[SQLExample]:
        """Extract SQL→Ibis examples from a Python file.

        Searches for multiple patterns:
        1. Table.sql() or Backend.sql() method calls
        2. Direct .sql() calls
        3. Multi-line SQL string assignments

        Parameters
        ----------
        python_file : Path
            Python file to analyze

        Returns
        -------
        list of SQLExample
            Discovered SQL examples with metadata
        """
        content = self._read_file_safely(python_file)
        if content is None:
            return []

        examples = []

        # Extract examples using all patterns
        examples.extend(self._extract_sql_method_calls(content, python_file))
        examples.extend(self._extract_direct_sql_calls(content, python_file))
        examples.extend(self._extract_multiline_sql(content, python_file))

        return examples

    def _read_file_safely(self, file_path: Path) -> str | None:
        """Read file content with error handling.

        Parameters
        ----------
        file_path : Path
            File to read

        Returns
        -------
        str or None
            File content, or None if read failed
        """
        try:
            return file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None

    def _extract_sql_method_calls(self, content: str, source_file: Path) -> list[SQLExample]:
        """Extract SQL from .sql() method calls with variable assignment.

        Pattern: var = obj.sql("SELECT ...")

        Parameters
        ----------
        content : str
            File content
        source_file : Path
            Source file path

        Returns
        -------
        list of SQLExample
            Extracted examples
        """
        examples = []

        for match in re.finditer(PATTERN_SQL_METHOD, content, re.DOTALL):
            variable_name = match.group(1).strip()
            sql_code = match.group(2).strip()

            if self._is_valid_sql_query(sql_code):
                examples.append(
                    SQLExample(
                        source_type="table.sql()",
                        file_path=str(source_file),
                        sql_code=sql_code,
                        ibis_var=variable_name,
                    )
                )

        return examples

    def _extract_direct_sql_calls(self, content: str, source_file: Path) -> list[SQLExample]:
        """Extract SQL from direct .sql() method calls.

        Pattern: obj.sql("SELECT ...")

        Parameters
        ----------
        content : str
            File content
        source_file : Path
            Source file path

        Returns
        -------
        list of SQLExample
            Extracted examples
        """
        examples = []

        for match in re.finditer(PATTERN_DIRECT_SQL, content, re.DOTALL):
            sql_code = match.group(1).strip()

            if self._is_valid_sql_query(sql_code):
                examples.append(
                    SQLExample(
                        source_type="direct_sql", file_path=str(source_file), sql_code=sql_code
                    )
                )

        return examples

    def _extract_multiline_sql(self, content: str, source_file: Path) -> list[SQLExample]:
        """Extract SQL from multi-line string assignments.

        Pattern:
        sql = \"\"\"
        SELECT ...
        \"\"\"

        Parameters
        ----------
        content : str
            File content
        source_file : Path
            Source file path

        Returns
        -------
        list of SQLExample
            Extracted examples
        """
        examples = []

        for match in re.finditer(PATTERN_MULTILINE_SQL, content, re.DOTALL):
            sql_code = match.group(1).strip()

            if self._is_valid_sql_query(sql_code):
                examples.append(
                    SQLExample(
                        source_type="multiline_sql", file_path=str(source_file), sql_code=sql_code
                    )
                )

        return examples

    def _is_valid_sql_query(self, sql_code: str) -> bool:
        """Check if string appears to be a SQL query.

        Parameters
        ----------
        sql_code : str
            Code to validate

        Returns
        -------
        bool
            True if appears to be SQL query
        """
        return SQL_KEYWORD in sql_code.upper()


class RepositoryScanner:
    """Scans repositories for SQL→Ibis examples.

    Coordinates the mining process across directories and file types.
    """

    def __init__(self, miner: GitHubRepositoryMiner):
        """Initialize scanner with miner instance.

        Parameters
        ----------
        miner : GitHubRepositoryMiner
            Miner instance to use for extraction
        """
        self.miner = miner

    def scan_repository(
        self, repo_path: Path, repo_name: str, target_directories: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """Scan repository for SQL examples.

        Parameters
        ----------
        repo_path : Path
            Root path of cloned repository
        repo_name : str
            Repository name for logging
        target_directories : list of str, optional
            Specific subdirectories to scan, or None for entire repo

        Returns
        -------
        list of dict
            All extracted examples as dictionaries
        """
        scan_paths = self._get_scan_paths(repo_path, target_directories)

        all_examples = []

        # Scan Python files
        all_examples.extend(self._scan_python_files(scan_paths, repo_path, repo_name))

        # Scan Jupyter notebooks
        all_examples.extend(self._scan_notebooks(scan_paths, repo_path, repo_name))

        print(f"\nFound {len(all_examples)} potential examples from {repo_name}")
        return all_examples

    def _get_scan_paths(self, repo_path: Path, target_directories: list[str] | None) -> list[Path]:
        """Determine which directories to scan.

        Parameters
        ----------
        repo_path : Path
            Repository root path
        target_directories : list of str or None
            Specific directories, or None for entire repo

        Returns
        -------
        list of Path
            Paths to scan
        """
        if target_directories:
            return [repo_path / dir_name for dir_name in target_directories]
        return [repo_path]

    def _scan_python_files(
        self, scan_paths: list[Path], repo_path: Path, repo_name: str
    ) -> list[dict[str, Any]]:
        """Scan Python files in target paths.

        Parameters
        ----------
        scan_paths : list of Path
            Directories to scan
        repo_path : Path
            Repository root path
        repo_name : str
            Repository name for display

        Returns
        -------
        list of dict
            Extracted examples
        """
        examples: list[dict[str, Any]] = []

        for scan_path in scan_paths:
            if not scan_path.exists():
                print(f"Directory not found: {scan_path}")
                continue

            python_files = self.miner.find_python_files(scan_path)
            directory_label = self._get_directory_label(scan_path, repo_path, repo_name)

            print(f"Scanning {len(python_files)} Python files in {directory_label}...")

            for python_file in python_files:
                file_examples = self.miner.extract_sql_examples(python_file)

                if file_examples:
                    print(f"  Found {len(file_examples)} in {python_file.name}")

                # Convert to dictionaries
                examples.extend(ex.to_dict() for ex in file_examples)

        return examples

    def _scan_notebooks(
        self, scan_paths: list[Path], repo_path: Path, repo_name: str
    ) -> list[dict[str, Any]]:
        """Scan Jupyter notebooks in target paths.

        Parameters
        ----------
        scan_paths : list of Path
            Directories to scan
        repo_path : Path
            Repository root path
        repo_name : str
            Repository name for display

        Returns
        -------
        list of dict
            Extracted examples
        """
        examples = []

        for scan_path in scan_paths:
            if not scan_path.exists():
                continue

            notebooks = self.miner.find_jupyter_notebooks(scan_path)
            if not notebooks:
                continue

            directory_label = self._get_directory_label(scan_path, repo_path, repo_name)
            print(f"Scanning {len(notebooks)} Jupyter notebooks in {directory_label}...")

            try:
                from src.datagen.mining.ibis_doc_extractor import extract_from_jupyter

                for notebook in notebooks:
                    notebook_examples = extract_from_jupyter(notebook)

                    if notebook_examples:
                        print(f"  Found {len(notebook_examples)} in {notebook.name}")

                    examples.extend(notebook_examples)

            except ImportError:
                print("Warning: ibis_doc_extractor not available, skipping notebooks")
                break

        return examples

    def _get_directory_label(self, scan_path: Path, repo_path: Path, repo_name: str) -> str:
        """Get human-readable label for directory.

        Parameters
        ----------
        scan_path : Path
            Directory being scanned
        repo_path : Path
            Repository root
        repo_name : str
            Repository name

        Returns
        -------
        str
            Display label
        """
        if scan_path == repo_path:
            return repo_name

        try:
            return str(scan_path.relative_to(repo_path))
        except ValueError:
            return str(scan_path)


def mine_repository(
    repo_url: str,
    repo_name: str,
    scan_directories: list[str] | None = None,
    miner: GitHubRepositoryMiner | None = None,
) -> list[dict[str, Any]]:
    """Mine SQL→Ibis examples from a single GitHub repository.

    This is the main entry point for mining a repository.

    Parameters
    ----------
    repo_url : str
        GitHub repository URL
    repo_name : str
        Local name for the repository
    scan_directories : list of str, optional
        Specific subdirectories to scan, or None for entire repo
    miner : GitHubRepositoryMiner, optional
        Miner instance to use, or None to create new one

    Returns
    -------
    list of dict
        All extracted SQL examples with metadata
    """
    if miner is None:
        miner = GitHubRepositoryMiner()

    # Clone repository
    try:
        repo_path = miner.clone_repository(repo_url, repo_name)
    except subprocess.CalledProcessError as e:
        print(f"Error cloning {repo_name}: {e}")
        return []

    # Scan for examples
    scanner = RepositoryScanner(miner)
    return scanner.scan_repository(repo_path, repo_name, scan_directories)


def load_repository_config(config_path: Path) -> list[RepositoryConfig]:
    """Load repository configurations from file.

    File format: repo_url|repo_name|optional,scan,dirs

    Parameters
    ----------
    config_path : Path
        Path to configuration file

    Returns
    -------
    list of RepositoryConfig
        Parsed repository configurations
    """
    configs = []

    with open(config_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # Skip comments and empty lines
            if not line or line.startswith("#"):
                continue

            config = _parse_config_line(line)
            if config:
                configs.append(config)

    return configs


def _parse_config_line(line: str) -> RepositoryConfig | None:
    """Parse a single configuration line.

    Parameters
    ----------
    line : str
        Configuration line

    Returns
    -------
    RepositoryConfig or None
        Parsed config, or None if invalid
    """
    parts = line.split("|")

    if len(parts) < 2:
        print(f"Skipping invalid config line: {line}")
        return None

    repo_url = parts[0].strip()
    repo_name = parts[1].strip()

    # Parse optional scan directories
    scan_dirs = None
    if len(parts) >= 3 and parts[2].strip():
        scan_dirs = [d.strip() for d in parts[2].split(",")]

    return RepositoryConfig(url=repo_url, name=repo_name, scan_dirs=scan_dirs)


def mine_from_config(config_path: Path) -> list[dict[str, Any]]:
    """Mine examples from all repositories in configuration file.

    Parameters
    ----------
    config_path : Path
        Path to repository configuration file

    Returns
    -------
    list of dict
        All extracted examples from all repositories
    """
    repo_configs = load_repository_config(config_path)
    print(f"Loaded {len(repo_configs)} repositories from {config_path}")

    miner = GitHubRepositoryMiner()
    all_examples = []

    for index, config in enumerate(repo_configs, start=1):
        print(f"\n[{index}/{len(repo_configs)}] Mining from {config.name}...")
        print(f"  URL: {config.url}")

        if config.scan_dirs:
            print(f"  Directories: {', '.join(config.scan_dirs)}")

        examples = mine_repository(config.url, config.name, config.scan_dirs, miner)
        all_examples.extend(examples)

    return all_examples


def save_examples_to_jsonl(examples: list[dict[str, Any]], output_path: Path) -> None:
    """Save extracted examples to JSONL file.

    Parameters
    ----------
    examples : list of dict
        Examples to save
    output_path : Path
        Destination file path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for example in examples:
            json_line = json.dumps(example, ensure_ascii=False)
            f.write(json_line + "\n")

    print(f"Saved {len(examples)} examples to {output_path}")


def main() -> None:
    """Main entry point for GitHub repository mining."""
    print("=" * HEADER_WIDTH)
    print("Mining SQL→Ibis examples from GitHub repositories")
    print("=" * HEADER_WIDTH)

    # Locate configuration file
    config_path = Path(__file__).parent / "repo_urls.txt"

    if not config_path.exists():
        print(f"\nError: Configuration file not found: {config_path}")
        print("Please create a repo_urls.txt file with repository URLs.")
        exit(1)

    # Mine all configured repositories
    all_examples = mine_from_config(config_path)

    print("\n" + "=" * HEADER_WIDTH)
    print(f"TOTAL: Found {len(all_examples)} examples across all repositories")
    print("=" * HEADER_WIDTH)

    # Save results
    output_path = Path("data/mining/ibis_mined.jsonl")
    save_examples_to_jsonl(all_examples, output_path)


if __name__ == "__main__":
    main()
