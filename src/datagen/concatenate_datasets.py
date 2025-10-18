"""Concatenate training datasets from multiple sources into a unified dataset.

This module combines SQL→Ibis training examples from various sources:
- Template-generated synthetic data
- Augmented variations of templates
- Mined examples from GitHub repositories
- Extracted examples from documentation

The concatenated dataset maintains provenance metadata and provides
comprehensive statistics about data distribution.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict, Counter


# Constants for dataset organization
SQL2IBIS_TRAINING_FILES = ["train.jsonl", "train_augmented.jsonl"]
REPOS_DIRNAME = "repos"  # Test fixtures directory to exclude
HEADER_WIDTH = 70
PERCENTAGE_PRECISION = 1


class DatasetConcatenator:
    """Manages the concatenation of multiple training dataset files.

    This class handles discovery, loading, and merging of JSONL training
    files from various sources while maintaining data provenance and
    generating statistics.
    """

    def __init__(self, data_dir: Path):
        """Initialize the concatenator.

        Parameters
        ----------
        data_dir : Path
            Root directory containing training data subdirectories
        """
        self.data_dir = data_dir
        self.sql2ibis_dir = data_dir / "sql2ibis"
        self.mining_dir = data_dir / "mining"

    def find_training_files(self) -> List[Path]:
        """Discover all JSONL files containing training data.

        Searches for:
        - Template-generated files in sql2ibis/
        - Mined data files in mining/ (excluding test fixtures)

        Returns
        -------
        list of Path
            Sorted list of training data file paths
        """
        training_files = []

        # Collect template-generated training files
        training_files.extend(self._find_sql2ibis_files())

        # Collect mined training files (excluding test fixtures)
        training_files.extend(self._find_mined_files())

        return sorted(training_files)

    def _find_sql2ibis_files(self) -> List[Path]:
        """Find template-generated SQL→Ibis training files.

        Returns
        -------
        list of Path
            Paths to generated training files
        """
        files = []

        if self.sql2ibis_dir.exists():
            for filename in SQL2IBIS_TRAINING_FILES:
                file_path = self.sql2ibis_dir / filename
                if file_path.exists():
                    files.append(file_path)

        return files

    def _find_mined_files(self) -> List[Path]:
        """Find mined training data files, excluding test fixtures.

        Returns
        -------
        list of Path
            Paths to mined data files
        """
        files = []

        if self.mining_dir.exists():
            for file_path in self.mining_dir.glob("*.jsonl"):
                # Exclude test fixtures from cloned repositories
                if REPOS_DIRNAME not in file_path.parts:
                    files.append(file_path)

        return files

    def load_examples_from_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load and parse examples from a JSONL file.

        Adds source file metadata to each example for provenance tracking.
        Handles malformed JSON gracefully with warnings.

        Parameters
        ----------
        file_path : Path
            Path to JSONL file

        Returns
        -------
        list of dict
            Parsed examples with added metadata
        """
        examples = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line_number, line in enumerate(f, start=1):
                    line = line.strip()

                    if not line:  # Skip blank lines
                        continue

                    example = self._parse_json_line(
                        line, file_path, line_number
                    )

                    if example:
                        self._add_source_metadata(example, file_path)
                        examples.append(example)

        except IOError as e:
            print(f"Error reading {file_path}: {e}")

        return examples

    def _parse_json_line(
        self,
        line: str,
        file_path: Path,
        line_number: int
    ) -> Optional[Dict[str, Any]]:
        """Parse a single JSON line with error handling.

        Parameters
        ----------
        line : str
            JSON string to parse
        file_path : Path
            File being processed (for error messages)
        line_number : int
            Line number being processed (for error messages)

        Returns
        -------
        dict or None
            Parsed JSON object, or None if parsing failed
        """
        try:
            return json.loads(line)
        except json.JSONDecodeError as e:
            print(f"Warning: Invalid JSON at {file_path}:{line_number}: {e}")
            return None

    def _add_source_metadata(
        self,
        example: Dict[str, Any],
        file_path: Path
    ) -> None:
        """Add source file metadata to example for provenance tracking.

        Modifies the example dict in place.

        Parameters
        ----------
        example : dict
            Example to augment with metadata
        file_path : Path
            Source file path
        """
        if "source_file" not in example:
            example["source_file"] = self._get_relative_path_str(file_path)

    def _get_relative_path_str(self, path: Path) -> str:
        """Get path as string relative to current directory.

        Falls back to absolute path if relative path cannot be computed.

        Parameters
        ----------
        path : Path
            Path to make relative

        Returns
        -------
        str
            Relative path string, or absolute if relative fails
        """
        try:
            return str(path.relative_to(Path.cwd()))
        except ValueError:
            return str(path)

    def concatenate(self, output_path: Path) -> Dict[str, Any]:
        """Concatenate all training datasets into a single file.

        Parameters
        ----------
        output_path : Path
            Destination path for concatenated dataset

        Returns
        -------
        dict
            Statistics about the concatenation operation including:
            - total_examples: Total number of examples
            - source_files: Number of source files
            - file_breakdown: Examples per source file
            - by_source_type: Examples grouped by source type
            - by_task: Examples grouped by task type
        """
        self._print_header("Concatenating Training Datasets")

        # Discover training files
        training_files = self.find_training_files()

        if not training_files:
            print(f"No training data files found in {self.data_dir}")
            return {}

        self._print_discovered_files(training_files)

        # Load all examples
        all_examples, file_stats = self._load_all_examples(training_files)

        # Write concatenated output
        self._write_jsonl(output_path, all_examples)

        # Generate and return statistics
        stats = self._compute_statistics(
            all_examples, training_files, file_stats
        )

        self._print_completion_message(output_path, stats)

        return stats

    def _print_discovered_files(self, file_paths: List[Path]) -> None:
        """Print list of discovered training files.

        Parameters
        ----------
        file_paths : list of Path
            Discovered file paths
        """
        print(f"\nFound {len(file_paths)} training data files:")
        for file_path in file_paths:
            relative_path = self._get_relative_path_str(file_path)
            print(f"  - {relative_path}")

    def _load_all_examples(
        self,
        file_paths: List[Path]
    ) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
        """Load examples from all files and track per-file statistics.

        Parameters
        ----------
        file_paths : list of Path
            Paths to load from

        Returns
        -------
        tuple of (list of dict, dict)
            All examples and per-file counts
        """
        all_examples = []
        file_stats = {}

        print("\nLoading data...")

        for file_path in file_paths:
            examples = self.load_examples_from_file(file_path)
            all_examples.extend(examples)

            relative_key = str(file_path.relative_to(self.data_dir))
            file_stats[relative_key] = len(examples)

            print(f"  {file_path.name}: {len(examples)} examples")

        return all_examples, file_stats

    def _write_jsonl(
        self,
        output_path: Path,
        examples: List[Dict[str, Any]]
    ) -> None:
        """Write examples to JSONL file.

        Parameters
        ----------
        output_path : Path
            Destination file path
        examples : list of dict
            Examples to write
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for example in examples:
                json_line = json.dumps(example, ensure_ascii=False)
                f.write(json_line + "\n")

    def _compute_statistics(
        self,
        examples: List[Dict[str, Any]],
        source_files: List[Path],
        file_breakdown: Dict[str, int]
    ) -> Dict[str, Any]:
        """Compute comprehensive statistics about the dataset.

        Parameters
        ----------
        examples : list of dict
            All loaded examples
        source_files : list of Path
            Source file paths
        file_breakdown : dict
            Per-file example counts

        Returns
        -------
        dict
            Statistics dictionary
        """
        return {
            "total_examples": len(examples),
            "source_files": len(source_files),
            "file_breakdown": file_breakdown,
            "by_source_type": self._count_by_source_type(examples),
            "by_task": self._count_by_task(examples),
        }

    def _count_by_source_type(
        self,
        examples: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """Count examples by source type.

        Checks both 'meta.source' and top-level 'source' fields.

        Parameters
        ----------
        examples : list of dict
            Examples to analyze

        Returns
        -------
        dict
            Source type counts
        """
        source_types = (
            self._extract_source_type(ex) for ex in examples
        )
        return dict(Counter(source_types))

    def _extract_source_type(self, example: Dict[str, Any]) -> str:
        """Extract source type from example metadata.

        Parameters
        ----------
        example : dict
            Example to analyze

        Returns
        -------
        str
            Source type identifier
        """
        # Check nested metadata first, then top-level
        meta_source = example.get("meta", {}).get("source")
        return meta_source or example.get("source", "unknown")

    def _count_by_task(
        self,
        examples: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """Count examples by task type.

        Parameters
        ----------
        examples : list of dict
            Examples to analyze

        Returns
        -------
        dict
            Task type counts
        """
        tasks = (ex.get("task", "unknown") for ex in examples)
        return dict(Counter(tasks))

    def _print_header(self, title: str) -> None:
        """Print a formatted section header.

        Parameters
        ----------
        title : str
            Header title
        """
        print("=" * HEADER_WIDTH)
        print(title)
        print("=" * HEADER_WIDTH)

    def _print_completion_message(
        self,
        output_path: Path,
        stats: Dict[str, Any]
    ) -> None:
        """Print completion message with summary.

        Parameters
        ----------
        output_path : Path
            Output file path
        stats : dict
            Concatenation statistics
        """
        print("\n" + "=" * HEADER_WIDTH)
        print(f"Concatenated {stats['total_examples']} examples to:")
        relative_output = self._get_relative_path_str(output_path)
        print(f"  {relative_output}")
        print("=" * HEADER_WIDTH)


class StatisticsPrinter:
    """Formats and prints dataset statistics in human-readable form."""

    def __init__(self, stats: Dict[str, Any]):
        """Initialize printer with statistics.

        Parameters
        ----------
        stats : dict
            Statistics dictionary from concatenation
        """
        self.stats = stats
        self.total_examples = stats.get("total_examples", 0)

    def print_all(self) -> None:
        """Print comprehensive statistics report."""
        self._print_header()
        self._print_summary()
        self._print_file_breakdown()
        self._print_source_type_breakdown()
        self._print_task_breakdown()

    def _print_header(self) -> None:
        """Print statistics header."""
        print("\n" + "=" * HEADER_WIDTH)
        print("Dataset Statistics")
        print("=" * HEADER_WIDTH)

    def _print_summary(self) -> None:
        """Print high-level summary."""
        print(f"\nTotal examples: {self.total_examples}")
        print(f"Source files: {self.stats.get('source_files', 0)}")

    def _print_file_breakdown(self) -> None:
        """Print per-file example counts."""
        file_breakdown = self.stats.get("file_breakdown", {})
        if not file_breakdown:
            return

        print("\nExamples per file:")
        for filename, count in sorted(file_breakdown.items()):
            percentage = self._calculate_percentage(count)
            print(f"  {filename}: {count} ({percentage:.{PERCENTAGE_PRECISION}f}%)")

    def _print_source_type_breakdown(self) -> None:
        """Print examples grouped by source type."""
        by_source = self.stats.get("by_source_type", {})
        if not by_source:
            return

        print("\nExamples by source type:")
        # Sort by count descending
        sorted_sources = sorted(by_source.items(), key=lambda x: -x[1])

        for source_type, count in sorted_sources:
            percentage = self._calculate_percentage(count)
            print(f"  {source_type}: {count} ({percentage:.{PERCENTAGE_PRECISION}f}%)")

    def _print_task_breakdown(self) -> None:
        """Print examples grouped by task type."""
        by_task = self.stats.get("by_task", {})
        if not by_task:
            return

        print("\nExamples by task:")
        # Sort by count descending
        sorted_tasks = sorted(by_task.items(), key=lambda x: -x[1])

        for task, count in sorted_tasks:
            percentage = self._calculate_percentage(count)
            print(f"  {task}: {count} ({percentage:.{PERCENTAGE_PRECISION}f}%)")

    def _calculate_percentage(self, count: int) -> float:
        """Calculate percentage of total examples.

        Parameters
        ----------
        count : int
            Number of examples

        Returns
        -------
        float
            Percentage (0-100)
        """
        if self.total_examples == 0:
            return 0.0
        return (count / self.total_examples) * 100


def main() -> None:
    """Main entry point for dataset concatenation."""
    data_dir = Path("data")
    output_path = data_dir / "train_complete.jsonl"

    concatenator = DatasetConcatenator(data_dir)
    stats = concatenator.concatenate(output_path)

    if stats:
        printer = StatisticsPrinter(stats)
        printer.print_all()


if __name__ == "__main__":
    main()
