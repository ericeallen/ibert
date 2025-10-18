"""Comprehensive test suite for dataset concatenation functionality.

This module tests the DatasetConcatenator and StatisticsPrinter classes,
ensuring correct behavior for:
- Dataset file discovery
- JSONL parsing and loading
- Data concatenation and provenance tracking
- Statistics computation
- Output formatting
"""

import json
import pytest
from pathlib import Path
from typing import List, Dict, Any
from unittest.mock import Mock, patch, mock_open

from src.datagen.concatenate_datasets import (
    DatasetConcatenator,
    StatisticsPrinter,
    SQL2IBIS_TRAINING_FILES,
    REPOS_DIRNAME,
)


class TestDatasetConcatenator:
    """Test suite for the DatasetConcatenator class."""

    @pytest.fixture
    def temp_data_dir(self, tmp_path: Path) -> Path:
        """Create a temporary data directory structure.

        Parameters
        ----------
        tmp_path : Path
            Pytest's temporary directory fixture

        Returns
        -------
        Path
            Temporary data directory
        """
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        return data_dir

    @pytest.fixture
    def concatenator(self, temp_data_dir: Path) -> DatasetConcatenator:
        """Create a DatasetConcatenator instance for testing.

        Parameters
        ----------
        temp_data_dir : Path
            Temporary data directory

        Returns
        -------
        DatasetConcatenator
            Configured concatenator instance
        """
        return DatasetConcatenator(temp_data_dir)

    @pytest.fixture
    def sample_examples(self) -> List[Dict[str, Any]]:
        """Create sample training examples.

        Returns
        -------
        list of dict
            Sample examples
        """
        return [
            {
                "task": "sql_to_ibis",
                "sql": "SELECT * FROM users",
                "ibis": "users.select()",
                "meta": {"source": "synthetic"}
            },
            {
                "task": "sql_to_ibis",
                "sql": "SELECT name FROM users WHERE age > 18",
                "ibis": "users.filter(users.age > 18).select('name')",
                "meta": {"source": "synthetic"}
            },
            {
                "source": "jupyter_notebook",
                "file": "example.ipynb",
                "sql": "SELECT COUNT(*) FROM events"
            }
        ]

    def test_initialization(self, temp_data_dir: Path):
        """Test DatasetConcatenator initialization."""
        concatenator = DatasetConcatenator(temp_data_dir)

        assert concatenator.data_dir == temp_data_dir
        assert concatenator.sql2ibis_dir == temp_data_dir / "sql2ibis"
        assert concatenator.mining_dir == temp_data_dir / "mining"

    def test_find_sql2ibis_files_empty(self, concatenator: DatasetConcatenator):
        """Test finding SQL2Ibis files when directory doesn't exist."""
        files = concatenator._find_sql2ibis_files()
        assert files == []

    def test_find_sql2ibis_files_with_data(
        self,
        concatenator: DatasetConcatenator,
        sample_examples: List[Dict[str, Any]]
    ):
        """Test finding SQL2Ibis files when they exist."""
        # Create sql2ibis directory
        concatenator.sql2ibis_dir.mkdir()

        # Create training files
        for filename in SQL2IBIS_TRAINING_FILES:
            file_path = concatenator.sql2ibis_dir / filename
            with open(file_path, "w") as f:
                f.write(json.dumps(sample_examples[0]) + "\n")

        files = concatenator._find_sql2ibis_files()

        assert len(files) == len(SQL2IBIS_TRAINING_FILES)
        assert all(f.exists() for f in files)
        assert all(f.name in SQL2IBIS_TRAINING_FILES for f in files)

    def test_find_mined_files_excludes_repos(
        self,
        concatenator: DatasetConcatenator
    ):
        """Test that mined files exclude test fixtures in repos directory."""
        # Create mining directory structure
        concatenator.mining_dir.mkdir()
        repos_dir = concatenator.mining_dir / REPOS_DIRNAME / "test_repo"
        repos_dir.mkdir(parents=True)

        # Create files
        valid_file = concatenator.mining_dir / "ibis_mined.jsonl"
        excluded_file = repos_dir / "test_fixture.jsonl"

        valid_file.write_text('{"test": "data"}\n')
        excluded_file.write_text('{"fixture": "data"}\n')

        files = concatenator._find_mined_files()

        assert len(files) == 1
        assert files[0] == valid_file
        assert excluded_file not in files

    def test_load_examples_from_file_valid_jsonl(
        self,
        concatenator: DatasetConcatenator,
        tmp_path: Path,
        sample_examples: List[Dict[str, Any]]
    ):
        """Test loading examples from valid JSONL file."""
        jsonl_file = tmp_path / "test.jsonl"

        with open(jsonl_file, "w") as f:
            for example in sample_examples:
                f.write(json.dumps(example) + "\n")

        loaded = concatenator.load_examples_from_file(jsonl_file)

        assert len(loaded) == len(sample_examples)
        assert all("source_file" in ex for ex in loaded)

    def test_load_examples_from_file_with_blank_lines(
        self,
        concatenator: DatasetConcatenator,
        tmp_path: Path
    ):
        """Test loading JSONL with blank lines."""
        jsonl_file = tmp_path / "test_blanks.jsonl"

        with open(jsonl_file, "w") as f:
            f.write('{"id": 1}\n')
            f.write('\n')
            f.write('{"id": 2}\n')
            f.write('   \n')
            f.write('{"id": 3}\n')

        loaded = concatenator.load_examples_from_file(jsonl_file)

        assert len(loaded) == 3
        assert loaded[0]["id"] == 1
        assert loaded[2]["id"] == 3

    def test_load_examples_from_file_malformed_json(
        self,
        concatenator: DatasetConcatenator,
        tmp_path: Path,
        capsys
    ):
        """Test graceful handling of malformed JSON."""
        jsonl_file = tmp_path / "test_malformed.jsonl"

        with open(jsonl_file, "w") as f:
            f.write('{"valid": 1}\n')
            f.write('{"invalid": \n')  # Malformed
            f.write('{"valid": 2}\n')

        loaded = concatenator.load_examples_from_file(jsonl_file)

        # Should load valid lines and skip malformed
        assert len(loaded) == 2

        # Check warning was printed
        captured = capsys.readouterr()
        assert "Warning: Invalid JSON" in captured.out

    def test_parse_json_line_valid(self, concatenator: DatasetConcatenator):
        """Test parsing valid JSON line."""
        line = '{"test": "data", "value": 123}'
        result = concatenator._parse_json_line(line, Path("test.jsonl"), 1)

        assert result is not None
        assert result["test"] == "data"
        assert result["value"] == 123

    def test_parse_json_line_invalid(
        self,
        concatenator: DatasetConcatenator,
        capsys
    ):
        """Test parsing invalid JSON line."""
        line = '{"invalid": '
        result = concatenator._parse_json_line(line, Path("test.jsonl"), 5)

        assert result is None

        captured = capsys.readouterr()
        assert "Warning: Invalid JSON" in captured.out
        assert "test.jsonl:5" in captured.out

    def test_add_source_metadata(self, concatenator: DatasetConcatenator):
        """Test adding source file metadata to examples."""
        example = {"sql": "SELECT 1"}
        file_path = Path("data/test.jsonl")

        concatenator._add_source_metadata(example, file_path)

        assert "source_file" in example
        assert "test.jsonl" in example["source_file"]

    def test_add_source_metadata_preserves_existing(
        self,
        concatenator: DatasetConcatenator
    ):
        """Test that existing source_file is not overwritten."""
        example = {"sql": "SELECT 1", "source_file": "original.jsonl"}
        file_path = Path("data/test.jsonl")

        concatenator._add_source_metadata(example, file_path)

        assert example["source_file"] == "original.jsonl"

    def test_get_relative_path_str(self, concatenator: DatasetConcatenator):
        """Test getting relative path string."""
        # Test with path that can be made relative
        path = Path.cwd() / "data" / "test.jsonl"
        result = concatenator._get_relative_path_str(path)

        assert "data/test.jsonl" in result or "data\\test.jsonl" in result

    def test_concatenate_no_files(
        self,
        concatenator: DatasetConcatenator,
        tmp_path: Path,
        capsys
    ):
        """Test concatenation when no files are found."""
        output_path = tmp_path / "output.jsonl"

        stats = concatenator.concatenate(output_path)

        assert stats == {}

        captured = capsys.readouterr()
        assert "No training data files found" in captured.out

    def test_concatenate_success(
        self,
        concatenator: DatasetConcatenator,
        sample_examples: List[Dict[str, Any]],
        tmp_path: Path
    ):
        """Test successful concatenation of multiple files."""
        # Create sql2ibis directory with files
        concatenator.sql2ibis_dir.mkdir()

        train_file = concatenator.sql2ibis_dir / "train.jsonl"
        with open(train_file, "w") as f:
            for ex in sample_examples[:2]:
                f.write(json.dumps(ex) + "\n")

        # Create mining directory with files
        concatenator.mining_dir.mkdir()

        mined_file = concatenator.mining_dir / "ibis_mined.jsonl"
        with open(mined_file, "w") as f:
            f.write(json.dumps(sample_examples[2]) + "\n")

        output_path = tmp_path / "complete.jsonl"
        stats = concatenator.concatenate(output_path)

        # Verify output file exists and has correct content
        assert output_path.exists()

        with open(output_path) as f:
            lines = f.readlines()
            assert len(lines) == 3

        # Verify statistics
        assert stats["total_examples"] == 3
        assert stats["source_files"] == 2
        assert "synthetic" in stats["by_source_type"]
        assert "jupyter_notebook" in stats["by_source_type"]

    def test_write_jsonl_creates_parent_dirs(
        self,
        concatenator: DatasetConcatenator,
        tmp_path: Path
    ):
        """Test that _write_jsonl creates parent directories."""
        output_path = tmp_path / "nested" / "dir" / "output.jsonl"
        examples = [{"test": "data"}]

        concatenator._write_jsonl(output_path, examples)

        assert output_path.exists()
        assert output_path.parent.exists()

    def test_compute_statistics(
        self,
        concatenator: DatasetConcatenator,
        sample_examples: List[Dict[str, Any]]
    ):
        """Test statistics computation."""
        file_breakdown = {"file1.jsonl": 2, "file2.jsonl": 1}
        source_files = [Path("file1.jsonl"), Path("file2.jsonl")]

        stats = concatenator._compute_statistics(
            sample_examples,
            source_files,
            file_breakdown
        )

        assert stats["total_examples"] == 3
        assert stats["source_files"] == 2
        assert stats["file_breakdown"] == file_breakdown
        assert "by_source_type" in stats
        assert "by_task" in stats

    def test_count_by_source_type(
        self,
        concatenator: DatasetConcatenator,
        sample_examples: List[Dict[str, Any]]
    ):
        """Test counting examples by source type."""
        counts = concatenator._count_by_source_type(sample_examples)

        assert counts["synthetic"] == 2
        assert counts["jupyter_notebook"] == 1

    def test_count_by_task(
        self,
        concatenator: DatasetConcatenator,
        sample_examples: List[Dict[str, Any]]
    ):
        """Test counting examples by task type."""
        counts = concatenator._count_by_task(sample_examples)

        assert counts["sql_to_ibis"] == 2
        assert counts["unknown"] == 1

    def test_extract_source_type_from_meta(
        self,
        concatenator: DatasetConcatenator
    ):
        """Test extracting source type from meta field."""
        example = {"meta": {"source": "synthetic"}}
        source = concatenator._extract_source_type(example)

        assert source == "synthetic"

    def test_extract_source_type_from_top_level(
        self,
        concatenator: DatasetConcatenator
    ):
        """Test extracting source type from top-level field."""
        example = {"source": "jupyter_notebook"}
        source = concatenator._extract_source_type(example)

        assert source == "jupyter_notebook"

    def test_extract_source_type_unknown(
        self,
        concatenator: DatasetConcatenator
    ):
        """Test extracting source type when not present."""
        example = {"task": "sql_to_ibis"}
        source = concatenator._extract_source_type(example)

        assert source == "unknown"


class TestStatisticsPrinter:
    """Test suite for the StatisticsPrinter class."""

    @pytest.fixture
    def sample_stats(self) -> Dict[str, Any]:
        """Create sample statistics for testing.

        Returns
        -------
        dict
            Sample statistics
        """
        return {
            "total_examples": 100,
            "source_files": 3,
            "file_breakdown": {
                "train.jsonl": 60,
                "train_augmented.jsonl": 30,
                "ibis_mined.jsonl": 10
            },
            "by_source_type": {
                "synthetic": 90,
                "jupyter_notebook": 10
            },
            "by_task": {
                "sql_to_ibis": 100
            }
        }

    def test_initialization(self, sample_stats: Dict[str, Any]):
        """Test StatisticsPrinter initialization."""
        printer = StatisticsPrinter(sample_stats)

        assert printer.stats == sample_stats
        assert printer.total_examples == 100

    def test_initialization_empty_stats(self):
        """Test initialization with empty stats."""
        printer = StatisticsPrinter({})

        assert printer.total_examples == 0

    def test_calculate_percentage(self, sample_stats: Dict[str, Any]):
        """Test percentage calculation."""
        printer = StatisticsPrinter(sample_stats)

        assert printer._calculate_percentage(50) == 50.0
        assert printer._calculate_percentage(25) == 25.0
        assert printer._calculate_percentage(100) == 100.0

    def test_calculate_percentage_zero_total(self):
        """Test percentage calculation with zero total."""
        printer = StatisticsPrinter({"total_examples": 0})

        assert printer._calculate_percentage(10) == 0.0

    def test_print_all(self, sample_stats: Dict[str, Any], capsys):
        """Test printing all statistics."""
        printer = StatisticsPrinter(sample_stats)
        printer.print_all()

        captured = capsys.readouterr()
        output = captured.out

        # Check all sections are present
        assert "Dataset Statistics" in output
        assert "Total examples: 100" in output
        assert "Source files: 3" in output
        assert "Examples per file:" in output
        assert "Examples by source type:" in output
        assert "Examples by task:" in output

    def test_print_summary(self, sample_stats: Dict[str, Any], capsys):
        """Test printing summary section."""
        printer = StatisticsPrinter(sample_stats)
        printer._print_summary()

        captured = capsys.readouterr()

        assert "Total examples: 100" in captured.out
        assert "Source files: 3" in captured.out

    def test_print_file_breakdown(self, sample_stats: Dict[str, Any], capsys):
        """Test printing file breakdown."""
        printer = StatisticsPrinter(sample_stats)
        printer._print_file_breakdown()

        captured = capsys.readouterr()
        output = captured.out

        assert "Examples per file:" in output
        assert "train.jsonl: 60 (60.0%)" in output
        assert "train_augmented.jsonl: 30 (30.0%)" in output
        assert "ibis_mined.jsonl: 10 (10.0%)" in output

    def test_print_source_type_breakdown(
        self,
        sample_stats: Dict[str, Any],
        capsys
    ):
        """Test printing source type breakdown."""
        printer = StatisticsPrinter(sample_stats)
        printer._print_source_type_breakdown()

        captured = capsys.readouterr()
        output = captured.out

        assert "Examples by source type:" in output
        assert "synthetic: 90 (90.0%)" in output
        assert "jupyter_notebook: 10 (10.0%)" in output

    def test_print_task_breakdown(self, sample_stats: Dict[str, Any], capsys):
        """Test printing task breakdown."""
        printer = StatisticsPrinter(sample_stats)
        printer._print_task_breakdown()

        captured = capsys.readouterr()
        output = captured.out

        assert "Examples by task:" in output
        assert "sql_to_ibis: 100 (100.0%)" in output

    def test_print_with_empty_sections(self, capsys):
        """Test printing when optional sections are empty."""
        stats = {
            "total_examples": 50,
            "source_files": 1,
            "file_breakdown": {}
        }

        printer = StatisticsPrinter(stats)
        printer.print_all()

        captured = capsys.readouterr()
        output = captured.out

        # Should still print header and summary
        assert "Dataset Statistics" in output
        assert "Total examples: 50" in output

        # Empty sections should not appear
        assert "Examples per file:" not in output


class TestIntegration:
    """Integration tests for the complete concatenation workflow."""

    def test_end_to_end_concatenation(self, tmp_path: Path):
        """Test complete end-to-end concatenation workflow."""
        # Setup data directory
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        sql2ibis_dir = data_dir / "sql2ibis"
        sql2ibis_dir.mkdir()

        mining_dir = data_dir / "mining"
        mining_dir.mkdir()

        # Create sample data files
        train_data = [
            {"task": "sql_to_ibis", "sql": "SELECT 1", "meta": {"source": "synthetic"}}
        ]

        mined_data = [
            {"source": "jupyter_notebook", "sql": "SELECT 2"}
        ]

        with open(sql2ibis_dir / "train.jsonl", "w") as f:
            for ex in train_data:
                f.write(json.dumps(ex) + "\n")

        with open(mining_dir / "ibis_mined.jsonl", "w") as f:
            for ex in mined_data:
                f.write(json.dumps(ex) + "\n")

        # Run concatenation
        concatenator = DatasetConcatenator(data_dir)
        output_path = data_dir / "complete.jsonl"
        stats = concatenator.concatenate(output_path)

        # Verify output
        assert output_path.exists()
        assert stats["total_examples"] == 2
        assert stats["source_files"] == 2

        # Verify statistics
        printer = StatisticsPrinter(stats)
        printer.print_all()

        # Verify output file content
        with open(output_path) as f:
            lines = [json.loads(line) for line in f]
            assert len(lines) == 2
            assert all("source_file" in line for line in lines)
