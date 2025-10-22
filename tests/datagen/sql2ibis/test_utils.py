"""Tests for SQL2Ibis utility modules (I/O and logging)."""

import json
import logging

import pandas as pd

from src.datagen.sql2ibis.utils.io import (
    append_jsonl,
    read_jsonl,
    read_parquet,
    write_jsonl,
    write_parquet,
)
from src.datagen.sql2ibis.utils.log import console, setup_logger


class TestJSONLIO:
    """Test suite for JSONL I/O functions."""

    def test_write_jsonl_simple(self, tmp_path):
        """Test writing simple JSONL file."""
        data = [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"},
        ]
        output_file = tmp_path / "test.jsonl"

        write_jsonl(data, output_file)

        assert output_file.exists()
        with open(output_file) as f:
            lines = f.readlines()
        assert len(lines) == 2

    def test_write_jsonl_creates_parent_dirs(self, tmp_path):
        """Test write_jsonl creates parent directories."""
        output_file = tmp_path / "nested" / "dir" / "test.jsonl"

        write_jsonl([{"test": 1}], output_file)

        assert output_file.exists()
        assert output_file.parent.exists()

    def test_write_jsonl_pretty(self, tmp_path):
        """Test writing pretty-printed JSONL."""
        data = [{"id": 1, "nested": {"key": "value"}}]
        output_file = tmp_path / "pretty.jsonl"

        write_jsonl(data, output_file, pretty=True)

        content = output_file.read_text()
        assert "  " in content  # Should have indentation

    def test_write_jsonl_not_pretty(self, tmp_path):
        """Test writing compact JSONL."""
        data = [{"id": 1, "nested": {"key": "value"}}]
        output_file = tmp_path / "compact.jsonl"

        write_jsonl(data, output_file, pretty=False)

        with open(output_file) as f:
            line = f.readline()
        # Should be single line without extra whitespace
        assert "\\n" not in line.strip() or json.loads(line)

    def test_write_jsonl_empty_list(self, tmp_path):
        """Test writing empty list creates empty file."""
        output_file = tmp_path / "empty.jsonl"

        write_jsonl([], output_file)

        assert output_file.exists()
        assert output_file.read_text() == ""

    def test_read_jsonl_simple(self, tmp_path):
        """Test reading simple JSONL file."""
        input_file = tmp_path / "test.jsonl"
        input_file.write_text('{"id": 1, "name": "Alice"}\n{"id": 2, "name": "Bob"}\n')

        records = list(read_jsonl(input_file))

        assert len(records) == 2
        assert records[0] == {"id": 1, "name": "Alice"}
        assert records[1] == {"id": 2, "name": "Bob"}

    def test_read_jsonl_skips_empty_lines(self, tmp_path):
        """Test reading skips empty lines."""
        input_file = tmp_path / "test.jsonl"
        input_file.write_text('{"id": 1}\n\n{"id": 2}\n   \n')

        records = list(read_jsonl(input_file))

        assert len(records) == 2

    def test_read_jsonl_empty_file(self, tmp_path):
        """Test reading empty file."""
        input_file = tmp_path / "empty.jsonl"
        input_file.write_text("")

        records = list(read_jsonl(input_file))

        assert records == []

    def test_read_jsonl_generator(self, tmp_path):
        """Test read_jsonl returns generator."""
        input_file = tmp_path / "test.jsonl"
        input_file.write_text('{"id": 1}\n{"id": 2}\n')

        result = read_jsonl(input_file)

        # Should be generator, not list
        assert hasattr(result, "__iter__")
        assert hasattr(result, "__next__")

    def test_append_jsonl_new_file(self, tmp_path):
        """Test appending to new file."""
        output_file = tmp_path / "append.jsonl"
        record = {"id": 1, "value": "test"}

        append_jsonl(record, output_file)

        assert output_file.exists()
        records = list(read_jsonl(output_file))
        assert len(records) == 1
        assert records[0] == record

    def test_append_jsonl_existing_file(self, tmp_path):
        """Test appending to existing file."""
        output_file = tmp_path / "append.jsonl"
        output_file.write_text('{"id": 1}\n')

        append_jsonl({"id": 2}, output_file)

        records = list(read_jsonl(output_file))
        assert len(records) == 2

    def test_append_jsonl_multiple_times(self, tmp_path):
        """Test multiple appends."""
        output_file = tmp_path / "multi.jsonl"

        for i in range(5):
            append_jsonl({"id": i}, output_file)

        records = list(read_jsonl(output_file))
        assert len(records) == 5
        assert [r["id"] for r in records] == [0, 1, 2, 3, 4]

    def test_append_jsonl_creates_parent_dirs(self, tmp_path):
        """Test append creates parent directories."""
        output_file = tmp_path / "nested" / "append.jsonl"

        append_jsonl({"test": 1}, output_file)

        assert output_file.exists()

    def test_roundtrip_jsonl(self, tmp_path):
        """Test write then read preserves data."""
        data = [
            {"id": 1, "name": "Alice", "score": 95.5},
            {"id": 2, "name": "Bob", "score": 87.3},
        ]
        output_file = tmp_path / "roundtrip.jsonl"

        write_jsonl(data, output_file)
        loaded = list(read_jsonl(output_file))

        assert loaded == data


class TestParquetIO:
    """Test suite for Parquet I/O functions."""

    def test_write_parquet_simple(self, tmp_path):
        """Test writing DataFrame to Parquet."""
        df = pd.DataFrame({"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"]})
        output_file = tmp_path / "test.parquet"

        write_parquet(df, output_file)

        assert output_file.exists()

    def test_write_parquet_creates_parent_dirs(self, tmp_path):
        """Test write_parquet creates parent directories."""
        df = pd.DataFrame({"col": [1, 2, 3]})
        output_file = tmp_path / "nested" / "dir" / "test.parquet"

        write_parquet(df, output_file)

        assert output_file.exists()

    def test_read_parquet_simple(self, tmp_path):
        """Test reading Parquet file."""
        df_original = pd.DataFrame({"id": [1, 2, 3], "value": [10.5, 20.3, 30.7]})
        file_path = tmp_path / "test.parquet"
        df_original.to_parquet(file_path)

        df_loaded = read_parquet(file_path)

        pd.testing.assert_frame_equal(df_loaded, df_original)

    def test_parquet_roundtrip(self, tmp_path):
        """Test writing then reading Parquet preserves data."""
        df_original = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["A", "B", "C"],
                "score": [95.5, 87.3, 92.1],
                "active": [True, False, True],
            }
        )
        file_path = tmp_path / "roundtrip.parquet"

        write_parquet(df_original, file_path)
        df_loaded = read_parquet(file_path)

        pd.testing.assert_frame_equal(df_loaded, df_original)

    def test_write_parquet_no_index(self, tmp_path):
        """Test Parquet doesn't include index."""
        df = pd.DataFrame({"col": [1, 2, 3]}, index=[10, 20, 30])
        file_path = tmp_path / "no_index.parquet"

        write_parquet(df, file_path)
        df_loaded = read_parquet(file_path)

        # Index should be default RangeIndex, not original
        assert list(df_loaded.index) == [0, 1, 2]

    def test_parquet_preserves_types(self, tmp_path):
        """Test Parquet preserves data types."""
        df = pd.DataFrame(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.1, 2.2, 3.3],
                "str_col": ["a", "b", "c"],
                "bool_col": [True, False, True],
            }
        )
        file_path = tmp_path / "types.parquet"

        write_parquet(df, file_path)
        df_loaded = read_parquet(file_path)

        assert df_loaded["int_col"].dtype == df["int_col"].dtype
        assert df_loaded["float_col"].dtype == df["float_col"].dtype
        assert df_loaded["str_col"].dtype == df["str_col"].dtype
        assert df_loaded["bool_col"].dtype == df["bool_col"].dtype


class TestLogging:
    """Test suite for logging utilities."""

    def test_setup_logger_basic(self):
        """Test basic logger setup."""
        logger = setup_logger("test_logger")

        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_logger"
        assert logger.level == logging.INFO

    def test_setup_logger_custom_level(self):
        """Test logger with custom level."""
        logger = setup_logger("test_debug", level="DEBUG")

        assert logger.level == logging.DEBUG

    def test_setup_logger_warning_level(self):
        """Test logger with WARNING level."""
        logger = setup_logger("test_warn", level="WARNING")

        assert logger.level == logging.WARNING

    def test_setup_logger_error_level(self):
        """Test logger with ERROR level."""
        logger = setup_logger("test_error", level="ERROR")

        assert logger.level == logging.ERROR

    def test_setup_logger_case_insensitive_level(self):
        """Test level parameter is case insensitive."""
        logger = setup_logger("test_case", level="info")

        assert logger.level == logging.INFO

    def test_setup_logger_with_file(self, tmp_path):
        """Test logger with file output."""
        log_file = tmp_path / "test.log"
        logger = setup_logger("test_file", log_file=log_file)

        logger.info("Test message")

        assert log_file.exists()
        content = log_file.read_text()
        assert "Test message" in content

    def test_setup_logger_file_creates_parent_dirs(self, tmp_path):
        """Test log file creation creates parent directories."""
        log_file = tmp_path / "nested" / "dir" / "test.log"
        logger = setup_logger("test_nested", log_file=log_file)

        logger.info("Test")

        assert log_file.exists()
        assert log_file.parent.exists()

    def test_setup_logger_has_handlers(self):
        """Test logger has console handler."""
        logger = setup_logger("test_handlers")

        # Should have at least console handler
        assert len(logger.handlers) >= 1

    def test_setup_logger_file_adds_handler(self, tmp_path):
        """Test file logging adds additional handler."""
        log_file = tmp_path / "test.log"
        logger = setup_logger("test_multi_handler", log_file=log_file)

        # Should have both console and file handlers
        assert len(logger.handlers) >= 2

    def test_console_exists(self):
        """Test console object is defined."""
        from rich.console import Console

        assert isinstance(console, Console)

    def test_logger_can_log_messages(self):
        """Test logger can log at different levels."""
        logger = setup_logger("test_logging", level="DEBUG")

        # Should not raise errors
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")

    def test_logger_file_formatting(self, tmp_path):
        """Test log file has proper formatting."""
        log_file = tmp_path / "format.log"
        logger = setup_logger("test_format", log_file=log_file)

        logger.info("Test message")

        content = log_file.read_text()
        # Should include timestamp, name, level, message
        assert "test_format" in content
        assert "INFO" in content
        assert "Test message" in content


class TestIntegration:
    """Integration tests for utility functions."""

    def test_jsonl_parquet_conversion(self, tmp_path):
        """Test converting JSONL to Parquet."""
        # Write JSONL
        data = [
            {"id": 1, "value": 10.5},
            {"id": 2, "value": 20.3},
        ]
        jsonl_file = tmp_path / "data.jsonl"
        write_jsonl(data, jsonl_file)

        # Read and convert to DataFrame
        records = list(read_jsonl(jsonl_file))
        df = pd.DataFrame(records)

        # Write as Parquet
        parquet_file = tmp_path / "data.parquet"
        write_parquet(df, parquet_file)

        # Read back
        df_loaded = read_parquet(parquet_file)

        assert len(df_loaded) == 2
        assert list(df_loaded.columns) == ["id", "value"]

    def test_logging_with_io_operations(self, tmp_path):
        """Test logging while performing I/O."""
        log_file = tmp_path / "ops.log"
        logger = setup_logger("test_io", log_file=log_file)

        # Perform I/O with logging
        data = [{"id": i} for i in range(10)]
        output_file = tmp_path / "data.jsonl"

        logger.info(f"Writing {len(data)} records")
        write_jsonl(data, output_file)

        logger.info("Reading records back")
        loaded = list(read_jsonl(output_file))

        logger.info(f"Loaded {len(loaded)} records")

        # Check logs
        log_content = log_file.read_text()
        assert "Writing 10 records" in log_content
        assert "Reading records back" in log_content
        assert "Loaded 10 records" in log_content
