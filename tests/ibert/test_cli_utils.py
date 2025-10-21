"""Tests for CLI utility functions."""

import io
import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

# Import the module to ensure it's loaded for coverage
import src.ibert.cli_utils
from src.ibert.cli_utils import read_input


class TestReadInput:
    """Test suite for read_input function."""

    def test_read_input_from_file(self, tmp_path):
        """Test reading input from file."""
        input_file = tmp_path / "input.txt"
        input_file.write_text("test input content")

        result = read_input(str(input_file), "test-script")

        assert result == "test input content"

    def test_read_input_from_file_with_newlines(self, tmp_path):
        """Test reading multiline content from file."""
        input_file = tmp_path / "input.txt"
        input_file.write_text("line 1\nline 2\nline 3")

        result = read_input(str(input_file))

        assert "line 1" in result
        assert "line 2" in result
        assert "line 3" in result

    def test_read_input_file_not_found(self, capsys):
        """Test reading from non-existent file exits with error."""
        with pytest.raises(SystemExit) as exc_info:
            read_input("/nonexistent/file.txt", "test-script")

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "File not found" in captured.err

    def test_read_input_file_read_error(self, tmp_path, capsys):
        """Test handling file read errors."""
        # Create a directory instead of file to trigger read error
        input_dir = tmp_path / "not_a_file"
        input_dir.mkdir()

        with pytest.raises(SystemExit) as exc_info:
            read_input(str(input_dir), "test-script")

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Error reading file" in captured.err

    @patch('sys.stdin')
    def test_read_input_from_stdin_with_select(self, mock_stdin):
        """Test reading from stdin when data is available."""
        mock_stdin.read.return_value = "stdin content"
        mock_stdin.isatty.return_value = False

        with patch('select.select') as mock_select:
            # Simulate data available on stdin
            mock_select.return_value = ([mock_stdin], [], [])

            result = read_input(None, "test-script")

        assert result == "stdin content"

    @patch('sys.stdin')
    def test_read_input_from_stdin_no_select(self, mock_stdin):
        """Test reading from stdin when select() fails."""
        mock_stdin.read.return_value = "stdin content"
        mock_stdin.isatty.return_value = False

        with patch('select.select', side_effect=OSError("select not available")):
            result = read_input(None, "test-script")

        assert result == "stdin content"

    @patch('sys.stdin')
    def test_read_input_stdin_value_error(self, mock_stdin):
        """Test handling ValueError from select."""
        mock_stdin.read.return_value = "stdin content"
        mock_stdin.isatty.return_value = False

        with patch('select.select', side_effect=ValueError("bad fd")):
            result = read_input(None, "test-script")

        assert result == "stdin content"

    @patch('sys.stdin')
    def test_read_input_tty_no_file_exits(self, mock_stdin, capsys):
        """Test interactive terminal with no file exits with usage."""
        mock_stdin.isatty.return_value = True

        with pytest.raises(SystemExit) as exc_info:
            read_input(None, "my-script")

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "No input provided" in captured.err
        assert "Usage:" in captured.err
        assert "my-script" in captured.err

    @patch('sys.stdin')
    def test_read_input_empty_stdin_exits(self, mock_stdin, capsys):
        """Test empty stdin exits with error."""
        mock_stdin.read.return_value = ""
        mock_stdin.isatty.return_value = False

        with patch('select.select') as mock_select:
            mock_select.return_value = ([mock_stdin], [], [])

            with pytest.raises(SystemExit) as exc_info:
                read_input(None, "test-script")

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "No input provided" in captured.err

    @patch('sys.stdin')
    def test_read_input_whitespace_only_stdin_exits(self, mock_stdin, capsys):
        """Test whitespace-only stdin exits with error."""
        mock_stdin.read.return_value = "   \n\t  \n  "
        mock_stdin.isatty.return_value = False

        with patch('select.select') as mock_select:
            mock_select.return_value = ([mock_stdin], [], [])

            with pytest.raises(SystemExit) as exc_info:
                read_input(None, "test-script")

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "No input provided" in captured.err

    def test_read_input_empty_file_exits(self, tmp_path, capsys):
        """Test reading empty file exits with error."""
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("")

        with pytest.raises(SystemExit) as exc_info:
            read_input(str(empty_file), "test-script")

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "No input provided" in captured.err

    def test_read_input_whitespace_only_file_exits(self, tmp_path, capsys):
        """Test reading whitespace-only file exits with error."""
        ws_file = tmp_path / "whitespace.txt"
        ws_file.write_text("   \n\t  \n  ")

        with pytest.raises(SystemExit) as exc_info:
            read_input(str(ws_file), "test-script")

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "No input provided" in captured.err

    @patch('sys.stdin')
    def test_read_input_no_data_ready_on_select(self, mock_stdin, capsys):
        """Test when select shows no data ready on stdin."""
        mock_stdin.isatty.return_value = False

        with patch('select.select') as mock_select:
            # No data ready
            mock_select.return_value = ([], [], [])

            with pytest.raises(SystemExit) as exc_info:
                read_input(None, "test-script")

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "No input provided" in captured.err

    def test_read_input_usage_message_format(self, tmp_path, capsys):
        """Test usage message contains all expected examples."""
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("")

        with pytest.raises(SystemExit):
            read_input(str(empty_file), "my-tool")

        captured = capsys.readouterr()
        # Check for all three usage examples
        assert "echo 'your input' | my-tool" in captured.err
        assert "my-tool input.txt" in captured.err
        assert "cat input.txt | my-tool" in captured.err

    def test_read_input_preserves_content(self, tmp_path):
        """Test that input content is preserved exactly."""
        content = "Special chars: !@#$%^&*()\nTabs:\t\tHere\nSpaces:    "
        input_file = tmp_path / "special.txt"
        input_file.write_text(content)

        result = read_input(str(input_file))

        assert result == content

    def test_read_input_large_file(self, tmp_path):
        """Test reading large file works correctly."""
        # Create file with substantial content
        large_content = "Line {}\n" * 10000
        large_content = large_content.format(*range(10000))
        input_file = tmp_path / "large.txt"
        input_file.write_text(large_content)

        result = read_input(str(input_file))

        assert len(result) > 50000
        assert "Line 0" in result
        assert "Line 9999" in result

    def test_read_input_default_script_name(self, tmp_path, capsys):
        """Test default script name is used when not provided."""
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("")

        with pytest.raises(SystemExit):
            read_input(str(empty_file))  # No script_name provided

        captured = capsys.readouterr()
        # Should use default "script" name
        assert "script" in captured.err

    @patch('sys.stdin')
    def test_read_input_stdin_with_content(self, mock_stdin):
        """Test reading actual content from stdin."""
        test_content = "SELECT * FROM table WHERE id = 1"
        mock_stdin.read.return_value = test_content
        mock_stdin.isatty.return_value = False

        with patch('select.select') as mock_select:
            mock_select.return_value = ([mock_stdin], [], [])

            result = read_input(None, "sql-tool")

        assert result == test_content

    def test_read_input_file_with_unicode(self, tmp_path):
        """Test reading file with Unicode characters."""
        unicode_content = "Hello 世界 🌍 Привет"
        input_file = tmp_path / "unicode.txt"
        input_file.write_text(unicode_content, encoding='utf-8')

        result = read_input(str(input_file))

        assert result == unicode_content
        assert "世界" in result
        assert "🌍" in result

    @patch('sys.stdin')
    def test_read_input_select_timeout(self, mock_stdin):
        """Test that select is called with timeout."""
        mock_stdin.read.return_value = "content"
        mock_stdin.isatty.return_value = False

        with patch('select.select') as mock_select:
            mock_select.return_value = ([mock_stdin], [], [])

            read_input(None, "test")

            # Verify select was called with 0.1 second timeout
            args = mock_select.call_args
            assert args[0][0] == [mock_stdin]  # readfds
            assert args[0][3] == 0.1  # timeout


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_read_input_none_as_file_path(self, capsys):
        """Test None file path tries to read from stdin."""
        with patch('sys.stdin') as mock_stdin:
            mock_stdin.isatty.return_value = True

            with pytest.raises(SystemExit):
                read_input(None, "test")

    def test_read_input_symlink_to_file(self, tmp_path):
        """Test reading from symlink works."""
        actual_file = tmp_path / "actual.txt"
        actual_file.write_text("content via symlink")

        symlink = tmp_path / "link.txt"
        symlink.symlink_to(actual_file)

        result = read_input(str(symlink))

        assert result == "content via symlink"

    @patch('sys.stdin')
    def test_read_input_stdin_binary_mode(self, mock_stdin):
        """Test reading text from stdin."""
        mock_stdin.read.return_value = "text content"
        mock_stdin.isatty.return_value = False

        with patch('select.select') as mock_select:
            mock_select.return_value = ([mock_stdin], [], [])

            result = read_input(None)

        assert isinstance(result, str)
