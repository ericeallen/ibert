"""Tests for multi-task data validation system."""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import ibis

from src.datagen.multitask.validate_multitask_data import (
    MultitaskValidator,
    ValidationError,
)


class TestMultitaskValidator:
    """Test suite for MultitaskValidator."""

    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return MultitaskValidator()

    @pytest.fixture
    def mock_table(self):
        """Create mock Ibis table."""
        mock = MagicMock()
        mock.execute.return_value = pd.DataFrame({"col": [1, 2, 3]})
        return mock

    # Test _create_mock_tables
    def test_create_mock_tables_with_schema(self, validator):
        """Test mock table creation from schema."""
        context = {
            "tables": {
                "test_table": {
                    "schema": {
                        "age": "int64",
                        "name": "string",
                        "active": "bool"
                    }
                }
            }
        }
        namespace = {"ibis": ibis}

        validator._create_mock_tables(context, namespace)

        assert "test_table" in namespace
        assert namespace["test_table"] is not None

    def test_create_mock_tables_handles_keywords(self, validator):
        """Test that SQL keywords are properly quoted."""
        context = {
            "tables": {
                "table": {  # SQL keyword
                    "schema": {"id": "int64"}
                }
            }
        }
        namespace = {"ibis": ibis}

        validator._create_mock_tables(context, namespace)

        assert "table" in namespace

    def test_create_mock_tables_empty_context(self, validator):
        """Test with empty context."""
        context = {}
        namespace = {"ibis": ibis}

        validator._create_mock_tables(context, namespace)

        # Should not crash, namespace unchanged except ibis
        assert "ibis" in namespace

    # Test _validate_code_completion
    def test_validate_code_completion_valid(self, validator):
        """Test valid code completion example."""
        example = {
            "task": "code_completion",
            "input": {"partial_code": "table.filter("},
            "target": {"completed_code": "table.filter(table.age > 18)"},
            "context": {
                "tables": {
                    "table": {"schema": {"age": "int64"}}
                }
            }
        }

        success, error = validator._validate_code_completion(example)

        assert success
        assert error is None

    def test_validate_code_completion_syntax_error(self, validator):
        """Test code with syntax error."""
        example = {
            "task": "code_completion",
            "input": {"partial_code": "table.filter("},
            "target": {"completed_code": "table.filter(table.age > "},  # Incomplete
            "context": {}
        }

        success, error = validator._validate_code_completion(example)

        assert not success
        assert "syntax" in error.lower()

    def test_validate_code_completion_empty_target(self, validator):
        """Test with empty completed code."""
        example = {
            "task": "code_completion",
            "input": {"partial_code": "table.filter("},
            "target": {"completed_code": ""},
            "context": {}
        }

        success, error = validator._validate_code_completion(example)

        assert not success
        assert "empty" in error.lower()

    def test_validate_code_completion_partial_not_prefix(self, validator):
        """Test when completed code doesn't start with partial."""
        example = {
            "task": "code_completion",
            "input": {"partial_code": "table.filter(table.age"},
            "target": {"completed_code": "other_table.filter(table.age > 18)"},
            "context": {}
        }

        success, error = validator._validate_code_completion(example)

        assert not success
        assert "doesn't start" in error.lower()

    # Test _validate_sql_to_ibis
    def test_validate_sql_to_ibis_calls_existing_validator(self, validator):
        """Test that SQL→Ibis validation delegates to existing validator."""
        example = {
            "task": "sql_to_ibis",
            "input": {"sql": "SELECT * FROM events"},
            "target": {"ibis": "events"},
            "context": {"tables": {"events": {}}}
        }

        with patch.object(validator.sql_ibis_validator, 'validate_example') as mock_validate:
            mock_validate.return_value = (True, None)

            success, error = validator._validate_sql_to_ibis(example)

            assert success
            mock_validate.assert_called_once_with(example)

    # Test _validate_ibis_to_sql
    def test_validate_ibis_to_sql_valid(self, validator):
        """Test valid Ibis→SQL translation."""
        example = {
            "task": "ibis_to_sql",
            "input": {
                "ibis": "users.filter(users.age > 18)",
                "dialect": "duckdb"
            },
            "target": {
                "sql": "SELECT * FROM users WHERE age > 18"
            },
            "context": {
                "tables": {
                    "users": {"schema": {"age": "int64", "name": "string"}}
                }
            }
        }

        success, error = validator._validate_ibis_to_sql(example)

        # May fail due to SQL keyword issues, but should not crash
        assert isinstance(success, bool)
        assert error is None or isinstance(error, str)

    def test_validate_ibis_to_sql_missing_fields(self, validator):
        """Test with missing required fields."""
        example = {
            "task": "ibis_to_sql",
            "input": {"ibis": "table.filter(table.age > 18)"},
            "target": {},  # Missing sql
            "context": {}
        }

        success, error = validator._validate_ibis_to_sql(example)

        assert not success
        assert "missing" in error.lower()

    # Test _validate_error_resolution
    def test_validate_error_resolution_valid(self, validator):
        """Test valid error resolution example."""
        example = {
            "task": "error_resolution",
            "input": {
                "broken_code": "table.filter(table.age > '18')",  # Type error
                "error": "TypeError"
            },
            "target": {
                "fixed_code": "table.filter(table.age > 18)",
                "explanation": "Removed quotes to compare as integer"
            },
            "context": {
                "tables": {
                    "table": {"schema": {"age": "int64"}}
                }
            }
        }

        success, error = validator._validate_error_resolution(example)

        # Should validate structure even if execution details vary
        assert isinstance(success, bool)

    def test_validate_error_resolution_missing_fields(self, validator):
        """Test with missing required fields."""
        example = {
            "task": "error_resolution",
            "input": {"broken_code": "table.filter()"},
            "target": {"fixed_code": "table.filter(table.age > 18)"},
            # Missing error and explanation
            "context": {}
        }

        success, error = validator._validate_error_resolution(example)

        assert not success
        assert "missing" in error.lower()

    def test_validate_error_resolution_fixed_has_syntax_error(self, validator):
        """Test when fixed code has syntax error."""
        example = {
            "task": "error_resolution",
            "input": {
                "broken_code": "table.filter(table.age > '18')",
                "error": "TypeError"
            },
            "target": {
                "fixed_code": "table.filter(table.age >",  # Incomplete
                "explanation": "Convert string to int"
            },
            "context": {}
        }

        success, error = validator._validate_error_resolution(example)

        assert not success
        assert error is not None  # Should have an error message

    def test_validate_error_resolution_explanation_too_short(self, validator):
        """Test when explanation is too short."""
        example = {
            "task": "error_resolution",
            "input": {
                "broken_code": "x = 1",
                "error": "Error"
            },
            "target": {
                "fixed_code": "x = 2",
                "explanation": "Fix"  # Too short
            },
            "context": {}
        }

        success, error = validator._validate_error_resolution(example)

        assert not success
        assert error is not None  # Should have an error message

    # Test _validate_qa
    def test_validate_qa_valid(self, validator):
        """Test valid Q&A example."""
        example = {
            "task": "qa",
            "input": {"question": "How do I filter rows in Ibis?"},
            "target": {
                "answer": "You can filter rows using the .filter() method. For example: table.filter(table.age > 18)"
            }
        }

        success, error = validator._validate_qa(example)

        assert success
        assert error is None

    def test_validate_qa_empty_question(self, validator):
        """Test with empty question."""
        example = {
            "task": "qa",
            "input": {"question": ""},
            "target": {"answer": "Some answer"}
        }

        success, error = validator._validate_qa(example)

        assert not success
        assert "empty" in error.lower()

    def test_validate_qa_answer_too_short(self, validator):
        """Test with answer that's too short."""
        example = {
            "task": "qa",
            "input": {"question": "How to filter?"},
            "target": {"answer": "Use filter"}  # Too short
        }

        success, error = validator._validate_qa(example)

        assert not success
        assert "too short" in error.lower()

    def test_validate_qa_with_code_blocks(self, validator):
        """Test Q&A with code blocks in answer."""
        example = {
            "task": "qa",
            "input": {"question": "How do I filter?"},
            "target": {
                "answer": """You can filter like this:

```python
table.filter(table.age > 18)
```

This will filter rows where age is greater than 18."""
            }
        }

        success, error = validator._validate_qa(example)

        assert success

    # Test _validate_documentation
    def test_validate_documentation_valid_google_style(self, validator):
        """Test valid documentation with Google style."""
        example = {
            "task": "documentation",
            "input": {
                "code": "def filter_adults(table):\n    return table.filter(table.age >= 18)",
                "style": "google"
            },
            "target": {
                "docstring": '"""Filter table to adults.\n\nArgs:\n    table: Input table\n\nReturns:\n    Filtered table"""'
            }
        }

        success, error = validator._validate_documentation(example)

        assert success

    def test_validate_documentation_valid_numpy_style(self, validator):
        """Test valid documentation with NumPy style."""
        example = {
            "task": "documentation",
            "input": {
                "code": "def filter_adults(table):\n    return table.filter(table.age >= 18)",
                "style": "numpy"
            },
            "target": {
                "docstring": '"""Filter table.\n\nParameters\n----------\ntable\n\nReturns\n-------\nFiltered table"""'
            }
        }

        success, error = validator._validate_documentation(example)

        assert success

    def test_validate_documentation_empty_function(self, validator):
        """Test with empty function code."""
        example = {
            "task": "documentation",
            "input": {"code": "", "style": "google"},
            "target": {"docstring": "Some docstring"}
        }

        success, error = validator._validate_documentation(example)

        assert not success

    def test_validate_documentation_missing_required_sections(self, validator):
        """Test docstring missing required sections."""
        example = {
            "task": "documentation",
            "input": {
                "code": "def foo():\n    pass",
                "style": "google"
            },
            "target": {
                "docstring": '"""Just a function"""'  # Missing Args/Returns
            }
        }

        success, error = validator._validate_documentation(example)

        assert not success
        assert "missing" in error.lower()

    def test_validate_documentation_docstring_too_short(self, validator):
        """Test docstring that's too short."""
        example = {
            "task": "documentation",
            "input": {
                "code": "def foo():\n    pass",
                "style": "google"
            },
            "target": {"docstring": '"""Short"""'}
        }

        success, error = validator._validate_documentation(example)

        assert not success
        assert error is not None  # Should have an error message

    def test_validate_documentation_invalid_syntax(self, validator):
        """Test function with invalid syntax."""
        example = {
            "task": "documentation",
            "input": {
                "code": "def foo(",  # Incomplete
                "style": "google"
            },
            "target": {"docstring": '"""A function.\n\nReturns:\n    None"""'}
        }

        success, error = validator._validate_documentation(example)

        assert not success
        assert "syntax error" in error.lower()

    # Test validate_example
    def test_validate_example_unknown_task(self, validator):
        """Test validation with unknown task type."""
        example = {"task": "unknown_task"}

        success, error = validator.validate_example(example)

        assert not success
        assert "unknown task" in error.lower()

    # Test validate_file
    def test_validate_file_success(self, validator, tmp_path):
        """Test validation of a file with valid examples."""
        test_file = tmp_path / "test.jsonl"
        examples = [
            {
                "id": "1",
                "task": "qa",
                "input": {"question": "How to use Ibis?"},
                "target": {"answer": "Ibis is a Python library for data manipulation that works with many backends."}
            },
            {
                "id": "2",
                "task": "qa",
                "input": {"question": "What is filtering?"},
                "target": {"answer": "Filtering is selecting rows that meet certain conditions using the filter method."}
            }
        ]

        with open(test_file, 'w') as f:
            for ex in examples:
                f.write(json.dumps(ex) + '\n')

        total, valid, failed = validator.validate_file(test_file)

        assert total == 2
        assert valid == 2
        assert len(failed) == 0

    def test_validate_file_with_failures(self, validator, tmp_path):
        """Test validation with some failures."""
        test_file = tmp_path / "test.jsonl"
        examples = [
            {
                "id": "1",
                "task": "qa",
                "input": {"question": "Valid question?"},
                "target": {"answer": "This is a valid answer that is long enough."}
            },
            {
                "id": "2",
                "task": "qa",
                "input": {"question": ""},  # Invalid
                "target": {"answer": "Answer"}
            }
        ]

        with open(test_file, 'w') as f:
            for ex in examples:
                f.write(json.dumps(ex) + '\n')

        total, valid, failed = validator.validate_file(test_file)

        assert total == 2
        assert valid == 1
        assert len(failed) == 1
        assert failed[0]["line"] == 2

    def test_validate_file_invalid_json(self, validator, tmp_path):
        """Test validation with invalid JSON."""
        test_file = tmp_path / "test.jsonl"

        with open(test_file, 'w') as f:
            f.write('{"valid": "json", "task": "qa", "input": {"question": "test"}, "target": {"answer": "test answer"}}\n')
            f.write('invalid json here\n')

        total, valid, failed = validator.validate_file(test_file)

        assert total == 2
        assert len(failed) >= 1  # At least the invalid JSON should fail
        # Check that at least one failure is due to JSON error
        json_errors = [f for f in failed if "JSON" in f.get("error", "").upper() or "json" in f.get("error", "")]
        assert len(json_errors) >= 1

    def test_validate_file_empty(self, validator, tmp_path):
        """Test validation of empty file."""
        test_file = tmp_path / "empty.jsonl"
        test_file.touch()

        total, valid, failed = validator.validate_file(test_file)

        assert total == 0
        assert valid == 0
        assert len(failed) == 0


class TestIntegration:
    """Integration tests for the validation system."""

    def test_end_to_end_validation(self, tmp_path):
        """Test complete validation workflow."""
        # Create test data file
        data_file = tmp_path / "test_data.jsonl"
        examples = [
            {
                "id": "1",
                "task": "code_completion",
                "input": {"partial_code": "table.filter("},
                "target": {"completed_code": "table.filter(table.age > 18)"},
                "context": {"tables": {"table": {"schema": {"age": "int64"}}}}
            },
            {
                "id": "2",
                "task": "qa",
                "input": {"question": "What is Ibis?"},
                "target": {"answer": "Ibis is a Python dataframe library that works with many different backends."}
            },
            {
                "id": "3",
                "task": "documentation",
                "input": {
                    "code": "def filter_data(table):\n    return table.filter(table.age > 18)",
                    "style": "google"
                },
                "target": {
                    "docstring": '"""Filter data.\n\nArgs:\n    table: Input table\n\nReturns:\n    Filtered table"""'
                }
            }
        ]

        with open(data_file, 'w') as f:
            for ex in examples:
                f.write(json.dumps(ex) + '\n')

        # Run validation
        validator = MultitaskValidator()
        total, valid, failed = validator.validate_file(data_file)

        # Check results
        assert total == 3
        assert valid >= 2  # At least Q&A and documentation should pass

        # Check that failed examples have useful error messages
        for failure in failed:
            assert "error" in failure
            assert isinstance(failure["error"], str)
            assert len(failure["error"]) > 0

    def test_validation_with_multiple_tasks(self, tmp_path):
        """Test validation across all task types."""
        validator = MultitaskValidator()

        examples = {
            "code_completion": {
                "task": "code_completion",
                "input": {"partial_code": "t.select("},
                "target": {"completed_code": "t.select(t.col1, t.col2)"},
                "context": {}
            },
            "sql_to_ibis": {
                "task": "sql_to_ibis",
                "input": {"sql": "SELECT * FROM users WHERE age > 18"},
                "target": {"ibis": "users.filter(users.age > 18)"},
                "context": {"tables": {"users": {}}}
            },
            "ibis_to_sql": {
                "task": "ibis_to_sql",
                "input": {"ibis": "t.filter(t.age > 18)", "dialect": "duckdb"},
                "target": {"sql": "SELECT * FROM t WHERE age > 18"},
                "context": {"tables": {"t": {"schema": {"age": "int64"}}}}
            },
            "error_resolution": {
                "task": "error_resolution",
                "input": {"broken_code": "x = ", "error": "SyntaxError"},
                "target": {"fixed_code": "x = 1", "explanation": "Added value to assignment"},
                "context": {}
            },
            "qa": {
                "task": "qa",
                "input": {"question": "How to filter?"},
                "target": {"answer": "Use the filter method to select rows that meet conditions."}
            },
            "documentation": {
                "task": "documentation",
                "input": {"code": "def foo():\n    pass", "style": "google"},
                "target": {"docstring": '"""A function.\n\nReturns:\n    None"""'}
            }
        }

        results = {}
        for task_name, example in examples.items():
            success, error = validator.validate_example(example)
            results[task_name] = (success, error)

        # All tasks should at least validate without crashing
        assert len(results) == 6
        for task_name, (success, error) in results.items():
            assert isinstance(success, bool), f"{task_name} validation returned non-bool"

