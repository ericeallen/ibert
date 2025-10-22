"""Tests for multi-task data generation system."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.datagen.multitask.generate_multitask_data import MultitaskDataGenerator


class TestMultitaskDataGenerator:
    """Test suite for MultitaskDataGenerator."""

    @pytest.fixture
    def temp_templates(self, tmp_path):
        """Create temporary template directory structure."""
        templates_dir = tmp_path / "templates"

        # Create code_completion templates
        cc_dir = templates_dir / "code_completion"
        cc_dir.mkdir(parents=True)
        (cc_dir / "test_template.yaml").write_text(
            """
name: test_completion
task: code_completion
system_prompt: "Complete the code"
difficulty: easy
features: ["filter"]
variations:
  - name: simple_filter
    input:
      partial_code: "table.filter("
    target:
      completed_code: "table.filter(table.age > 18)"
      explanation: "Filter rows"
    context:
      tables:
        table:
          schema:
            age: "int64"
"""
        )

        # Create documentation templates
        doc_dir = templates_dir / "documentation"
        doc_dir.mkdir(parents=True)
        (doc_dir / "test_doc.yaml").write_text(
            """
name: test_docs
task: documentation
system_prompt: "Generate docstring"
difficulty: easy
features: ["docstring"]
variations:
  - name: google_style
    input:
      code: "def foo():\\n    pass"
      style: "google"
    target:
      docstring: '\"\"\"A function.\\n\\nReturns:\\n    None\"\"\"'
"""
        )

        # Create Q&A templates
        qa_dir = templates_dir / "qa"
        qa_dir.mkdir(parents=True)
        (qa_dir / "test_qa.yaml").write_text(
            """
name: test_qa
task: qa
system_prompt: "Answer questions"
difficulty: easy
features: ["basic"]
variations:
  - name: filter_question
    input:
      question: "How do I filter?"
    target:
      answer: "Use the filter method to select rows meeting criteria."
"""
        )

        return templates_dir

    @pytest.fixture
    def generator(self, temp_templates, tmp_path):
        """Create generator instance."""
        output_dir = tmp_path / "output"
        return MultitaskDataGenerator(temp_templates, output_dir)

    # Test initialization
    def test_generator_initialization(self, generator, temp_templates, tmp_path):
        """Test generator initializes correctly."""
        assert generator.templates_dir == temp_templates
        assert generator.output_dir.exists()
        assert len(generator.generators) == 6

    def test_output_directory_created(self, tmp_path):
        """Test output directory is created if it doesn't exist."""
        templates_dir = tmp_path / "templates"
        templates_dir.mkdir()
        output_dir = tmp_path / "nested" / "output"

        MultitaskDataGenerator(templates_dir, output_dir)

        assert output_dir.exists()
        assert output_dir.is_dir()

    # Test _load_templates
    def test_load_templates_success(self, generator):
        """Test loading templates from directory."""
        templates = generator._load_templates("code_completion")

        assert len(templates) >= 1
        assert templates[0]["name"] == "test_completion"
        assert templates[0]["task"] == "code_completion"

    def test_load_templates_nonexistent_directory(self, generator):
        """Test loading from nonexistent directory."""
        templates = generator._load_templates("nonexistent")

        assert templates == []

    def test_load_templates_invalid_yaml(self, generator, tmp_path):
        """Test handling of invalid YAML."""
        bad_dir = generator.templates_dir / "bad_task"
        bad_dir.mkdir()
        (bad_dir / "invalid.yaml").write_text("{ invalid yaml {{")

        templates = generator._load_templates("bad_task")

        # Should handle error gracefully
        assert isinstance(templates, list)

    # Test _write_jsonl
    def test_write_jsonl(self, generator):
        """Test writing JSONL output."""
        examples = [
            {"id": "1", "task": "test", "data": "example1"},
            {"id": "2", "task": "test", "data": "example2"},
        ]

        generator._write_jsonl("test_output", examples)

        output_file = generator.output_dir / "test_output.jsonl"
        assert output_file.exists()

        with open(output_file) as f:
            loaded = [json.loads(line) for line in f]

        assert len(loaded) == 2
        assert loaded[0]["id"] == "1"

    # Test _generate_code_completion
    def test_generate_code_completion(self, generator):
        """Test code completion generation."""
        count = generator._generate_code_completion()

        assert count == 1  # One variation in template
        output_file = generator.output_dir / "code_completion.jsonl"
        assert output_file.exists()

        with open(output_file) as f:
            examples = [json.loads(line) for line in f]

        assert len(examples) == 1
        ex = examples[0]
        assert ex["task"] == "code_completion"
        assert "id" in ex
        assert "system_prompt" in ex
        assert "input" in ex
        assert "target" in ex
        assert "meta" in ex
        assert ex["meta"]["template"] == "test_completion"

    def test_generate_code_completion_creates_unique_ids(self, generator):
        """Test that each example gets a unique ID."""
        generator._generate_code_completion()

        output_file = generator.output_dir / "code_completion.jsonl"
        with open(output_file) as f:
            examples = [json.loads(line) for line in f]

        ids = [ex["id"] for ex in examples]
        assert len(ids) == len(set(ids))  # All unique

    # Test _generate_sql_to_ibis
    def test_generate_sql_to_ibis_uses_existing_data(self, generator, tmp_path):
        """Test SQL→Ibis generation uses existing file."""
        # Create actual file that the method can use
        sql_data_dir = tmp_path / "data" / "sql2ibis"
        sql_data_dir.mkdir(parents=True)
        train_file = sql_data_dir / "train.jsonl"
        train_file.write_text(
            '{"task":"sql_to_ibis","data":"test1"}\n{"task":"sql_to_ibis","data":"test2"}\n'
        )

        # Patch Path class to return our temporary file location
        with patch("src.datagen.multitask.generate_multitask_data.Path") as mock_path_cls:
            # When Path("data/sql2ibis/train.jsonl") is called, return our temp file
            mock_path_cls.return_value = train_file

            count = generator._generate_sql_to_ibis()

            # Should count 2 lines
            assert count == 2

    def test_generate_sql_to_ibis_handles_missing_file(self, generator):
        """Test SQL→Ibis handles missing existing file."""
        # The method checks for data/sql2ibis/train.jsonl which won't exist in test
        with patch.object(Path, "exists", return_value=False):
            count = generator._generate_sql_to_ibis()

            # Should return 0 and print warning
            assert count == 0

    # Test _generate_ibis_to_sql
    def test_generate_ibis_to_sql(self, generator, temp_templates):
        """Test Ibis→SQL generation."""
        # Create ibis_to_sql template
        ibis_sql_dir = temp_templates / "ibis_to_sql"
        ibis_sql_dir.mkdir()
        (ibis_sql_dir / "test.yaml").write_text(
            """
name: test_ibis_to_sql
task: ibis_to_sql
system_prompt: "Translate to SQL"
variations:
  - name: simple
    input:
      ibis: "table.filter(table.age > 18)"
      dialect: "duckdb"
    target:
      sql: "SELECT * FROM table WHERE age > 18"
"""
        )

        count = generator._generate_ibis_to_sql()

        assert count == 1
        output_file = generator.output_dir / "ibis_to_sql.jsonl"
        with open(output_file) as f:
            examples = [json.loads(line) for line in f]

        ex = examples[0]
        assert ex["task"] == "ibis_to_sql"
        assert ex["dialect"] == "duckdb"

    # Test _generate_error_resolution
    def test_generate_error_resolution(self, generator, temp_templates):
        """Test error resolution generation."""
        # Create error_resolution template
        err_dir = temp_templates / "error_resolution"
        err_dir.mkdir()
        (err_dir / "test.yaml").write_text(
            """
name: test_errors
task: error_resolution
system_prompt: "Fix errors"
variations:
  - name: type_error
    input:
      broken_code: "x = '1' + 1"
      error: "TypeError"
    target:
      fixed_code: "x = int('1') + 1"
      explanation: "Convert string to int"
    error_type: "type"
"""
        )

        count = generator._generate_error_resolution()

        assert count == 1
        output_file = generator.output_dir / "error_resolution.jsonl"
        with open(output_file) as f:
            examples = [json.loads(line) for line in f]

        ex = examples[0]
        assert ex["task"] == "error_resolution"
        assert ex["meta"]["error_type"] == "type"

    # Test _generate_qa
    def test_generate_qa(self, generator):
        """Test Q&A generation."""
        count = generator._generate_qa()

        assert count == 1
        output_file = generator.output_dir / "qa.jsonl"
        with open(output_file) as f:
            examples = [json.loads(line) for line in f]

        ex = examples[0]
        assert ex["task"] == "qa"
        assert "question" in ex["input"]
        assert "answer" in ex["target"]

    # Test _generate_documentation
    def test_generate_documentation(self, generator):
        """Test documentation generation."""
        count = generator._generate_documentation()

        assert count == 1
        output_file = generator.output_dir / "documentation.jsonl"
        with open(output_file) as f:
            examples = [json.loads(line) for line in f]

        ex = examples[0]
        assert ex["task"] == "documentation"
        assert ex["meta"]["style"] == "google"

    # Test _combine_all_tasks
    def test_combine_all_tasks(self, generator):
        """Test combining all task files."""
        # Generate some data first
        generator._generate_code_completion()
        generator._generate_qa()
        generator._generate_documentation()

        generator._combine_all_tasks()

        combined_file = generator.output_dir / "train_complete.jsonl"
        assert combined_file.exists()

        with open(combined_file) as f:
            examples = [json.loads(line) for line in f]

        # Should have examples from multiple tasks
        tasks = {ex["task"] for ex in examples}
        assert len(tasks) >= 2

    def test_combine_all_tasks_excludes_combined_file(self, generator):
        """Test that combined file doesn't include itself."""
        # Create individual files
        generator._generate_qa()

        # Create an existing combined file
        combined = generator.output_dir / "train_complete.jsonl"
        combined.write_text('{"task":"old","data":"should be excluded"}\n')

        # Recombine
        generator._combine_all_tasks()

        # Read result
        with open(combined) as f:
            examples = [json.loads(line) for line in f]

        # Should only have new content, not old
        assert all(ex.get("data") != "should be excluded" for ex in examples)

    # Test generate_task
    def test_generate_task_specific(self, generator):
        """Test generating specific task."""
        count = generator.generate_task("documentation")

        assert count == 1

    def test_generate_task_unknown(self, generator):
        """Test generating unknown task raises error."""
        with pytest.raises(ValueError, match="Unknown task"):
            generator.generate_task("nonexistent")

    # Test generate_all
    def test_generate_all(self, generator):
        """Test generating all tasks."""
        stats = generator.generate_all()

        assert isinstance(stats, dict)
        assert len(stats) == 6
        assert all(
            task in stats
            for task in [
                "code_completion",
                "sql_to_ibis",
                "ibis_to_sql",
                "error_resolution",
                "qa",
                "documentation",
            ]
        )

    def test_generate_all_creates_all_files(self, generator):
        """Test that generate_all creates all output files."""
        generator.generate_all()

        expected_files = [
            "code_completion.jsonl",
            "documentation.jsonl",
            "qa.jsonl",
            "train_complete.jsonl",
        ]

        for filename in expected_files:
            path = generator.output_dir / filename
            if filename != "sql_to_ibis.jsonl":  # May not exist if no source
                assert (
                    path.exists()
                    or filename == "error_resolution.jsonl"
                    or filename == "ibis_to_sql.jsonl"
                )


class TestIntegration:
    """Integration tests for multi-task data generation."""

    def test_end_to_end_generation(self, tmp_path):
        """Test complete generation workflow."""
        # Create comprehensive templates
        templates_dir = tmp_path / "templates"

        # Code completion
        cc_dir = templates_dir / "code_completion"
        cc_dir.mkdir(parents=True)
        (cc_dir / "filters.yaml").write_text(
            """
name: filter_examples
task: code_completion
system_prompt: "Complete Ibis code"
variations:
  - name: numeric_filter
    input:
      partial_code: "table.filter(table.age >"
    target:
      completed_code: "table.filter(table.age > 18)"
  - name: string_filter
    input:
      partial_code: "table.filter(table.name =="
    target:
      completed_code: 'table.filter(table.name == "Alice")'
"""
        )

        # Q&A
        qa_dir = templates_dir / "qa"
        qa_dir.mkdir()
        (qa_dir / "basic.yaml").write_text(
            """
name: basic_qa
task: qa
system_prompt: "Answer questions about Ibis"
variations:
  - name: what_is_ibis
    input:
      question: "What is Ibis?"
    target:
      answer: "Ibis is a Python library for working with data."
  - name: how_to_filter
    input:
      question: "How do I filter rows?"
    target:
      answer: "Use the filter method with a boolean expression."
"""
        )

        # Run generation
        output_dir = tmp_path / "output"
        generator = MultitaskDataGenerator(templates_dir, output_dir)
        stats = generator.generate_all()

        # Verify results
        assert stats["code_completion"] == 2
        assert stats["qa"] == 2

        # Check combined file
        combined = output_dir / "train_complete.jsonl"
        assert combined.exists()

        with open(combined) as f:
            examples = [json.loads(line) for line in f]

        assert len(examples) >= 4
        tasks = {ex["task"] for ex in examples}
        assert "code_completion" in tasks
        assert "qa" in tasks

    def test_generation_with_multiple_template_files(self, tmp_path):
        """Test generation with multiple template files per task."""
        templates_dir = tmp_path / "templates"
        qa_dir = templates_dir / "qa"
        qa_dir.mkdir(parents=True)

        # Create multiple template files
        (qa_dir / "basic.yaml").write_text(
            """
name: basic
task: qa
variations:
  - name: q1
    input:
      question: "Question 1?"
    target:
      answer: "Answer 1"
"""
        )

        (qa_dir / "advanced.yaml").write_text(
            """
name: advanced
task: qa
variations:
  - name: q2
    input:
      question: "Question 2?"
    target:
      answer: "Answer 2"
"""
        )

        output_dir = tmp_path / "output"
        generator = MultitaskDataGenerator(templates_dir, output_dir)
        count = generator.generate_task("qa")

        assert count == 2  # One from each file

    def test_generation_preserves_metadata(self, tmp_path):
        """Test that generation preserves all metadata."""
        templates_dir = tmp_path / "templates"
        doc_dir = templates_dir / "documentation"
        doc_dir.mkdir(parents=True)

        (doc_dir / "test.yaml").write_text(
            """
name: test_template
task: documentation
system_prompt: "Generate docs"
difficulty: hard
features: ["docstring", "typing"]
variations:
  - name: complex_func
    input:
      code: "def process(data):\\n    pass"
      style: "google"
    target:
      docstring: "Complete docstring"
"""
        )

        output_dir = tmp_path / "output"
        generator = MultitaskDataGenerator(templates_dir, output_dir)
        generator.generate_task("documentation")

        output_file = output_dir / "documentation.jsonl"
        with open(output_file) as f:
            ex = json.loads(f.readline())

        # Check all metadata preserved
        assert ex["meta"]["template"] == "test_template"
        assert ex["meta"]["variation"] == "complex_func"
        assert ex["meta"]["difficulty"] == "hard"
        assert "docstring" in ex["meta"]["features"]
        assert ex["meta"]["style"] == "google"

    def test_handles_empty_templates_directory(self, tmp_path):
        """Test handling of empty templates directory."""
        templates_dir = tmp_path / "empty_templates"
        templates_dir.mkdir()

        output_dir = tmp_path / "output"
        generator = MultitaskDataGenerator(templates_dir, output_dir)
        stats = generator.generate_all()

        # Should complete without errors
        assert all(count == 0 for task, count in stats.items() if task != "sql_to_ibis")


class TestMainFunction:
    """Tests for the main() function CLI."""

    @patch("src.datagen.multitask.generate_multitask_data.MultitaskDataGenerator")
    @patch("sys.argv", ["script.py"])
    def test_main_default_all_tasks(self, mock_generator_cls):
        """Test main() with no arguments generates all tasks."""
        from src.datagen.multitask.generate_multitask_data import main

        mock_generator = Mock()
        mock_generator.generate_all.return_value = {
            "code_completion": 10,
            "sql_to_ibis": 20,
            "ibis_to_sql": 15,
            "error_resolution": 5,
            "qa": 8,
            "documentation": 12,
        }
        mock_generator_cls.return_value = mock_generator

        main()

        # Should call generate_all, not generate_task
        mock_generator.generate_all.assert_called_once()
        mock_generator.generate_task.assert_not_called()

    @patch("src.datagen.multitask.generate_multitask_data.MultitaskDataGenerator")
    @patch("sys.argv", ["script.py", "--task", "code_completion"])
    def test_main_specific_task(self, mock_generator_cls):
        """Test main() with --task argument."""
        from src.datagen.multitask.generate_multitask_data import main

        mock_generator = Mock()
        mock_generator.generate_task.return_value = 25
        mock_generator_cls.return_value = mock_generator

        main()

        # Should call generate_task with specified task
        mock_generator.generate_task.assert_called_once_with("code_completion")
        mock_generator.generate_all.assert_not_called()

    @patch("src.datagen.multitask.generate_multitask_data.MultitaskDataGenerator")
    @patch(
        "sys.argv", ["script.py", "--templates", "/custom/templates", "--output", "/custom/output"]
    )
    def test_main_custom_paths(self, mock_generator_cls):
        """Test main() with custom paths."""
        from src.datagen.multitask.generate_multitask_data import main

        mock_generator = Mock()
        mock_generator.generate_all.return_value = {}
        mock_generator_cls.return_value = mock_generator

        main()

        # Check generator was initialized with custom paths
        args = mock_generator_cls.call_args[0]
        assert str(args[0]) == "/custom/templates"
        assert str(args[1]) == "/custom/output"

    @patch("src.datagen.multitask.generate_multitask_data.MultitaskDataGenerator")
    @patch("sys.argv", ["script.py", "--task", "qa"])
    def test_main_prints_single_task_stats(self, mock_generator_cls, capsys):
        """Test main() prints stats for single task."""
        from src.datagen.multitask.generate_multitask_data import main

        mock_generator = Mock()
        mock_generator.generate_task.return_value = 42
        mock_generator_cls.return_value = mock_generator

        main()

        captured = capsys.readouterr()
        assert "42" in captured.out
        assert "qa" in captured.out

    @patch("src.datagen.multitask.generate_multitask_data.MultitaskDataGenerator")
    @patch("sys.argv", ["script.py"])
    def test_main_prints_all_tasks_summary(self, mock_generator_cls, capsys):
        """Test main() prints summary for all tasks."""
        from src.datagen.multitask.generate_multitask_data import main

        mock_generator = Mock()
        mock_generator.generate_all.return_value = {
            "code_completion": 10,
            "sql_to_ibis": 20,
            "ibis_to_sql": 15,
            "error_resolution": 5,
            "qa": 8,
            "documentation": 12,
        }
        mock_generator_cls.return_value = mock_generator

        main()

        captured = capsys.readouterr()
        assert "SUMMARY" in captured.out
        assert "code_completion" in captured.out
        assert "TOTAL" in captured.out
        # Total should be 70
        assert "70" in captured.out
