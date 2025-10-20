"""Tests for error resolution task."""

import pytest

from src.ibert.tasks import ErrorResolutionTask


class TestErrorResolutionTask:
    """Tests for ErrorResolutionTask."""

    def test_system_prompt(self, mock_model):
        """Test system prompt generation."""
        task = ErrorResolutionTask(mock_model)
        prompt = task.get_system_prompt()
        assert "error" in prompt.lower()
        assert "fix" in prompt.lower()
        assert "ibis" in prompt.lower()

    def test_format_prompt_simple(self, mock_model):
        """Test simple prompt formatting."""
        task = ErrorResolutionTask(mock_model)
        prompt = task.format_prompt('table.filter(table.age > "18")')
        assert 'table.filter(table.age > "18")' in prompt

    def test_format_prompt_with_error(self, mock_model):
        """Test prompt formatting with error message."""
        task = ErrorResolutionTask(mock_model)
        error = "TypeError: '>' not supported between 'IntegerColumn' and 'str'"
        prompt = task.format_prompt(
            'table.filter(table.age > "18")',
            error=error
        )
        assert error in prompt
        assert 'table.filter(table.age > "18")' in prompt

    def test_format_prompt_with_context(self, mock_model):
        """Test prompt formatting with context."""
        task = ErrorResolutionTask(mock_model)
        prompt = task.format_prompt(
            'table.filter(table.age > "18")',
            context="age is an integer column"
        )
        assert "age is an integer column" in prompt

    def test_execute(self, mock_model_with_response):
        """Test executing error resolution."""
        fixed_code = "table.filter(table.age > 18)"
        model = mock_model_with_response(fixed_code)
        task = ErrorResolutionTask(model)
        result = task.execute('table.filter(table.age > "18")')
        assert fixed_code in result

    def test_post_process_removes_code_blocks(self, mock_model):
        """Test post-processing removes markdown code blocks."""
        task = ErrorResolutionTask(mock_model)
        output = "```python\ntable.filter(table.age > 18)\n```"
        result = task.post_process(output)
        assert result == "table.filter(table.age > 18)"
        assert "```" not in result
