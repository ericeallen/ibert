"""Tests for function documentation task."""

from src.ibert.tasks import FunctionDocumentationTask


class TestFunctionDocumentationTask:
    """Tests for FunctionDocumentationTask."""

    def test_system_prompt(self, mock_model):
        """Test system prompt generation."""
        task = FunctionDocumentationTask(mock_model)
        prompt = task.get_system_prompt()
        assert "docstring" in prompt.lower()
        assert "documentation" in prompt.lower()

    def test_format_prompt_simple(self, mock_model):
        """Test simple prompt formatting."""
        task = FunctionDocumentationTask(mock_model)
        code = "def filter_by_age(table, min_age):\n    return table.filter(table.age >= min_age)"
        prompt = task.format_prompt(code)
        assert code in prompt
        assert "google" in prompt.lower()  # default style

    def test_format_prompt_numpy_style(self, mock_model):
        """Test prompt formatting with NumPy style."""
        task = FunctionDocumentationTask(mock_model)
        code = "def filter_by_age(table, min_age):\n    return table.filter(table.age >= min_age)"
        prompt = task.format_prompt(code, style="numpy")
        assert code in prompt
        assert "numpy" in prompt.lower()

    def test_format_prompt_no_examples(self, mock_model):
        """Test prompt formatting without examples."""
        task = FunctionDocumentationTask(mock_model)
        code = "def filter_by_age(table, min_age):\n    return table.filter(table.age >= min_age)"
        prompt = task.format_prompt(code, include_examples=False)
        assert code in prompt
        # The prompt should not explicitly ask for examples

    def test_execute(self, mock_model_with_response):
        """Test executing documentation generation."""
        documented_code = '''def filter_by_age(table, min_age):
    """Filter table by minimum age."""
    return table.filter(table.age >= min_age)'''
        model = mock_model_with_response(documented_code)
        task = FunctionDocumentationTask(model)
        code = "def filter_by_age(table, min_age):\n    return table.filter(table.age >= min_age)"
        result = task.execute(code)
        assert "Filter table by minimum age" in result

    def test_post_process_removes_code_blocks(self, mock_model):
        """Test post-processing removes markdown code blocks."""
        task = FunctionDocumentationTask(mock_model)
        output = '''```python
def foo():
    """Docstring."""
    pass
```'''
        result = task.post_process(output)
        assert '"""Docstring."""' in result
        assert "```" not in result
