"""Tests for code completion task."""

from src.ibert.tasks import CodeCompletionTask


class TestCodeCompletionTask:
    """Tests for CodeCompletionTask."""

    def test_system_prompt(self, mock_model):
        """Test system prompt generation."""
        task = CodeCompletionTask(mock_model)
        prompt = task.get_system_prompt()
        assert "code completion" in prompt.lower()
        assert "ibis" in prompt.lower()

    def test_format_prompt_simple(self, mock_model):
        """Test simple prompt formatting."""
        task = CodeCompletionTask(mock_model)
        prompt = task.format_prompt("table.filter(")
        assert "table.filter(" in prompt
        assert "complete" in prompt.lower()

    def test_format_prompt_with_context(self, mock_model):
        """Test prompt formatting with context."""
        task = CodeCompletionTask(mock_model)
        prompt = task.format_prompt(
            "table.filter(", context="We have a users table with age column"
        )
        assert "table.filter(" in prompt
        assert "users table" in prompt
        assert "age column" in prompt

    def test_execute(self, mock_model_with_response):
        """Test executing code completion."""
        model = mock_model_with_response("table.filter(table.age > 18)")
        task = CodeCompletionTask(model)
        result = task.execute("table.filter(")
        assert "table.filter(table.age > 18)" in result

    def test_post_process_removes_code_blocks(self, mock_model):
        """Test post-processing removes markdown code blocks."""
        task = CodeCompletionTask(mock_model)
        output = "```python\ntable.filter(table.age > 18)\n```"
        result = task.post_process(output)
        assert result == "table.filter(table.age > 18)"
        assert "```" not in result

    def test_post_process_strips_whitespace(self, mock_model):
        """Test post-processing strips whitespace."""
        task = CodeCompletionTask(mock_model)
        output = "\n  table.filter(table.age > 18)  \n"
        result = task.post_process(output)
        assert result == "table.filter(table.age > 18)"
