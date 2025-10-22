"""Tests for SQL to Ibis translation task."""

from src.ibert.tasks import SQLToIbisTask


class TestSQLToIbisTask:
    """Tests for SQLToIbisTask."""

    def test_system_prompt(self, mock_model):
        """Test system prompt generation."""
        task = SQLToIbisTask(mock_model)
        prompt = task.get_system_prompt()
        assert "sql" in prompt.lower()
        assert "ibis" in prompt.lower()
        assert "translate" in prompt.lower()

    def test_format_prompt_simple(self, mock_model):
        """Test simple prompt formatting."""
        task = SQLToIbisTask(mock_model)
        prompt = task.format_prompt("SELECT * FROM users WHERE age > 18")
        assert "SELECT * FROM users WHERE age > 18" in prompt

    def test_format_prompt_with_table_name(self, mock_model):
        """Test prompt formatting with table name."""
        task = SQLToIbisTask(mock_model)
        prompt = task.format_prompt("SELECT * FROM users WHERE age > 18", table_name="users")
        assert "users" in prompt

    def test_format_prompt_with_schema(self, mock_model):
        """Test prompt formatting with schema."""
        task = SQLToIbisTask(mock_model)
        prompt = task.format_prompt(
            "SELECT * FROM users WHERE age > 18", schema="id: int, name: string, age: int"
        )
        assert "id: int" in prompt
        assert "name: string" in prompt

    def test_execute(self, mock_model_with_response):
        """Test executing translation."""
        ibis_response = "table.filter(table.age > 18)"
        model = mock_model_with_response(ibis_response)
        task = SQLToIbisTask(model)
        result = task.execute("SELECT * FROM users WHERE age > 18")
        assert ibis_response in result

    def test_post_process_removes_code_blocks(self, mock_model):
        """Test post-processing removes markdown code blocks."""
        task = SQLToIbisTask(mock_model)
        output = "```python\ntable.filter(table.age > 18)\n```"
        result = task.post_process(output)
        assert result == "table.filter(table.age > 18)"
        assert "```" not in result
