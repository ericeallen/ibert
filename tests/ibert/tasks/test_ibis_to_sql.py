"""Tests for Ibis to SQL translation task."""

from src.ibert.tasks import IbisToSQLTask


class TestIbisToSQLTask:
    """Tests for IbisToSQLTask."""

    def test_system_prompt(self, mock_model):
        """Test system prompt generation."""
        task = IbisToSQLTask(mock_model)
        prompt = task.get_system_prompt()
        assert "ibis" in prompt.lower()
        assert "sql" in prompt.lower()
        assert "translate" in prompt.lower()

    def test_format_prompt_simple(self, mock_model):
        """Test simple prompt formatting."""
        task = IbisToSQLTask(mock_model)
        prompt = task.format_prompt("table.filter(table.age > 18)")
        assert "table.filter(table.age > 18)" in prompt
        assert "standard SQL" in prompt

    def test_format_prompt_with_dialect(self, mock_model):
        """Test prompt formatting with dialect."""
        task = IbisToSQLTask(mock_model)
        prompt = task.format_prompt("table.filter(table.age > 18)", dialect="postgres")
        assert "postgres" in prompt

    def test_format_prompt_with_table_name(self, mock_model):
        """Test prompt formatting with table name."""
        task = IbisToSQLTask(mock_model)
        prompt = task.format_prompt("table.filter(table.age > 18)", table_name="users")
        assert "users" in prompt

    def test_execute(self, mock_model_with_response):
        """Test executing translation."""
        sql_response = "SELECT * FROM users WHERE age > 18"
        model = mock_model_with_response(sql_response)
        task = IbisToSQLTask(model)
        result = task.execute("table.filter(table.age > 18)")
        assert sql_response in result

    def test_post_process_removes_code_blocks(self, mock_model):
        """Test post-processing removes markdown code blocks."""
        task = IbisToSQLTask(mock_model)
        output = "```sql\nSELECT * FROM users\n```"
        result = task.post_process(output)
        assert result == "SELECT * FROM users"
        assert "```" not in result
