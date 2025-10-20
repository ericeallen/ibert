"""SQL to Ibis translation task."""

from typing import Optional

from .base import BaseTask


class SQLToIbisTask(BaseTask):
    """Translate SQL queries to Ibis code.

    Converts SQL queries into equivalent Python Ibis expressions.
    This is the reverse of Ibis to SQL translation.
    """

    def get_system_prompt(self) -> str:
        """Get the system prompt for SQL to Ibis translation."""
        return """You are an expert at translating SQL queries to Ibis code.

Your task is to convert SQL queries into equivalent Python Ibis expressions.
Ibis is a lazy-evaluated dataframe library that provides a Pythonic API for working with data.

Guidelines:
- Translate SQL to idiomatic Ibis code
- Use proper Ibis API methods and chaining
- Preserve the semantics and logic of the SQL query
- Handle SELECT, WHERE, JOIN, GROUP BY, ORDER BY, etc.
- Use appropriate Ibis expressions for aggregations and window functions
- Include necessary imports (ibis, ibis.expr.types, etc.)
- Ensure the code is executable and follows Ibis best practices

Example:
Input:
```sql
SELECT name, age
FROM users
WHERE age > 18
ORDER BY age DESC
```

Output:
```python
import ibis

# Assuming connection to backend
table = ibis.table("users")
result = (
    table
    .filter(table.age > 18)
    .select(table.name, table.age)
    .order_by(table.age.desc())
)
```
"""

    def format_prompt(self, input_text: str, **kwargs) -> str:
        """Format the prompt for SQL to Ibis translation.

        Args:
            input_text: SQL query to translate
            **kwargs: Optional parameters:
                - table_name: Name of the table
                - schema: Schema information

        Returns:
            Formatted prompt
        """
        table_name = kwargs.get("table_name", "")
        schema = kwargs.get("schema", "")

        prompt = "Translate the following SQL query to Ibis code:\n\n"

        prompt += f"""```sql
{input_text}
```
"""

        if table_name:
            prompt += f"\nTable name: {table_name}\n"

        if schema:
            prompt += f"\nSchema: {schema}\n"

        prompt += "\nProvide the Ibis code with necessary imports."

        return prompt

    def execute(self, input_text: str, **kwargs) -> str:
        """Execute SQL to Ibis translation.

        Args:
            input_text: SQL query to translate
            **kwargs: Additional parameters (table_name, schema, etc.)

        Returns:
            Ibis code
        """
        system_prompt = self.get_system_prompt()
        user_prompt = self.format_prompt(input_text, **kwargs)

        output = self.model.generate(
            prompt=user_prompt, system_prompt=system_prompt
        )

        return self.post_process(output)

    def post_process(self, output: str) -> str:
        """Post-process the Ibis code output.

        Args:
            output: Raw model output

        Returns:
            Cleaned Ibis code
        """
        output = output.strip()

        # Remove markdown code blocks if present
        if output.startswith("```"):
            lines = output.split("\n")
            # Remove first line (```python or ```)
            lines = lines[1:]
            # Remove last line if it's ```
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            output = "\n".join(lines)

        return output.strip()
