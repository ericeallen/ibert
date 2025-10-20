"""Ibis to SQL translation task."""

from typing import Optional

from .base import BaseTask


class IbisToSQLTask(BaseTask):
    """Translate Ibis code to SQL.

    Converts Python Ibis expressions to equivalent SQL queries.
    """

    def get_system_prompt(self) -> str:
        """Get the system prompt for Ibis to SQL translation."""
        return """You are an expert at translating Ibis code to SQL.

Your task is to convert Python Ibis expressions into equivalent SQL queries.
Ibis is a lazy-evaluated dataframe library that compiles to SQL.

Guidelines:
- Translate the Ibis code to standard SQL
- Use the SQL dialect specified (default to standard SQL)
- Ensure the SQL is correct and executable
- Preserve the semantics and logic of the Ibis code
- Use clear, readable SQL formatting
- Include necessary JOINs, WHERE clauses, GROUP BY, ORDER BY as needed

Example:
Input:
```python
table.filter(table.age > 18).select(table.name, table.age)
```

Output:
```sql
SELECT name, age
FROM table
WHERE age > 18
```
"""

    def format_prompt(self, input_text: str, **kwargs) -> str:
        """Format the prompt for Ibis to SQL translation.

        Args:
            input_text: Ibis code to translate
            **kwargs: Optional parameters:
                - dialect: SQL dialect (postgres, mysql, duckdb, etc.)
                - table_name: Name of the table if not in code

        Returns:
            Formatted prompt
        """
        dialect = kwargs.get("dialect", "standard SQL")
        table_name = kwargs.get("table_name", "")

        prompt = f"""Translate the following Ibis code to {dialect}:

```python
{input_text}
```
"""

        if table_name:
            prompt += f"\nTable name: {table_name}\n"

        prompt += "\nProvide only the SQL query."

        return prompt

    def execute(self, input_text: str, **kwargs) -> str:
        """Execute Ibis to SQL translation.

        Args:
            input_text: Ibis code to translate
            **kwargs: Additional parameters (dialect, table_name, etc.)

        Returns:
            SQL query
        """
        system_prompt = self.get_system_prompt()
        user_prompt = self.format_prompt(input_text, **kwargs)

        output = self.model.generate(
            prompt=user_prompt, system_prompt=system_prompt
        )

        return self.post_process(output)

    def post_process(self, output: str) -> str:
        """Post-process the SQL output.

        Args:
            output: Raw model output

        Returns:
            Cleaned SQL query
        """
        output = output.strip()

        # Remove markdown code blocks if present
        if output.startswith("```"):
            lines = output.split("\n")
            # Remove first line (```sql or ```)
            lines = lines[1:]
            # Remove last line if it's ```
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            output = "\n".join(lines)

        return output.strip()
