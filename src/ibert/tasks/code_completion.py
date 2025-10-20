"""Code completion task for Ibis expressions."""

from typing import Optional

from .base import BaseTask


class CodeCompletionTask(BaseTask):
    """Complete partial Ibis expressions.

    Given a partial Ibis code snippet, this task completes it to form
    a valid, working expression.
    """

    def get_system_prompt(self) -> str:
        """Get the system prompt for code completion."""
        return """You are an expert Ibis code completion assistant.

Your task is to complete partial Ibis expressions into valid, working code.
Ibis is a Python dataframe library that compiles to SQL.

Guidelines:
- Complete the code naturally and idiomatically
- Use proper Ibis API patterns and methods
- Ensure the completion is syntactically correct
- Keep completions concise and focused
- Only return the completion, not the entire code unless necessary
- If context is unclear, make reasonable assumptions based on Ibis best practices

Example:
Input: "table.filter(table.age >"
Output: "table.filter(table.age > 18)"
"""

    def format_prompt(self, input_text: str, **kwargs) -> str:
        """Format the prompt for code completion.

        Args:
            input_text: Partial code to complete
            **kwargs: Optional parameters:
                - context: Additional context about the code

        Returns:
            Formatted prompt
        """
        context = kwargs.get("context", "")

        if context:
            return f"""Context: {context}

Complete the following partial Ibis code:

```python
{input_text}
```

Provide only the completed code."""
        else:
            return f"""Complete the following partial Ibis code:

```python
{input_text}
```

Provide only the completed code."""

    def execute(self, input_text: str, **kwargs) -> str:
        """Execute code completion.

        Args:
            input_text: Partial code to complete
            **kwargs: Additional parameters (context, etc.)

        Returns:
            Completed code
        """
        system_prompt = self.get_system_prompt()
        user_prompt = self.format_prompt(input_text, **kwargs)

        output = self.model.generate(
            prompt=user_prompt, system_prompt=system_prompt
        )

        return self.post_process(output)

    def post_process(self, output: str) -> str:
        """Post-process the completion output.

        Args:
            output: Raw model output

        Returns:
            Cleaned code completion
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
