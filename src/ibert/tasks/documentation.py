"""Function documentation generation task."""

from typing import Optional

from .base import BaseTask


class FunctionDocumentationTask(BaseTask):
    """Generate docstrings for Ibis functions.

    Creates comprehensive documentation for Ibis code including functions,
    methods, and code blocks.
    """

    def get_system_prompt(self) -> str:
        """Get the system prompt for documentation generation."""
        return """You are an expert technical writer specializing in Ibis documentation.

Your task is to generate clear, comprehensive docstrings for Ibis code.

Guidelines:
- Follow Google-style or NumPy-style docstring format
- Include description of what the function/code does
- Document all parameters with types
- Document return values with types
- Include usage examples when helpful
- Note any important behaviors or edge cases
- Keep descriptions clear and concise
- Use proper reStructuredText formatting

Example:
Input:
```python
def filter_by_age(table, min_age):
    return table.filter(table.age >= min_age)
```

Output:
```python
def filter_by_age(table, min_age):
    \"\"\"Filter table rows by minimum age.

    Args:
        table: Ibis table expression to filter
        min_age (int): Minimum age threshold

    Returns:
        Ibis table expression with filtered rows where age >= min_age

    Example:
        >>> filtered = filter_by_age(users, 18)
        >>> result = filtered.execute()
    \"\"\"
    return table.filter(table.age >= min_age)
```
"""

    def format_prompt(self, input_text: str, **kwargs) -> str:
        """Format the prompt for documentation generation.

        Args:
            input_text: Code to document
            **kwargs: Optional parameters:
                - style: Docstring style (google, numpy)
                - include_examples: Whether to include examples

        Returns:
            Formatted prompt
        """
        style = kwargs.get("style", "google")
        include_examples = kwargs.get("include_examples", True)

        prompt = f"""Generate a {style}-style docstring for the following Ibis code:

```python
{input_text}
```

"""

        if include_examples:
            prompt += "Include usage examples in the docstring.\n"

        prompt += "\nProvide the complete code with the docstring added."

        return prompt

    def execute(self, input_text: str, **kwargs) -> str:
        """Execute documentation generation.

        Args:
            input_text: Code to document
            **kwargs: Additional parameters (style, include_examples, etc.)

        Returns:
            Code with documentation
        """
        system_prompt = self.get_system_prompt()
        user_prompt = self.format_prompt(input_text, **kwargs)

        output = self.model.generate(
            prompt=user_prompt, system_prompt=system_prompt
        )

        return self.post_process(output)

    def post_process(self, output: str) -> str:
        """Post-process the documented code.

        Args:
            output: Raw model output

        Returns:
            Cleaned documented code
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
