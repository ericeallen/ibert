"""Error resolution task for fixing Ibis code errors."""

from .base import BaseTask


class ErrorResolutionTask(BaseTask):
    """Fix compilation and type errors in Ibis code.

    Given Ibis code with errors, this task diagnoses and fixes them.
    """

    def get_system_prompt(self) -> str:
        """Get the system prompt for error resolution."""
        return """You are an expert Ibis debugging assistant.

Your task is to fix compilation, type, and runtime errors in Ibis code.
Ibis is a Python dataframe library that compiles to SQL.

Guidelines:
- Analyze the error message carefully
- Identify the root cause of the error
- Fix the code to resolve the error
- Ensure the fixed code is valid and follows Ibis best practices
- Preserve the original intent and logic
- Only return the corrected code, not explanations unless requested

Common Ibis errors:
- Type mismatches in operations
- Invalid column references
- Incorrect method chaining
- Missing imports
- Schema inconsistencies
- Invalid aggregation operations

Example:
Input:
Code: table.filter(table.age > "18")
Error: TypeError: '>' not supported between 'IntegerColumn' and 'str'

Output:
```python
table.filter(table.age > 18)
```
"""

    def format_prompt(self, input_text: str, **kwargs) -> str:
        """Format the prompt for error resolution.

        Args:
            input_text: Code with errors
            **kwargs: Optional parameters:
                - error: Error message/traceback
                - context: Additional context

        Returns:
            Formatted prompt
        """
        error = kwargs.get("error", "")
        context = kwargs.get("context", "")

        prompt = "Fix the following Ibis code:\n\n"

        if context:
            prompt += f"Context: {context}\n\n"

        prompt += f"""Code:
```python
{input_text}
```
"""

        if error:
            prompt += f"""
Error:
```
{error}
```
"""

        prompt += "\nProvide the corrected code."

        return prompt

    def execute(self, input_text: str, **kwargs) -> str:
        """Execute error resolution.

        Args:
            input_text: Code with errors
            **kwargs: Additional parameters (error, context, etc.)

        Returns:
            Fixed code
        """
        system_prompt = self.get_system_prompt()
        user_prompt = self.format_prompt(input_text, **kwargs)

        output = self.model.generate(prompt=user_prompt, system_prompt=system_prompt)

        return self.post_process(output)

    def post_process(self, output: str) -> str:
        """Post-process the fixed code output.

        Args:
            output: Raw model output

        Returns:
            Cleaned fixed code
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
