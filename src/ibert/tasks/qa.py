"""Q&A task for answering Ibis-related questions."""

from typing import Optional

from .base import BaseTask


class QATask(BaseTask):
    """Answer questions about Ibis.

    Provides answers to questions about Ibis API, usage, best practices, etc.
    """

    def get_system_prompt(self) -> str:
        """Get the system prompt for Q&A."""
        return """You are an expert Ibis consultant and educator.

Your task is to answer questions about Ibis, the Python dataframe library that compiles to SQL.

Guidelines:
- Provide accurate, helpful answers about Ibis
- Include code examples when relevant
- Explain concepts clearly and concisely
- Reference Ibis documentation and best practices
- Compare to alternatives (pandas, SQL) when helpful
- Point out common pitfalls and gotchas

Topics you can help with:
- Ibis API and methods
- Lazy evaluation and query compilation
- Backend support (DuckDB, PostgreSQL, BigQuery, etc.)
- Performance optimization
- Data types and type system
- Expression building and chaining
- Aggregations and window functions
- Joins and set operations
- User-defined functions (UDFs)
- Best practices and patterns

Provide clear, actionable answers with examples where appropriate.
"""

    def format_prompt(self, input_text: str, **kwargs) -> str:
        """Format the prompt for Q&A.

        Args:
            input_text: Question to answer
            **kwargs: Optional parameters:
                - context: Additional context for the question

        Returns:
            Formatted prompt
        """
        context = kwargs.get("context", "")

        if context:
            return f"""Context: {context}

Question: {input_text}

Provide a clear, helpful answer with examples if relevant."""
        else:
            return f"""Question: {input_text}

Provide a clear, helpful answer with examples if relevant."""

    def execute(self, input_text: str, **kwargs) -> str:
        """Execute Q&A.

        Args:
            input_text: Question to answer
            **kwargs: Additional parameters (context, etc.)

        Returns:
            Answer to the question
        """
        system_prompt = self.get_system_prompt()
        user_prompt = self.format_prompt(input_text, **kwargs)

        output = self.model.generate(
            prompt=user_prompt, system_prompt=system_prompt
        )

        return self.post_process(output)

    def post_process(self, output: str) -> str:
        """Post-process the answer.

        Args:
            output: Raw model output

        Returns:
            Cleaned answer
        """
        return output.strip()
