"""Tests for Q&A task."""

import pytest

from src.ibert.tasks import QATask


class TestQATask:
    """Tests for QATask."""

    def test_system_prompt(self, mock_model):
        """Test system prompt generation."""
        task = QATask(mock_model)
        prompt = task.get_system_prompt()
        assert "question" in prompt.lower()
        assert "ibis" in prompt.lower()

    def test_format_prompt_simple(self, mock_model):
        """Test simple prompt formatting."""
        task = QATask(mock_model)
        question = "What is lazy evaluation in Ibis?"
        prompt = task.format_prompt(question)
        assert question in prompt

    def test_format_prompt_with_context(self, mock_model):
        """Test prompt formatting with context."""
        task = QATask(mock_model)
        question = "How do I use window functions?"
        prompt = task.format_prompt(
            question,
            context="I'm working with a sales table"
        )
        assert question in prompt
        assert "sales table" in prompt

    def test_execute(self, mock_model_with_response):
        """Test executing Q&A."""
        answer = "Lazy evaluation means queries are not executed until needed."
        model = mock_model_with_response(answer)
        task = QATask(model)
        result = task.execute("What is lazy evaluation in Ibis?")
        assert answer in result

    def test_post_process(self, mock_model):
        """Test post-processing."""
        task = QATask(mock_model)
        output = "  This is an answer.  \n"
        result = task.post_process(output)
        assert result == "This is an answer."
