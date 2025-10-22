"""Pytest fixtures for iBERT tests."""

import pytest

from src.ibert.models.base import BaseModel


class MockModel(BaseModel):
    """Mock model for testing."""

    def __init__(self, config=None, response="mock response"):
        super().__init__(config or {})
        self._response = response
        self._last_prompt = None
        self._last_system_prompt = None
        self._last_messages = None

    def generate(self, prompt, system_prompt=None, temperature=None, max_tokens=None):
        """Mock generate method."""
        self._last_prompt = prompt
        self._last_system_prompt = system_prompt
        return self._response

    def generate_chat(self, messages, temperature=None, max_tokens=None):
        """Mock chat generation method."""
        self._last_messages = messages
        return self._response

    @property
    def model_name(self):
        """Return mock model name."""
        return "mock-model"


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    return MockModel()


@pytest.fixture
def mock_model_with_response():
    """Create a mock model factory with custom response."""

    def _create_mock(response):
        return MockModel(response=response)

    return _create_mock
