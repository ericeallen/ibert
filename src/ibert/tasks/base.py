"""Base class for task handlers."""

from abc import ABC, abstractmethod
from typing import Optional

from ..models.base import BaseModel


class BaseTask(ABC):
    """Abstract base class for task handlers.

    Each task type (code completion, translation, etc.) implements this interface.
    """

    def __init__(self, model: BaseModel):
        """Initialize task with a model.

        Args:
            model: Language model to use for the task
        """
        self.model = model

    @abstractmethod
    def execute(self, input_text: str, **kwargs) -> str:
        """Execute the task on input text.

        Args:
            input_text: Input text to process
            **kwargs: Additional task-specific parameters

        Returns:
            Task output as string
        """
        pass

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get the system prompt for this task.

        Returns:
            System prompt string that defines the task behavior
        """
        pass

    def format_prompt(self, input_text: str, **kwargs) -> str:
        """Format the user prompt for this task.

        Args:
            input_text: Input text to process
            **kwargs: Additional task-specific parameters

        Returns:
            Formatted prompt string
        """
        return input_text

    def post_process(self, output: str) -> str:
        """Post-process the model output.

        Args:
            output: Raw model output

        Returns:
            Cleaned/processed output
        """
        return output.strip()
