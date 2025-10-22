"""Abstract base class for language models."""

from abc import ABC, abstractmethod
from typing import Any


class BaseModel(ABC):
    """Abstract base class for language models used in iBERT.

    This interface allows easy swapping between different model providers
    (Mistral, OpenAI, Anthropic, local models, etc.) and future fine-tuned models.
    """

    def __init__(self, config: dict[str, Any]):
        """Initialize the model with configuration.

        Args:
            config: Dictionary containing model configuration parameters
        """
        self.config = config

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Generate text completion from a prompt.

        Args:
            prompt: The input prompt
            system_prompt: Optional system prompt to set context
            temperature: Sampling temperature (overrides config if provided)
            max_tokens: Maximum tokens to generate (overrides config if provided)

        Returns:
            Generated text completion
        """
        pass

    @abstractmethod
    def generate_chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Generate response from a chat conversation.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            temperature: Sampling temperature (overrides config if provided)
            max_tokens: Maximum tokens to generate (overrides config if provided)

        Returns:
            Generated response
        """
        pass

    def train(self, training_data: Any, **kwargs) -> None:  # noqa: B027
        """Train or fine-tune the model.

        For baseline models, this is a no-op. Subclasses for fine-tuned models
        should override this method.

        Args:
            training_data: Training data in appropriate format
            **kwargs: Additional training parameters
        """
        pass

    def save(self, path: str) -> None:  # noqa: B027
        """Save model weights/checkpoints.

        For API-based models, this is a no-op. Subclasses for local models
        should override this method.

        Args:
            path: Path where to save the model
        """
        pass

    def load(self, path: str) -> None:  # noqa: B027
        """Load model weights/checkpoints.

        For API-based models, this is a no-op. Subclasses for local models
        should override this method.

        Args:
            path: Path from where to load the model
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the name/identifier of the model."""
        pass

    @property
    def supports_training(self) -> bool:
        """Whether this model supports training/fine-tuning.

        Returns:
            True if model can be trained, False otherwise
        """
        return False
