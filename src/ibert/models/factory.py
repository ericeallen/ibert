"""Factory for creating model instances."""

from typing import Any, Dict

from ..config import Config
from .base import BaseModel
from .mistral_model import MistralModel


def create_model(config: Config) -> BaseModel:
    """Create a model instance based on configuration.

    Args:
        config: Configuration object

    Returns:
        Initialized model instance

    Raises:
        ValueError: If provider is not supported
    """
    provider = config.model.provider.lower()

    model_config = {
        "model_name": config.model.model_name,
        "temperature": config.model.temperature,
        "max_tokens": config.model.max_tokens,
        "device": config.model.device,
        "load_in_8bit": config.model.load_in_8bit,
        "cache_dir": config.model.cache_dir,
        "api_key": config.model.api_key,
        "base_url": config.model.base_url,
    }

    if provider in ("mistral", "huggingface"):
        # Both use the same MistralModel class (which is a HuggingFace transformers wrapper)
        return MistralModel(model_config)
    else:
        raise ValueError(
            f"Unsupported model provider: {provider}. "
            f"Supported providers: huggingface, mistral"
        )
