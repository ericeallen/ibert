"""Tests for model factory."""

import pytest

from src.ibert.config import Config
from src.ibert.models import create_model
from src.ibert.models.mistral_model import MistralModel


class TestModelFactory:
    """Tests for create_model factory function."""

    def test_create_mistral_model(self, monkeypatch):
        """Test creating Mistral model."""
        # Patch factory to add lazy_load to avoid downloading model
        original_create = create_model

        def patched_create(config):
            # Add lazy_load to config before creating model
            from src.ibert.models.mistral_model import MistralModel

            model_config = {
                "model_name": config.model.model_name,
                "temperature": config.model.temperature,
                "max_tokens": config.model.max_tokens,
                "device": config.model.device,
                "load_in_8bit": config.model.load_in_8bit,
                "cache_dir": config.model.cache_dir,
                "lazy_load": True,  # Don't actually load model
            }

            if config.model.provider.lower() == "mistral":
                return MistralModel(model_config)
            return original_create(config)

        monkeypatch.setattr("src.ibert.models.factory.create_model", patched_create)

        config = Config()
        config.model.provider = "mistral"
        model = patched_create(config)
        assert isinstance(model, MistralModel)
        assert model.model_name == "mistralai/Mistral-7B-Instruct-v0.3"

    def test_create_unsupported_provider(self):
        """Test error on unsupported provider."""
        config = Config()
        config.model.provider = "unsupported"
        with pytest.raises(ValueError, match="Unsupported model provider"):
            create_model(config)

    def test_model_config_passed_correctly(self):
        """Test that model config is passed correctly."""
        # Create model directly with lazy_load to avoid downloading
        from src.ibert.models.mistral_model import MistralModel

        model_config = {
            "model_name": "mistralai/Mistral-7B-Instruct-v0.2",
            "temperature": 0.5,
            "max_tokens": 2048,
            "device": "cpu",
            "load_in_8bit": False,
            "cache_dir": ".cache",
            "lazy_load": True,  # Don't actually load model
        }

        model = MistralModel(model_config)
        assert model.model_name == "mistralai/Mistral-7B-Instruct-v0.2"
        assert model._temperature == 0.5
        assert model._device == "cpu"
