"""Tests for configuration management."""

import tempfile
from pathlib import Path

import pytest
import yaml

from src.ibert.config import Config, ModelConfig, load_config, save_config


class TestModelConfig:
    """Tests for ModelConfig class."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ModelConfig()
        assert config.provider == "mistral"
        assert config.model_name == "mistralai/Mistral-7B-Instruct-v0.3"
        assert config.temperature == 0.2
        assert config.max_tokens == 2048
        assert config.device == "auto"
        assert config.load_in_8bit == False
        assert config.cache_dir == ".cache"

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ModelConfig(
            provider="openai",
            model_name="gpt-4",
            temperature=0.5,
            max_tokens=4096,
        )
        assert config.provider == "openai"
        assert config.model_name == "gpt-4"
        assert config.temperature == 0.5
        assert config.max_tokens == 4096


class TestConfig:
    """Tests for Config class."""

    def test_default_values(self):
        """Test default configuration values."""
        config = Config()
        assert config.model.provider == "mistral"
        assert config.data_dir == Path("data")
        assert config.cache_dir == Path(".cache")
        assert config.log_level == "INFO"

    def test_from_dict(self):
        """Test creating Config from dictionary."""
        data = {
            "model": {
                "provider": "mistral",
                "model_name": "mistral-large-latest",
                "temperature": 0.3,
            },
            "data_dir": "my_data",
            "log_level": "DEBUG",
        }
        config = Config.from_dict(data)
        assert config.model.provider == "mistral"
        assert config.model.model_name == "mistral-large-latest"
        assert config.model.temperature == 0.3
        assert config.data_dir == Path("my_data")
        assert config.log_level == "DEBUG"

    def test_to_dict(self):
        """Test converting Config to dictionary."""
        config = Config()
        data = config.to_dict()
        assert "model" in data
        assert "data_dir" in data
        assert "cache_dir" in data
        assert "log_level" in data
        assert data["model"]["provider"] == "mistral"


class TestConfigIO:
    """Tests for config loading and saving."""

    def test_load_nonexistent_config(self):
        """Test loading when config file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "nonexistent.yaml"
            config = load_config(config_path)
            # Should return default config
            assert config.model.provider == "mistral"

    def test_save_and_load_config(self):
        """Test saving and loading config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"

            # Create and save config
            original = Config()
            original.model.temperature = 0.7
            original.log_level = "DEBUG"
            save_config(original, config_path)

            # Load it back
            loaded = load_config(config_path)
            assert loaded.model.temperature == 0.7
            assert loaded.log_level == "DEBUG"

    def test_load_invalid_yaml(self):
        """Test loading invalid YAML returns default config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            with open(config_path, "w") as f:
                f.write("")  # Empty file

            config = load_config(config_path)
            # Should return default config
            assert config.model.provider == "mistral"
