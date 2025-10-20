"""Configuration management for iBERT system."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class ModelConfig:
    """Configuration for model selection and parameters."""

    provider: str = "mistral"  # mistral (local), openai, anthropic, etc.
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.3"  # HuggingFace model ID
    temperature: float = 0.2
    max_tokens: int = 2048
    device: str = "auto"  # auto, cpu, cuda, mps
    load_in_8bit: bool = False  # Load in 8-bit for lower memory
    cache_dir: str = ".cache"  # Directory for model cache
    api_key: Optional[str] = None  # For API-based providers
    base_url: Optional[str] = None  # For custom API endpoints

    def __post_init__(self):
        """Load API key from environment if not provided."""
        if self.api_key is None and self.provider == "openai":
            self.api_key = os.getenv("OPENAI_API_KEY")
        elif self.api_key is None and self.provider == "anthropic":
            self.api_key = os.getenv("ANTHROPIC_API_KEY")


@dataclass
class Config:
    """Main configuration for iBERT system."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data_dir: Path = field(default_factory=lambda: Path("data"))
    cache_dir: Path = field(default_factory=lambda: Path(".cache"))
    log_level: str = "INFO"

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        """Create Config from dictionary."""
        model_data = data.get("model", {})
        model = ModelConfig(**model_data)

        return cls(
            model=model,
            data_dir=Path(data.get("data_dir", "data")),
            cache_dir=Path(data.get("cache_dir", ".cache")),
            log_level=data.get("log_level", "INFO"),
        )

    def to_dict(self) -> dict:
        """Convert Config to dictionary."""
        return {
            "model": {
                "provider": self.model.provider,
                "model_name": self.model.model_name,
                "temperature": self.model.temperature,
                "max_tokens": self.model.max_tokens,
                "device": self.model.device,
                "load_in_8bit": self.model.load_in_8bit,
                "cache_dir": self.model.cache_dir,
                "base_url": self.model.base_url,
            },
            "data_dir": str(self.data_dir),
            "cache_dir": str(self.cache_dir),
            "log_level": self.log_level,
        }


def load_config(config_path: Optional[Path] = None) -> Config:
    """Load configuration from file or use defaults.

    Args:
        config_path: Path to YAML config file. If None, looks for config.yaml
                    in current directory or uses defaults.

    Returns:
        Config object
    """
    if config_path is None:
        config_path = Path("config.yaml")

    if config_path.exists():
        with open(config_path) as f:
            data = yaml.safe_load(f)
            return Config.from_dict(data or {})

    return Config()


def save_config(config: Config, config_path: Path = Path("config.yaml")):
    """Save configuration to YAML file.

    Args:
        config: Config object to save
        config_path: Path where to save the config file
    """
    with open(config_path, "w") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False)
