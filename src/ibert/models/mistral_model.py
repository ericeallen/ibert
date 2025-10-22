"""Mistral model implementation using local inference."""

from typing import Any

from .base import BaseModel


class MistralModel(BaseModel):
    """Mistral model implementation using local inference with transformers.

    This baseline implementation downloads and runs a small Mistral model locally.
    Recommended model: Mistral-7B-Instruct-v0.3 (7B parameters)

    Requirements:
        pip install transformers torch accelerate
    """

    def __init__(self, config: dict[str, Any]):
        """Initialize local Mistral model.

        Args:
            config: Configuration dict with keys:
                - model_name: HuggingFace model ID (default: mistralai/Mistral-7B-Instruct-v0.3)
                - temperature: Sampling temperature (default: 0.2)
                - max_tokens: Maximum tokens to generate (default: 2048)
                - device: Device to run on (default: auto - uses GPU if available)
                - load_in_8bit: Load model in 8-bit for lower memory (default: False)
                - cache_dir: Directory to cache downloaded models (default: .cache)
                - lazy_load: If True, don't load model until first use (default: False, for testing)
        """
        super().__init__(config)
        self._model_name: str = str(config.get("model_name", "mistralai/Mistral-7B-Instruct-v0.3"))
        self._temperature: float = float(config.get("temperature", 0.2))
        self._max_tokens: int = int(config.get("max_tokens", 2048))
        self._device: str = str(config.get("device", "auto"))
        self._load_in_8bit: bool = bool(config.get("load_in_8bit", False))
        self._cache_dir: str = str(config.get("cache_dir", ".cache"))
        self._lazy_load: bool = bool(config.get("lazy_load", False))

        self._model = None
        self._tokenizer = None

        # Only initialize if not lazy loading (lazy_load is for testing)
        if not self._lazy_load:
            self._initialize_model()

    def _initialize_model(self):
        """Initialize the local Mistral model and tokenizer."""
        import sys
        from pathlib import Path

        try:
            import torch  # noqa: F401
            from transformers import AutoModelForCausalLM, AutoTokenizer

            # Check if model is already cached
            cache_path = Path(self._cache_dir)
            model_cache = cache_path / "models--mistralai--Mistral-7B-Instruct-v0.3"
            is_cached = model_cache.exists()

            if not is_cached:
                print(
                    "\n╭─────────────────────────────────────────────────────────────╮",
                    file=sys.stderr,
                )
                print(
                    "│  First-time model download (~14GB)                         │",
                    file=sys.stderr,
                )
                print(
                    "│  This will take 5-10 minutes depending on your connection  │",
                    file=sys.stderr,
                )
                print(
                    "│  Subsequent runs will be much faster!                      │",
                    file=sys.stderr,
                )
                print(
                    "╰─────────────────────────────────────────────────────────────╯\n",
                    file=sys.stderr,
                )

            # Load tokenizer
            print(f"Loading tokenizer for {self._model_name}...", file=sys.stderr)
            # Safe: model_name is from config, not user input - using trusted model registry
            self._tokenizer = AutoTokenizer.from_pretrained(  # nosec B615
                self._model_name, cache_dir=self._cache_dir
            )

            # Set padding token if not set
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            # Load model with appropriate settings
            if not is_cached:
                print(
                    f"\nDownloading model to {self._cache_dir}/...",
                    file=sys.stderr,
                )
                print("This may take several minutes...\n", file=sys.stderr)
            else:
                print("Loading model from cache...", file=sys.stderr)

            # Special handling for MPS (Apple Silicon) to avoid memory allocation issues
            if self._device == "mps":
                print("Loading model to CPU first, then moving to MPS...", file=sys.stderr)
                load_kwargs = {
                    "cache_dir": self._cache_dir,
                    "low_cpu_mem_usage": True,
                    "torch_dtype": "auto",
                }
            else:
                load_kwargs = {
                    "cache_dir": self._cache_dir,
                    "device_map": self._device,
                    "low_cpu_mem_usage": True,
                }

            if self._load_in_8bit:
                load_kwargs["load_in_8bit"] = True
                print("Using 8-bit quantization for lower memory", file=sys.stderr)

            # Safe: model_name is from config, not user input - using trusted model registry
            self._model = AutoModelForCausalLM.from_pretrained(self._model_name, **load_kwargs)  # nosec B615

            # Move to MPS if requested
            if self._device == "mps" and not self._load_in_8bit:
                print("Moving model to MPS device...", file=sys.stderr)
                self._model = self._model.to("mps")

            # Set to evaluation mode
            self._model.eval()

            # Show success message
            device_info = f"on {self._device}" if self._device != "auto" else ""
            if hasattr(self._model, "device"):
                device_info = f"on {self._model.device}"

            print(f"\n✓ Model loaded successfully {device_info}\n", file=sys.stderr)

        except ImportError as e:
            # Model will remain None, generate methods will raise helpful error
            missing_package = str(e).split("'")[1] if "'" in str(e) else "required packages"
            self._import_error = (
                f"Missing dependency: {missing_package}. "
                "Install with: pip install transformers torch accelerate"
            )
            print(f"\nImportError during model initialization: {e}", file=sys.stderr)
        except Exception as e:
            # Catch any other errors during model loading
            self._import_error = (
                f"Error loading model: {e}\n"
                "This may be due to network issues, insufficient memory, or corrupted cache."
            )
            print(f"\nError during model initialization: {type(e).__name__}: {e}", file=sys.stderr)

    def _ensure_model(self):
        """Ensure model is initialized, raise helpful error if not."""
        if self._model is None or self._tokenizer is None:
            error_msg = getattr(
                self,
                "_import_error",
                "Model not initialized. Install with: pip install transformers torch accelerate",
            )
            raise RuntimeError(error_msg)

    def _format_messages(self, messages: list[dict[str, str]]) -> str:
        """Format messages into Mistral instruction format.

        Args:
            messages: List of message dicts with 'role' and 'content' keys

        Returns:
            Formatted prompt string for Mistral
        """
        # Mistral instruction format:
        # <s>[INST] instruction [/INST] model answer</s>[INST] instruction [/INST]

        formatted = ""
        for i, msg in enumerate(messages):
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                # System messages are incorporated into first user message
                if i == 0 and len(messages) > 1 and messages[1]["role"] == "user":
                    continue  # Will be handled with next message
                formatted += f"[INST] {content} [/INST]\n"
            elif role == "user":
                # Check if there's a preceding system message
                system_content = ""
                if i > 0 and messages[i - 1]["role"] == "system":
                    system_content = messages[i - 1]["content"] + "\n\n"
                formatted += f"[INST] {system_content}{content} [/INST]"
            elif role == "assistant":
                formatted += f" {content}</s>"

        return formatted

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
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        return self.generate_chat(messages, temperature, max_tokens)

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
        self._ensure_model()

        import torch

        temp = temperature if temperature is not None else self._temperature
        max_tok = max_tokens if max_tokens is not None else self._max_tokens

        # Format messages for Mistral
        formatted_prompt = self._format_messages(messages)

        # Tokenize (tokenizer is guaranteed to be non-None by _ensure_model)
        # Safe: type assertion for type checker - _ensure_model guarantees non-None
        assert self._tokenizer is not None  # nosec B101
        inputs = self._tokenizer(formatted_prompt, return_tensors="pt", padding=True)

        # Move to same device as model
        # Safe: type assertion for type checker - _ensure_model guarantees non-None
        assert self._model is not None  # nosec B101
        if hasattr(self._model, "device"):
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tok,
                temperature=temp,
                do_sample=temp > 0,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )

        # Decode only the new tokens (not the input)
        generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        response = self._tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Ensure response is a string
        return str(response).strip()

    @property
    def model_name(self) -> str:
        """Return the name/identifier of the model."""
        return self._model_name

    @property
    def supports_training(self) -> bool:
        """Mistral baseline does not support training.

        Returns:
            False (inference-only model, no training support in baseline)
        """
        return False
