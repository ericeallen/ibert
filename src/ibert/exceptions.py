"""Custom exception hierarchy for iBERT.

This module defines a comprehensive exception hierarchy for better
error handling, debugging, and user feedback throughout the iBERT system.
"""


class IBERTError(Exception):
    """Base exception for all iBERT errors.

    All iBERT-specific exceptions should inherit from this class.
    This allows catching all iBERT errors with a single except clause.
    """

    pass


# =============================================================================
# Model Errors
# =============================================================================


class ModelError(IBERTError):
    """Base class for model-related errors."""

    pass


class ModelLoadError(ModelError):
    """Failed to load or initialize model.

    Examples
    --------
    >>> raise ModelLoadError("Failed to download model weights")
    """

    pass


class ModelInferenceError(ModelError):
    """Error during model inference/generation.

    Examples
    --------
    >>> raise ModelInferenceError("CUDA out of memory during generation")
    """

    pass


class ModelConfigError(ModelError):
    """Invalid model configuration.

    Examples
    --------
    >>> raise ModelConfigError("Missing required parameter: model_name")
    """

    pass


class ModelTimeoutError(ModelError):
    """Model inference timed out.

    Examples
    --------
    >>> raise ModelTimeoutError("Generation exceeded 60 second timeout")
    """

    pass


# =============================================================================
# Data Generation Errors
# =============================================================================


class DataGenerationError(IBERTError):
    """Base class for data generation errors."""

    pass


class TemplateError(DataGenerationError):
    """Invalid template or template loading error.

    Examples
    --------
    >>> raise TemplateError("Template missing required field: sql_template")
    """

    pass


class TemplateLoadError(TemplateError):
    """Failed to load template file.

    Examples
    --------
    >>> raise TemplateLoadError("Template file not found: templates/joins.yaml")
    """

    pass


class TemplateValidationError(TemplateError):
    """Template failed validation.

    Examples
    --------
    >>> raise TemplateValidationError("SQL template contains syntax errors")
    """

    pass


class ValidationError(DataGenerationError):
    """Data validation failed.

    Examples
    --------
    >>> raise ValidationError("SQL and Ibis results don't match")
    """

    pass


class SQLExecutionError(ValidationError):
    """SQL execution failed during validation.

    Examples
    --------
    >>> raise SQLExecutionError("DuckDB error: table 'users' not found")
    """

    pass


class IbisExecutionError(ValidationError):
    """Ibis code execution failed during validation.

    Examples
    --------
    >>> raise IbisExecutionError("Ibis error: column 'age' not found")
    """

    pass


class ResultMismatchError(ValidationError):
    """SQL and Ibis results don't match.

    Examples
    --------
    >>> raise ResultMismatchError("Expected 10 rows, got 8")
    """

    pass


class MiningError(DataGenerationError):
    """Repository mining error.

    Examples
    --------
    >>> raise MiningError("Failed to clone repository: github.com/...")
    """

    pass


class RepositoryCloneError(MiningError):
    """Failed to clone repository.

    Examples
    --------
    >>> raise RepositoryCloneError("Git clone failed: connection refused")
    """

    pass


class CodeExtractionError(MiningError):
    """Failed to extract code patterns from repository.

    Examples
    --------
    >>> raise CodeExtractionError("AST parsing failed for file: api.py")
    """

    pass


class AugmentationError(DataGenerationError):
    """Data augmentation error.

    Examples
    --------
    >>> raise AugmentationError("Column substitution failed: no valid alternatives")
    """

    pass


# =============================================================================
# Task Execution Errors
# =============================================================================


class TaskError(IBERTError):
    """Base class for task execution errors."""

    pass


class TaskConfigError(TaskError):
    """Invalid task configuration.

    Examples
    --------
    >>> raise TaskConfigError("Unknown task type: invalid_task")
    """

    pass


class TaskExecutionError(TaskError):
    """Task execution failed.

    Examples
    --------
    >>> raise TaskExecutionError("Code completion failed: model returned empty response")
    """

    pass


class TaskInputError(TaskError):
    """Invalid input provided to task.

    Examples
    --------
    >>> raise TaskInputError("Input too long: 10000 tokens (max: 4096)")
    """

    pass


class TaskTimeoutError(TaskError):
    """Task execution timed out.

    Examples
    --------
    >>> raise TaskTimeoutError("Task exceeded 30 second timeout")
    """

    pass


# =============================================================================
# Configuration Errors
# =============================================================================


class ConfigError(IBERTError):
    """Base class for configuration errors."""

    pass


class ConfigLoadError(ConfigError):
    """Failed to load configuration file.

    Examples
    --------
    >>> raise ConfigLoadError("Config file not found: config.yaml")
    """

    pass


class ConfigValidationError(ConfigError):
    """Configuration validation failed.

    Examples
    --------
    >>> raise ConfigValidationError("Invalid temperature: -0.5 (must be >= 0)")
    """

    pass


# =============================================================================
# CLI Errors
# =============================================================================


class CLIError(IBERTError):
    """Base class for CLI errors."""

    pass


class InputError(CLIError):
    """Invalid input provided to CLI.

    Examples
    --------
    >>> raise InputError("No input provided (expected file or stdin)")
    """

    pass


class OutputError(CLIError):
    """Failed to write output.

    Examples
    --------
    >>> raise OutputError("Failed to write to output file: permission denied")
    """

    pass


# =============================================================================
# I/O Errors
# =============================================================================


class IOError(IBERTError):
    """Base class for I/O errors."""

    pass


class FileReadError(IOError):
    """Failed to read file.

    Examples
    --------
    >>> raise FileReadError("Cannot read file: data.jsonl (not found)")
    """

    pass


class FileWriteError(IOError):
    """Failed to write file.

    Examples
    --------
    >>> raise FileWriteError("Cannot write file: output.jsonl (permission denied)")
    """

    pass


class JSONDecodeError(IOError):
    """Failed to decode JSON.

    Examples
    --------
    >>> raise JSONDecodeError("Invalid JSON on line 42: unexpected token")
    """

    pass


class YAMLDecodeError(IOError):
    """Failed to decode YAML.

    Examples
    --------
    >>> raise YAMLDecodeError("Invalid YAML: indentation error on line 15")
    """

    pass
