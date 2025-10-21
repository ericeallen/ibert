# Contributing to iBERT

Thank you for your interest in contributing to iBERT! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Development Setup](#development-setup)
- [Code Quality Standards](#code-quality-standards)
- [Testing Requirements](#testing-requirements)
- [Pull Request Process](#pull-request-process)
- [Coding Conventions](#coding-conventions)

## Development Setup

### Prerequisites

- Python 3.13+
- Git
- 16GB+ RAM (for local model testing)
- Optional: NVIDIA GPU or Apple Silicon for faster model inference

### Installation Steps

1. **Fork and clone the repository**

```bash
git clone https://github.com/your-username/ibert.git
cd ibert
```

2. **Create and activate virtual environment**

```bash
python3.13 -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows
```

3. **Install in editable mode with dev dependencies**

```bash
pip install -e .[dev]
```

4. **Install pre-commit hooks**

```bash
pre-commit install
```

This will automatically run code quality checks before each commit.

5. **Verify installation**

```bash
# Run tests
just test

# Or without just
PYTHONPATH=. .venv/bin/python -m pytest tests/
```

## Code Quality Standards

### Formatting and Linting

We use strict code quality tools to ensure consistency:

- **Black** - Code formatting (100 character line length)
- **Ruff** - Fast Python linter
- **isort** - Import sorting
- **mypy** - Static type checking
- **Bandit** - Security vulnerability scanning
- **pydocstyle** - Docstring linting

### Running Code Quality Checks

```bash
# Run all checks (happens automatically on commit with pre-commit hooks)
pre-commit run --all-files

# Run individual tools
black src/ tests/
ruff check src/ tests/
mypy src/
```

### Type Hints

**All public functions must have type hints:**

```python
from typing import Dict, List, Optional, Tuple

def process_examples(
    examples: List[Dict[str, Any]],
    max_count: Optional[int] = None
) -> Tuple[int, int]:
    """Process training examples.

    Parameters
    ----------
    examples : list of dict
        Training examples to process
    max_count : int, optional
        Maximum examples to process

    Returns
    -------
    processed : int
        Number of examples processed
    failed : int
        Number of examples that failed
    """
    ...
```

### Docstrings

**All functions must have NumPy-style docstrings:**

```python
def validate_example(example: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validate a training example.

    Parameters
    ----------
    example : dict
        Training example with input/target fields

    Returns
    -------
    is_valid : bool
        True if example passes validation
    error_message : str or None
        Error message if validation failed, None otherwise

    Examples
    --------
    >>> example = {"task": "qa", "input": {...}, "target": {...}}
    >>> valid, error = validate_example(example)
    >>> assert valid
    """
    ...
```

### Logging

Use the centralized logging framework instead of print statements:

```python
from ibert.logging import get_logger

log = get_logger(__name__)

def my_function():
    log.info("Processing started")
    try:
        # ... work
        log.debug("Intermediate result: %s", result)
    except Exception as e:
        log.error("Processing failed: %s", str(e), exc_info=True)
        raise
```

### Exception Handling

Use custom exceptions from `ibert.exceptions`:

```python
from ibert.exceptions import ModelLoadError, ValidationError

# Instead of:
raise ValueError("Model not found")

# Use:
raise ModelLoadError("Failed to load model: model_name not found in config")
```

## Testing Requirements

### Coverage Requirements

- **Minimum coverage**: 95% for new code
- **Current coverage**: 81% (we're working to improve this)
- All new features must include tests

### Writing Tests

```python
"""Tests for my_module.py"""

import pytest
from src.my_module import MyClass

class TestMyClass:
    """Test suite for MyClass."""

    @pytest.fixture
    def instance(self):
        """Create test instance."""
        return MyClass()

    def test_basic_functionality(self, instance):
        """Test basic use case."""
        result = instance.method()
        assert result == expected

    def test_error_handling(self, instance):
        """Test error case."""
        with pytest.raises(ValueError):
            instance.method(invalid_input)

    @pytest.mark.parametrize("input,expected", [
        ("test1", "result1"),
        ("test2", "result2"),
    ])
    def test_multiple_cases(self, instance, input, expected):
        """Test multiple input scenarios."""
        result = instance.method(input)
        assert result == expected
```

### Running Tests

```bash
# Run all tests
just test

# Run specific test file
PYTHONPATH=. .venv/bin/python -m pytest tests/datagen/test_concatenate_datasets.py -v

# Run with coverage
PYTHONPATH=. .venv/bin/python -m pytest tests/ --cov=src --cov-report=html

# Run tests matching a pattern
PYTHONPATH=. .venv/bin/python -m pytest tests/ -k "test_validate"
```

## Pull Request Process

### Before Submitting

1. **Create a feature branch**

```bash
git checkout -b feature/amazing-feature
```

2. **Make your changes**

- Follow code style guidelines
- Add tests for new functionality
- Update documentation

3. **Run all checks**

```bash
# Tests must pass
just test

# Coverage must be maintained
PYTHONPATH=. .venv/bin/python -m pytest tests/ --cov=src

# Pre-commit hooks must pass
pre-commit run --all-files
```

4. **Commit your changes**

```bash
git add .
git commit -m "Add amazing feature"

# Pre-commit hooks will run automatically
# Fix any issues they find
```

5. **Push to your fork**

```bash
git push origin feature/amazing-feature
```

### Pull Request Guidelines

- **Title**: Clear, descriptive title (e.g., "Add logging framework for observability")
- **Description**: Explain what and why, not how
- **Tests**: Include tests for new features
- **Documentation**: Update relevant documentation
- **Changelog**: Add entry to CHANGELOG.md (if exists)

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] All tests pass
- [ ] New tests added for new functionality
- [ ] Coverage maintained or improved

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] No new warnings generated
```

## Coding Conventions

### Code Style

- **Line length**: 100 characters (enforced by Black)
- **Indentation**: 4 spaces (no tabs)
- **Quotes**: Double quotes for strings, single for characters
- **Imports**: Sorted alphabetically, grouped (stdlib, third-party, local)

### Naming Conventions

- **Functions/variables**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private**: Prefix with underscore (`_private_method`)

```python
# Good
class DataGenerator:
    DEFAULT_BATCH_SIZE = 100

    def __init__(self):
        self._cache = {}

    def generate_examples(self, count: int) -> List[Dict]:
        ...
```

### File Organization

```python
"""Module docstring explaining purpose."""

# Standard library imports
import os
import sys
from pathlib import Path

# Third-party imports
import pandas as pd
import torch
from transformers import AutoModel

# Local imports
from ibert.logging import get_logger
from ibert.exceptions import DataGenerationError

# Constants
MAX_RETRIES = 3
DEFAULT_TIMEOUT = 30

# Module-level setup
log = get_logger(__name__)

# Classes and functions
class MyClass:
    ...
```

### Error Messages

- Be specific and actionable
- Include context and suggestions

```python
# Bad
raise ValueError("Invalid input")

# Good
raise ValidationError(
    f"Invalid SQL query on line {line_num}: missing FROM clause. "
    f"Expected format: SELECT ... FROM table WHERE ..."
)
```

## Project Structure

```
ibert/
├── src/
│   └── ibert/              # Main package
│       ├── models/         # Model implementations
│       ├── tasks/          # Task handlers
│       ├── config/         # Configuration
│       ├── logging.py      # Logging framework
│       └── exceptions.py   # Custom exceptions
├── tests/                  # Test suite
├── data/                   # Generated datasets
├── bin/                    # CLI scripts
├── .github/
│   └── workflows/          # GitHub Actions CI/CD
├── pyproject.toml          # Project configuration
├── requirements.txt        # Dependencies
└── requirements-dev.txt    # Dev dependencies
```

## Getting Help

- **Documentation**: See [DOCUMENTATION.md](DOCUMENTATION.md)
- **Issues**: Check [GitHub Issues](https://github.com/yourusername/ibert/issues)
- **Discussions**: Use [GitHub Discussions](https://github.com/yourusername/ibert/discussions)

## License

By contributing to iBERT, you agree that your contributions will be licensed under the Apache License 2.0.
