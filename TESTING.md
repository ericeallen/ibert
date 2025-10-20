# Testing iBERT

## Running Tests

### All Tests

```bash
# Run all iBERT tests
just test

# Or with pytest directly
PYTHONPATH=. .venv/bin/python -m pytest tests/ibert/ -v
```

### Specific Test Categories

```bash
# Config tests only
pytest tests/ibert/test_config.py -v

# Model tests only
pytest tests/ibert/models/ -v

# Task tests only
pytest tests/ibert/tasks/ -v

# Specific task
pytest tests/ibert/tasks/test_code_completion.py -v
```

## Test Performance

All tests run **very quickly** (< 0.3s) because they use:

- **Mock models** for task tests
- **Lazy loading** for model factory tests
- **No actual model downloads** during testing

## Lazy Loading for Tests

The MistralModel supports a `lazy_load` parameter specifically for testing:

```python
# In tests - model is created but not loaded
model_config = {
    "model_name": "mistralai/Mistral-7B-Instruct-v0.3",
    "lazy_load": True,  # Prevents downloading 14GB model!
}
model = MistralModel(model_config)
```

This allows tests to:
- ✅ Verify configuration is passed correctly
- ✅ Test model initialization logic
- ✅ Run in < 1 second
- ❌ Not download 14GB models
- ❌ Not use GPU/CPU for inference

## Test Coverage

Current test suite: **46 tests, 100% passing**

### Configuration Tests (8 tests)
- Default values
- Custom values
- Dictionary serialization
- File I/O operations

### Model Tests (3 tests)
- Model factory creation
- Configuration passing
- Provider validation

### Task Tests (35 tests)
Six test suites, one per task:

1. **Code Completion** (6 tests)
   - System prompt generation
   - Prompt formatting
   - Execution
   - Post-processing

2. **Ibis→SQL** (6 tests)
   - Dialect support
   - Table name injection
   - Translation accuracy

3. **SQL→Ibis** (6 tests)
   - Schema handling
   - Import generation
   - Code formatting

4. **Error Resolution** (6 tests)
   - Error message handling
   - Context integration
   - Fix validation

5. **Q&A** (5 tests)
   - Question formatting
   - Context handling
   - Answer generation

6. **Documentation** (6 tests)
   - Style selection (Google/NumPy)
   - Example inclusion
   - Code formatting

## Writing New Tests

### Testing Tasks

Task tests use a `MockModel` that doesn't require actual inference:

```python
def test_my_task(mock_model_with_response):
    # Create mock that returns specific response
    model = mock_model_with_response("expected output")
    task = MyTask(model)

    result = task.execute("input")

    assert "expected output" in result
```

### Testing Models

Model tests use `lazy_load` to avoid downloading:

```python
def test_my_model():
    config = {
        "model_name": "mistralai/Mistral-7B-Instruct-v0.3",
        "lazy_load": True,  # Important!
    }

    model = MistralModel(config)

    # Test configuration
    assert model.model_name == "mistralai/Mistral-7B-Instruct-v0.3"

    # Note: Can't test actual generation without loading model
```

### Testing Configuration

Configuration tests are straightforward:

```python
def test_config():
    config = Config()
    config.model.temperature = 0.5

    # Test serialization
    data = config.to_dict()
    assert data["model"]["temperature"] == 0.5

    # Test deserialization
    loaded = Config.from_dict(data)
    assert loaded.model.temperature == 0.5
```

## Integration Testing

For **real inference testing** (not in CI), you can:

```python
# Manual test - actually loads model
def test_real_inference():
    config = {
        "model_name": "mistralai/Mistral-7B-Instruct-v0.3",
        "lazy_load": False,  # Actually load the model
        "device": "cpu",
    }

    model = MistralModel(config)
    result = model.generate("What is Ibis?")

    assert len(result) > 0
    print(result)
```

**Warning:** This will:
- Download ~14GB on first run
- Use significant CPU/GPU resources
- Take 30-60 seconds to complete

## Continuous Integration

For CI environments (GitHub Actions, etc.):

```yaml
# .github/workflows/test.yml
- name: Run tests
  run: |
    pytest tests/ibert/ -v
  # Fast! No model downloads needed
```

## Test Fixtures

### Mock Model (`conftest.py`)

```python
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
```

Usage:
```python
def test_with_mock(mock_model):
    task = MyTask(mock_model)
    # model.generate() returns "mock response"

def test_with_custom_response(mock_model_with_response):
    model = mock_model_with_response("custom output")
    task = MyTask(model)
    # model.generate() returns "custom output"
```

## Debugging Tests

### Run with verbose output

```bash
pytest tests/ibert/ -vv -s
```

### Run specific test

```bash
pytest tests/ibert/tasks/test_code_completion.py::TestCodeCompletionTask::test_execute -v
```

### See print statements

```bash
pytest tests/ibert/ -s
```

### Run with debugger

```bash
pytest tests/ibert/ --pdb
```

## Common Issues

### Tests Hang

**Problem:** Tests take forever or hang

**Cause:** Model is actually being downloaded/loaded

**Solution:** Ensure tests use `lazy_load=True` or `MockModel`

```python
# Bad - will hang
model = MistralModel({"model_name": "mistralai/Mistral-7B-Instruct-v0.3"})

# Good - fast
model = MistralModel({"model_name": "...", "lazy_load": True})
```

### Import Errors

**Problem:** `ModuleNotFoundError`

**Solution:** Set PYTHONPATH

```bash
PYTHONPATH=. pytest tests/ibert/ -v
# Or use just
just test
```

### Slow Tests

**Problem:** Tests take > 1 second

**Check:**
- Are you using `lazy_load=True`?
- Are you using `MockModel` for task tests?
- Are you accidentally loading the real model?

## Best Practices

1. ✅ Use `MockModel` for task tests
2. ✅ Use `lazy_load=True` for model tests
3. ✅ Keep tests fast (< 0.5s total)
4. ✅ Test behavior, not implementation
5. ✅ Use fixtures for common setup
6. ❌ Don't download models in tests
7. ❌ Don't require GPU for tests
8. ❌ Don't test actual inference in unit tests

## Summary

The iBERT test suite is designed to be:
- **Fast** - All tests run in < 0.3s
- **Isolated** - No external dependencies
- **Deterministic** - Same results every time
- **CI-friendly** - Works without GPU or large downloads
- **Comprehensive** - 46 tests covering all components

For actual model testing, use manual integration tests or end-to-end testing scripts outside the main test suite.
