# iBERT: Multi-Task Code Generation for Lazy-Evaluated DSLs

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-443%20passing-green.svg)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-81%25-brightgreen.svg)](htmlcov/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **📖 [Complete Documentation](DOCUMENTATION.md)** - Comprehensive guide with table of contents
> **🚀 [Quick Start](QUICKSTART.md)** - Get started in 5 minutes
> **⚙️ [Model Setup](LOCAL_MODEL_SETUP.md)** - Local inference configuration
> **🧪 [Testing Guide](TESTING.md)** - Testing and coverage details

## 🎯 Overview

iBERT (inspired by BERT, but not a BERT architecture) is a comprehensive training data generation system and model-training system for SQL→Ibis code translation. Currently, we have implemented the training data generation component. We are actively working on the model generation component. The project creates high-quality, validated training examples by:

- **Generating** synthetic SQL→Ibis pairs from parameterized templates
- **Mining** real-world examples from GitHub repositories and documentation
- **Validating** all examples through compiler/type-checker verification with DuckDB
- **Augmenting** data with variations and parameter expansion

### Multi-Task System

iBERT now includes a complete baseline implementation supporting six core tasks:
1. **Code Completion** : Complete partial Ibis expressions
2. **Ibis→SQL Translation** : Convert between representations
3. **Error Resolution** : Fix compilation/type errors
4. **Q&A** : Answer Ibis-related questions
5. **Function Documentation** : Generate docstrings
6. **SQL→Ibis** : Reverse translation

Each task can be invoked via command-line scripts or the `just` command runner.

---

### What is Ibis?

[Ibis](https://ibis-project.org/) is a lazy-evaluated Python framework that compiles to SQL, providing a unified interface across 20+ database backends. iBERT helps train models to translate between SQL and Ibis expressions.

```python
# SQL
SELECT user_id, COUNT(*) as event_count
FROM events
WHERE event_date > '2024-01-01'
GROUP BY user_id

# Ibis (what iBERT generates)
events.filter(events.event_date > '2024-01-01') \
      .group_by('user_id') \
      .agg(event_count=ibis._.count())
```

---

## ✨ Key Features

### 🏭 Multi-Source Data Generation

1. **Template-Based Generation**
   - 15+ YAML templates covering SQL patterns
   - Parameterized variations (column names, conditions, aggregations)
   - DuckDB validation ensures correctness

2. **GitHub Repository Mining**
   - Extracts SQL→Ibis examples from 29+ repositories
   - Pattern matching for `.sql()` calls, multiline queries
   - Jupyter notebook support

3. **Documentation Extraction**
   - Parses Markdown, Quarto, and Jupyter notebooks
   - Identifies sequential SQL→Ibis code blocks
   - Preserves context for better training

4. **Data Augmentation**
   - Parameter expansion for broader coverage
   - Synthetic variations with Claude AI
   - Maintains semantic correctness

### 🔬 Validation & Quality

- **DuckDB Backend**: Executes both SQL and Ibis code for equivalence testing
- **Type Checking**: Validates Ibis expressions compile correctly
- **Result Comparison**: Ensures output matches with numeric tolerance
- **Provenance Tracking**: Every example includes source metadata

### 🧪 Testing

- **443 comprehensive tests** with 81% coverage
- Unit and integration tests for all components
- Fast execution (<4s for full suite)
- Pytest-based with coverage reporting

---

## 🚀 Quick Start

### Prerequisites

- Python 3.13+
- [just](https://github.com/casey/just) command runner
- Git (for repository mining)
- 16GB+ RAM (for local model inference)
- Optional: NVIDIA GPU or Apple Silicon for faster inference

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ibert.git
cd ibert

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Option 1: Modern installation (recommended)
pip install -e .              # Production dependencies only
pip install -e .[dev]         # With development tools (testing, linting, pre-commit)

# Option 2: Traditional installation
pip install -r requirements.txt      # Production dependencies
pip install -r requirements-dev.txt  # Development dependencies
```

### Developer Setup (After Installation)

If you're contributing to iBERT, set up pre-commit hooks for automatic code quality checks:

```bash
# Install pre-commit hooks (runs linting/formatting on git commit)
.venv/bin/pre-commit install

# Optional: Run hooks manually on all files
.venv/bin/pre-commit run --all-files
```

**See [CONTRIBUTING.md](CONTRIBUTING.md) for complete development guidelines.**

### Local Model Setup

iBERT uses **Qwen2.5-Coder-1.5B-Instruct** running locally (no API key needed!):

```bash
# Copy and customize configuration
cp config.yaml.example config.yaml

# First run downloads the model (~3GB, one-time, 2-5 minutes)
echo "What is Ibis?" | just qa
```

**See [LOCAL_MODEL_SETUP.md](LOCAL_MODEL_SETUP.md) for:**
- Hardware requirements and optimization
- GPU acceleration setup (CUDA/MPS)
- Memory reduction with 8-bit quantization
- Platform-specific instructions
- Performance tuning tips

### Generate Your First Dataset

```bash
# Generate SQL→Ibis training examples from templates
just generate-data

# View statistics
just dataset-stats

# See what was generated
head data/sql2ibis/train.jsonl
```

---

## 📖 Usage Guide

### Task Execution Commands

iBERT provides command-line tools for all six tasks. Each tool accepts input from either a file or stdin.

#### Code Completion

Complete partial Ibis expressions:

```bash
# From stdin
echo "table.filter(table.age >" | just complete

# From file
just complete mycode.py

# Direct script usage
./bin/ibert-complete mycode.py
```

#### Ibis to SQL Translation

Convert Ibis code to SQL:

```bash
# Default (standard SQL)
echo "table.filter(table.age > 18).select('name', 'age')" | just to-sql

# Specific dialect
just to-sql mycode.py postgres

# With table name
./bin/ibert-to-sql mycode.py --dialect duckdb --table-name users
```

#### SQL to Ibis Translation

Convert SQL queries to Ibis code:

```bash
# From stdin
echo "SELECT name, age FROM users WHERE age > 18" | just from-sql

# From file
just from-sql query.sql

# With schema information
./bin/sql-to-ibert query.sql --schema "id: int, name: string, age: int"
```

#### Error Resolution

Fix compilation and type errors:

```bash
# From stdin
echo 'table.filter(table.age > "18")' | just fix

# With error message
./bin/ibert-fix buggy.py --error "TypeError: '>' not supported"

# With context
./bin/ibert-fix buggy.py --context "age column is integer type"
```

#### Q&A

Ask questions about Ibis:

```bash
# Simple question
echo "What is lazy evaluation in Ibis?" | just qa

# From file
just qa question.txt

# With context
./bin/ibert-qa question.txt --context "I'm using DuckDB backend"
```

#### Function Documentation

Generate docstrings for functions:

```bash
# Default (Google style)
just doc myfunction.py

# NumPy style
./bin/ibert-doc myfunction.py --style numpy

# Without examples
./bin/ibert-doc myfunction.py --no-examples
```

### Data Generation Commands

#### 🚀 Quick Start - Generate Everything

```bash
# Generate ALL training data with one command!
# This runs the complete pipeline:
#   1. Template-based SQL→Ibis generation
#   2. Augmented variations
#   3. Multi-task generation (all 6 tasks)
#   4. Data validation
#   5. Mining from Ibis codebase
#   6. Final concatenation
just generate-all
```

#### Individual Generation Steps

```bash
# Generate multi-task training data (all 6 tasks)
just generate-multitask

# Validate all multi-task data
just validate-multitask

# Mine examples from Ibis codebase (all 6 tasks)
just mine-multitask

# Mine specific task
just mine-task documentation

# Generate SQL→Ibis data from templates
just generate-data

# Generate augmented data with variations
just generate-augmented

# Concatenate all data sources (templates + mined + multitask)
just concatenate-data
```

**See [MULTITASK_MINING.md](MULTITASK_MINING.md) for mining details and [VALIDATION_RESULTS.md](VALIDATION_RESULTS.md) for validation results.**

### View Statistics

```bash
# Show all dataset statistics
just all-stats

# Output:
# ╔════════════════════════════════════════════════════════════════════╗
# ║                    iBERT Dataset Statistics                        ║
# ╚════════════════════════════════════════════════════════════════════╝
#
# 📊 Generated Datasets:
#   train.jsonl:           333 examples
#   train_augmented.jsonl: 1012 examples
#
# ⛏️  Mined Datasets:
#   ibis_mined.jsonl:      93 examples
#   ibis_docs_mined.jsonl: 13 examples
#
# 🎯 Complete Dataset:
#   train_complete.jsonl:  1451 examples
```

### Testing

```bash
# Run all tests
just test

# Run tests with coverage report
just test-cov

# Generate HTML coverage report
just test-coverage-html

# Run specific test file
just test-file tests/datagen/test_concatenate_datasets.py

# Run tests matching a pattern
just test-pattern "test_extract"
```

### Advanced Workflows

```bash
# Complete data generation pipeline
just generate-data && \
just generate-augmented && \
just mine-ibis-repo && \
just mine-ibis-docs && \
just concatenate-data && \
just complete-stats

# Clean and regenerate everything
rm -rf data/sql2ibis/*.jsonl data/mining/*.jsonl data/train_complete.jsonl
just generate-augmented
just mine-ibis-repo
just concatenate-data
```

---

## 📁 Project Structure

```
ibert/
├── bin/                           # Command-line executables
│   ├── ibert-complete            # Code completion
│   ├── ibert-to-sql              # Ibis → SQL translation
│   ├── sql-to-ibert              # SQL → Ibis translation
│   ├── ibert-fix                 # Error resolution
│   ├── ibert-qa                  # Q&A system
│   └── ibert-doc                 # Documentation generation
│
├── src/
│   ├── ibert/                    # Core iBERT system
│   │   ├── config/               # Configuration management
│   │   │   └── config.py         # Config loading/saving
│   │   ├── models/               # Model implementations
│   │   │   ├── base.py           # Abstract model interface
│   │   │   ├── mistral_model.py  # HuggingFace model wrapper
│   │   │   └── factory.py        # Model factory
│   │   └── tasks/                # Task handlers
│   │       ├── code_completion.py
│   │       ├── ibis_to_sql.py
│   │       ├── sql_to_ibis.py
│   │       ├── error_resolution.py
│   │       ├── qa.py
│   │       └── documentation.py
│   │
│   └── datagen/                  # Data generation pipeline
│       ├── concatenate_datasets.py # Merge all data sources
│       ├── sql2ibis/            # Template-based generation
│       │   ├── templates/       # YAML template definitions
│       │   ├── template_loader/ # Template parsing & expansion
│       │   ├── eval/            # DuckDB validation
│       │   └── translator/      # SQL parsing utilities
│       ├── mining/              # Repository & doc mining
│       │   ├── github_miner.py  # GitHub repository mining
│       │   ├── ibis_doc_extractor.py # Documentation extraction
│       │   └── repo_urls.txt    # Repositories to mine
│       └── augmentation/        # Data augmentation
│           └── augmenter.py     # Variation generation
│
├── data/                        # Generated training data
│   ├── sql2ibis/               # Template-generated examples
│   │   ├── train.jsonl         # Base dataset
│   │   └── train_augmented.jsonl # Augmented dataset
│   ├── mining/                 # Mined examples
│   │   ├── ibis_mined.jsonl    # GitHub examples
│   │   ├── ibis_docs_mined.jsonl # Documentation examples
│   │   └── repos/              # Cloned repositories
│   └── train_complete.jsonl    # Concatenated dataset
│
├── tests/                       # Comprehensive test suite
│   ├── ibert/                  # Core system tests
│   │   ├── models/             # Model tests
│   │   ├── tasks/              # Task handler tests
│   │   └── test_config.py      # Config tests
│   ├── datagen/                # Data generation tests
│   │   ├── test_concatenate_datasets.py
│   │   └── mining/
│   │       ├── test_github_miner.py
│   │       └── test_ibis_doc_extractor.py
│   └── README.md               # Test documentation
│
├── config.yaml.example         # Example configuration
├── justfile                    # Command definitions
├── pytest.ini                  # Test configuration
├── requirements.txt            # Production dependencies
├── requirements-dev.txt        # Development dependencies
├── CLAUDE.md                   # Project guidance for Claude
└── README.md                   # This file
```

---

## 🔧 Configuration

### Model Configuration

Copy the example configuration and customize it:

```bash
cp config.yaml.example config.yaml
```

Edit `config.yaml`:

```yaml
model:
  # Model provider (huggingface with local inference)
  provider: huggingface

  # HuggingFace model ID for local models
  model_name: Qwen/Qwen2.5-Coder-1.5B-Instruct

  # Sampling temperature (0.0 to 1.0)
  temperature: 0.2

  # Maximum tokens to generate
  max_tokens: 256

  # Device: "cpu", "cuda", "mps", "auto"
  device: cpu

  # Load in 8-bit for lower memory (requires bitsandbytes, CUDA only)
  load_in_8bit: false

  # Cache directory for downloaded models (~3GB for default model)
  cache_dir: .cache

# Data directory for datasets
data_dir: data

# Logging level
log_level: INFO
```

**Local Model Benefits:**
- ✅ No API key required
- ✅ Complete privacy (no data sent to external servers)
- ✅ Zero costs
- ✅ Fast inference (20-60s per request on CPU for 1.5B model)

**Model Options:**
- `Qwen/Qwen2.5-Coder-1.5B-Instruct` (default, 1.5B parameters, code-specialized)
- `HuggingFaceTB/SmolLM2-1.7B-Instruct` (1.7B, general purpose)
- `meta-llama/Llama-3.2-1B-Instruct` (1B, Meta's smallest)
- `stabilityai/stable-code-3b` (3B, code-focused)
- `mistralai/Mistral-7B-Instruct-v0.3` (7B, slower but more capable)
- Any HuggingFace causal LM (experimental)

The system is designed to be model-agnostic. You can easily add new model providers by:
1. Implementing a new model class that extends `BaseModel`
2. Registering it in the factory
3. Updating your config to use the new provider

### Adding New Repositories to Mine

Edit [`src/datagen/mining/repo_urls.txt`](src/datagen/mining/repo_urls.txt):

```
# Format: repo_url|repo_name|optional,scan,dirs
https://github.com/ibis-project/ibis.git|ibis|ibis/tests,docs/examples
https://github.com/your-org/your-repo.git|your-repo|src,tests
```

Then run:
```bash
just mine-ibis-repo
```

### Creating New Templates

Add YAML files to [`src/datagen/sql2ibis/templates/`](src/datagen/sql2ibis/templates/):

```yaml
name: select_where
description: Basic SELECT with WHERE clause
schema:
  events:
    user_id: int64
    amount: float64

sql_template: |
  SELECT {columns}
  FROM {table}
  WHERE {condition}

ibis_template: |
  {table}.filter({condition})[[{columns}]]

variations:
  - columns: ["user_id", "amount"]
    table: events
    condition: "amount > 10"
```

---

## 📊 Dataset Format

Training examples are stored in JSONL format:

```json
{
  "id": "unique-id",
  "task": "sql_to_ibis",
  "dialect": "duckdb",
  "backend": "duckdb",
  "ibis_version": "9.5.0",
  "context": {
    "tables": {
      "events": {
        "schema": {"user_id": "int64", "amount": "float64"}
      }
    }
  },
  "input": {
    "sql": "SELECT user_id FROM events WHERE amount > 10"
  },
  "target": {
    "ibis": "events.filter(events.amount > 10)[['user_id']]",
    "expr_name": "expr"
  },
  "meta": {
    "template": "select_where",
    "source": "synthetic",
    "features": ["select", "where", "filter"],
    "difficulty": "easy"
  },
  "source_file": "data/sql2ibis/train.jsonl"
}
```

---

## 🧬 Architecture

### Data Generation Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                   Template Generation                        │
│  YAML Templates → Loader → Validator → train.jsonl          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    Augmentation                              │
│  train.jsonl → Parameter Expansion → train_augmented.jsonl  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                  Repository Mining                           │
│  GitHub Repos → Pattern Extraction → ibis_mined.jsonl       │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│               Documentation Extraction                       │
│  Docs/Notebooks → Parser → ibis_docs_mined.jsonl           │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   Concatenation                              │
│  All Sources → Merge + Stats → train_complete.jsonl        │
└─────────────────────────────────────────────────────────────┘
```

### Validation Strategy

Every generated example passes through:

1. **SQL Parsing**: Validates SQL syntax with `sqlglot`
2. **Ibis Compilation**: Ensures Ibis code compiles correctly
3. **Execution**: Runs both SQL and Ibis against DuckDB
4. **Comparison**: Verifies results match (with numeric tolerance)

Only validated examples make it into the training dataset.

---

## 🧪 Testing

The project includes a comprehensive test suite:

- **443 tests** (7 skipped)
- **81% coverage** (1,769/2,180 statements)
- Fast execution (<4 seconds)
- 100% passing rate

See [TESTING.md](TESTING.md) and [TEST_COVERAGE.md](TEST_COVERAGE.md) for detailed testing documentation.

```bash
# Run all tests
PYTHONPATH=. .venv/bin/python -m pytest tests/ -v

# Example output:
# ============================= test session starts ==============================
# collected 450 items
#
# tests/ibert/tasks/... ✓✓✓✓✓✓ (all 6 task handlers - 100% coverage)
# tests/datagen/sql2ibis/... ✓✓✓✓✓ (template generation)
# tests/datagen/multitask/... ✓✓✓✓✓ (multi-task generation)
# tests/datagen/mining/... ✓✓✓✓✓ (repository mining)
#
# ======================== 443 passed, 7 skipped in 3.6s ========================
```

---

## 📈 Current Dataset Statistics

As of the latest generation:

| Dataset | Examples | Source |
|---------|----------|--------|
| `train.jsonl` | 333 | Template-generated |
| `train_augmented.jsonl` | 1,012 | Augmented variations |
| `ibis_mined.jsonl` | 93 | GitHub repositories |
| `ibis_docs_mined.jsonl` | 13 | Documentation |
| **`train_complete.jsonl`** | **1,451** | **All sources combined** |

**Breakdown by source type:**
- Synthetic: 92.7%
- Direct SQL: 2.6%
- table.sql(): 2.3%
- Jupyter notebook: 1.5%
- Quarto doc: 0.8%
- Markdown doc: 0.1%

---

## 🛠️ Development

### Code Quality

The codebase adheres to the following engineering practices:

- **Type hints** on all functions and methods
- **Comprehensive docstrings** (NumPy style)
- **Named constants** for magic values
- **DRY principle** - no code duplication
- **Single responsibility** - focused classes and functions
- **Error handling** with graceful degradation

### Adding New Features

1. Write tests first (TDD approach)
2. Implement feature with type hints and docstrings
3. Run tests: `just test`
4. Check coverage: `just test-cov`
5. Update documentation

### Code Style

```bash
# Format code (if black is installed)
black src/ tests/

# Sort imports (if isort is installed)
isort src/ tests/

# Type check (if mypy is installed)
mypy src/
```

---

## 🔮 Future Enhancements

- [x] Multi-task system with all six tasks
- [x] Model-agnostic architecture with baseline implementation
- [ ] LoRA fine-tuning pipeline for specialized models
- [ ] Model serving infrastructure (REST API)
- [ ] Interactive web UI for data exploration
- [ ] Support for additional model providers (OpenAI, Anthropic, local models)
- [ ] Support for additional DSLs beyond Ibis
- [ ] Distributed mining across compute cluster
- [ ] Active learning for hard examples
- [ ] Enhanced validation with execution feedback

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass: `just test`
5. Submit a pull request

---

## 📝 License

See LICENSE.md

---

## 🙏 Acknowledgments

- **Alibaba Cloud**: For the Qwen2.5-Coder baseline model
- **HuggingFace**: For the transformers library and model hub
- **DuckDB**: For fast, embedded SQL execution
- **pytest**: For the testing framework

---

**Built with ❤️ for the future of code generation**

*Generate training data. Train models. Transform SQL. All with compiler-verified correctness.*
