# iBERT: Multi Task Model for Lazy-Evaluated DSLs

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-102%20passing-green.svg)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-94%25-brightgreen.svg)](htmlcov/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 🎯 Overview

iBERT (inspired by BERT, but not a BERT architecture) is a comprehensive training data generation system and model-training system for SQL→Ibis code translation. Currently, we have implemented the training data generation component. We are actively working on the model generation component. The project creates high-quality, validated training examples by:

- **Generating** synthetic SQL→Ibis pairs from parameterized templates
- **Mining** real-world examples from GitHub repositories and documentation
- **Validating** all examples through compiler/type-checker verification with DuckDB
- **Augmenting** data with variations and parameter expansion

### **ASPIRATIONAL: Multi-Task Training Framework**

Once trained, the iBERT model is intended to support six core tasks with weighted sampling:
1. **Code Completion** : Complete partial Ibis expressions
2. **Ibis→SQL Translation** : Convert between representations
3. **Error Resolution** : Fix compilation/type errors
4. **Q&A** : Answer Ibis-related questions
5. **Function Documentation** : Generate docstrings
6. **SQL→Ibis** : Reverse translation

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

- **102 comprehensive tests** with 94% coverage
- Unit and integration tests for all components
- Fast execution (< 0.5s for full suite)
- Pytest-based with coverage reporting

---

## 🚀 Quick Start

### Prerequisites

- Python 3.13+
- [just](https://github.com/casey/just) command runner
- Git (for repository mining)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ibert.git
cd ibert

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies (for testing)
pip install -r requirements-dev.txt
```

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

### Data Generation Commands

```bash
# Generate base training data from templates
just generate-data

# Generate augmented data with variations
just generate-augmented

# Mine examples from GitHub repositories
just mine-ibis-repo

# Extract examples from documentation
just mine-ibis-docs

# Concatenate all data sources into single file
just concatenate-data
```

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
├── src/
│   └── datagen/                    # Data generation pipeline
│       ├── concatenate_datasets.py # Merge all data sources
│       ├── sql2ibis/              # Template-based generation
│       │   ├── templates/         # YAML template definitions
│       │   ├── template_loader/   # Template parsing & expansion
│       │   ├── eval/              # DuckDB validation
│       │   └── translator/        # SQL parsing utilities
│       ├── mining/                # Repository & doc mining
│       │   ├── github_miner.py    # GitHub repository mining
│       │   ├── ibis_doc_extractor.py # Documentation extraction
│       │   └── repo_urls.txt      # Repositories to mine
│       └── augmentation/          # Data augmentation
│           └── augmenter.py       # Variation generation
│
├── data/                          # Generated training data
│   ├── sql2ibis/                 # Template-generated examples
│   │   ├── train.jsonl           # Base dataset
│   │   └── train_augmented.jsonl # Augmented dataset
│   ├── mining/                   # Mined examples
│   │   ├── ibis_mined.jsonl      # GitHub examples
│   │   ├── ibis_docs_mined.jsonl # Documentation examples
│   │   └── repos/                # Cloned repositories
│   └── train_complete.jsonl      # Concatenated dataset
│
├── tests/                         # Comprehensive test suite
│   ├── datagen/
│   │   ├── test_concatenate_datasets.py
│   │   └── mining/
│   │       ├── test_github_miner.py
│   │       └── test_ibis_doc_extractor.py
│   └── README.md                 # Test documentation
│
├── justfile                      # Command definitions
├── pytest.ini                    # Test configuration
├── requirements.txt              # Production dependencies
├── requirements-dev.txt          # Development dependencies
├── CLAUDE.md                     # Project guidance for Claude
└── README.md                     # This file
```

---

## 🔧 Configuration

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

- **102 tests** across 3 test modules
- **94% coverage** for tested modules
- Fast execution (< 0.5 seconds)
- 100% passing rate

See [tests/README.md](tests/README.md) for detailed testing documentation.

```bash
# Run all tests
just test

# Example output:
# ============================= test session starts ==============================
# collected 102 items
#
# tests/datagen/test_concatenate_datasets.py .................... [ 31%]
# tests/datagen/mining/test_github_miner.py ..................... [ 69%]
# tests/datagen/mining/test_ibis_doc_extractor.py ............... [100%]
#
# ============================= 102 passed in 0.43s ==============================
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

- [ ] Multi-task training framework integration
- [ ] LoRA fine-tuning pipeline
- [ ] Model serving infrastructure
- [ ] Interactive web UI for data exploration
- [ ] Support for additional DSLs beyond Ibis
- [ ] Distributed mining across compute cluster
- [ ] Active learning for hard examples
- [ ] Synthetic data generation with LLMs

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

- **Mistral AI**: For the Devstral base model
- **DuckDB**: For fast, embedded SQL execution
- **pytest**: For the testing framework

---

**Built with ❤️ for the future of code generation**

*Generate training data. Train models. Transform SQL. All with compiler-verified correctness.*
