# iBERT Complete Documentation

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-443%20passing-green.svg)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-81%25-brightgreen.svg)](htmlcov/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **Multi-Task Code Generation System for Lazy-Evaluated DSLs**
> Generate training data. Train models. Transform SQL. All with compiler-verified correctness.

---

## Table of Contents

### 🚀 Getting Started
- [1. Overview](#1-overview)
- [2. What is Ibis?](#2-what-is-ibis)
- [3. Installation](#3-installation)
- [4. Quick Start](#4-quick-start)

### 💻 Using iBERT
- [5. Task Execution](#5-task-execution)
  - [5.1 Code Completion](#51-code-completion)
  - [5.2 Ibis to SQL Translation](#52-ibis-to-sql-translation)
  - [5.3 SQL to Ibis Translation](#53-sql-to-ibis-translation)
  - [5.4 Error Resolution](#54-error-resolution)
  - [5.5 Q&A System](#55-qa-system)
  - [5.6 Function Documentation](#56-function-documentation)

### 📊 Data Generation
- [6. Data Generation Pipeline](#6-data-generation-pipeline)
  - [6.1 Quick Start - Generate Everything](#61-quick-start---generate-everything)
  - [6.2 Template-Based Generation](#62-template-based-generation)
  - [6.3 Repository Mining](#63-repository-mining)
  - [6.4 Data Augmentation](#64-data-augmentation)
  - [6.5 Multi-Task Generation](#65-multi-task-generation)
- [7. Dataset Format](#7-dataset-format)
- [8. Data Validation](#8-data-validation)

### ⚙️ Configuration & Setup
- [9. Model Configuration](#9-model-configuration)
  - [9.1 Local Model Setup](#91-local-model-setup)
  - [9.2 Hardware Requirements](#92-hardware-requirements)
  - [9.3 GPU Acceleration](#93-gpu-acceleration)
  - [9.4 Model Options](#94-model-options)
- [10. Creating Custom Templates](#10-creating-custom-templates)
- [11. Adding New Repositories](#11-adding-new-repositories)

### 🏗️ Architecture & Development
- [12. Project Structure](#12-project-structure)
- [13. Architecture Overview](#13-architecture-overview)
- [14. Testing](#14-testing)
  - [14.1 Running Tests](#141-running-tests)
  - [14.2 Test Coverage](#142-test-coverage)
  - [14.3 Writing Tests](#143-writing-tests)
- [15. Code Quality](#15-code-quality)
- [16. Development Workflow](#16-development-workflow)

### 📚 Reference
- [17. Command Reference](#17-command-reference)
- [18. Dataset Statistics](#18-dataset-statistics)
- [19. Troubleshooting](#19-troubleshooting)
- [20. Contributing](#20-contributing)
- [21. License & Acknowledgments](#21-license--acknowledgments)

---

# 🚀 Getting Started

## 1. Overview

**iBERT** (inspired by BERT, but not a BERT architecture) is a comprehensive multi-task system for lazy-evaluated DSLs, specifically targeting Ibis (Python→SQL framework). The project provides:

- ✅ **Complete multi-task baseline** supporting 6 core tasks
- ✅ **High-quality data generation** with compiler/type-checker validation
- ✅ **Multiple data sources**: templates, GitHub mining, documentation extraction
- ✅ **Local model inference** (no API key required!)
- ✅ **Comprehensive testing** (443 passing tests, 81% coverage)
- ✅ **Production-ready code** with rigorous engineering standards

### Key Features

1. **🏭 Multi-Source Data Generation**
   - Template-based generation with 15+ YAML templates
   - GitHub repository mining from 29+ repos
   - Documentation extraction from Markdown, Quarto, Jupyter
   - Synthetic augmentation with parameter variations

2. **🔬 Validation & Quality**
   - DuckDB backend for execution validation
   - Type checking via Ibis compiler
   - Result comparison with numeric tolerance
   - Provenance tracking for all examples

3. **🎯 Multi-Task System**
   - Code Completion
   - SQL→Ibis Translation
   - Ibis→SQL Translation
   - Error Resolution
   - Q&A
   - Function Documentation

4. **🧪 Comprehensive Testing**
   - 443 passing tests (100% success rate)
   - 84% code coverage
   - Fast execution (<4s for full suite)
   - pytest-based with detailed reporting

---

## 2. What is Ibis?

[Ibis](https://ibis-project.org/) is a lazy-evaluated Python framework that compiles to SQL, providing a unified interface across 20+ database backends (PostgreSQL, DuckDB, BigQuery, Snowflake, etc.).

**Example transformation:**

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

**Why Ibis?**
- **Backend agnostic**: Write once, run on any supported database
- **Type-safe**: Catches errors before execution
- **Lazy evaluation**: Builds expression trees, optimizes before execution
- **Pythonic**: Use familiar Python syntax instead of SQL strings

---

## 3. Installation

### Prerequisites

- **Python 3.13+**
- **Git** (for repository mining)
- **16GB+ RAM** (for local model inference)
- **Optional**: NVIDIA GPU or Apple Silicon for faster inference
- **Optional**: [just](https://github.com/casey/just) command runner (recommended)

### Installation Steps

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/ibert.git
cd ibert

# 2. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install production dependencies
pip install -r requirements.txt

# 4. Install development dependencies (for testing/linting)
pip install -r requirements-dev.txt

# 5. Copy example configuration
cp config.yaml.example config.yaml

# 6. Download model (one-time, ~3GB, 2-5 minutes)
echo "What is Ibis?" | .venv/bin/python bin/ibert-qa
```

### Verify Installation

```bash
# Run tests to verify everything works
PYTHONPATH=. .venv/bin/python -m pytest tests/ -v

# Should see: ======================== 443 passed, 7 skipped in ~4s ========================
```

---

## 4. Quick Start

### Generate Your First Dataset

```bash
# Option 1: Using just (recommended)
just generate-data
just dataset-stats

# Option 2: Direct Python
PYTHONPATH=. .venv/bin/python src/datagen/sql2ibis/generate_dataset.py
```

### Try a Task

```bash
# Code completion
echo "table.filter(table.age >" | .venv/bin/python bin/ibert-complete

# Q&A
echo "What is lazy evaluation in Ibis?" | .venv/bin/python bin/ibert-qa

# SQL to Ibis
echo "SELECT * FROM users WHERE age > 18" | .venv/bin/python bin/sql-to-ibert
```

### View Generated Data

```bash
# View first example
head -1 data/sql2ibis/train.jsonl | python -m json.tool

# Count examples
wc -l data/sql2ibis/train.jsonl
```

---

# 💻 Using iBERT

## 5. Task Execution

iBERT provides command-line tools for all six tasks. Each accepts input from stdin or files.

### 5.1 Code Completion

Complete partial Ibis expressions:

```bash
# From stdin
echo "table.filter(table.age >" | .venv/bin/python bin/ibert-complete

# From file
.venv/bin/python bin/ibert-complete mycode.py

# With just
just complete mycode.py

# Output:
# Completed: table.filter(table.age > 18)
```

**Options:**
- `--temperature` - Sampling temperature (default: 0.2)
- `--max-tokens` - Max tokens to generate (default: 256)

### 5.2 Ibis to SQL Translation

Convert Ibis code to SQL:

```bash
# Default (standard SQL)
echo "table.filter(table.age > 18).select('name', 'age')" | \
  .venv/bin/python bin/ibert-to-sql

# Specific dialect
.venv/bin/python bin/ibert-to-sql mycode.py --dialect postgres

# With table name
.venv/bin/python bin/ibert-to-sql mycode.py --table-name users

# With just
just to-sql mycode.py postgres
```

**Supported dialects:**
- `duckdb`, `postgres`, `mysql`, `sqlite`, `bigquery`, `snowflake`, `trino`, etc.

### 5.3 SQL to Ibis Translation

Convert SQL queries to Ibis code:

```bash
# From stdin
echo "SELECT name, age FROM users WHERE age > 18" | \
  .venv/bin/python bin/sql-to-ibert

# With schema
.venv/bin/python bin/sql-to-ibert query.sql \
  --schema "id: int, name: string, age: int"

# With just
just from-sql query.sql
```

### 5.4 Error Resolution

Fix compilation and type errors:

```bash
# From stdin
echo 'table.filter(table.age > "18")' | .venv/bin/python bin/ibert-fix

# With error message
.venv/bin/python bin/ibert-fix buggy.py \
  --error "TypeError: '>' not supported"

# With context
.venv/bin/python bin/ibert-fix buggy.py \
  --context "age column is integer type"

# With just
just fix buggy.py
```

### 5.5 Q&A System

Ask questions about Ibis:

```bash
# Simple question
echo "What is lazy evaluation in Ibis?" | .venv/bin/python bin/ibert-qa

# From file
.venv/bin/python bin/ibert-qa question.txt

# With context
.venv/bin/python bin/ibert-qa question.txt \
  --context "I'm using DuckDB backend"

# With just
just qa question.txt
```

### 5.6 Function Documentation

Generate docstrings for functions:

```bash
# Default (Google style)
.venv/bin/python bin/ibert-doc myfunction.py

# NumPy style
.venv/bin/python bin/ibert-doc myfunction.py --style numpy

# Without examples
.venv/bin/python bin/ibert-doc myfunction.py --no-examples

# With just
just doc myfunction.py
```

---

# 📊 Data Generation

## 6. Data Generation Pipeline

### 6.1 Quick Start - Generate Everything

Generate **ALL** training data with one command:

```bash
# This runs the complete pipeline:
#   1. Template-based SQL→Ibis generation
#   2. Augmented variations
#   3. Multi-task generation (all 6 tasks)
#   4. Data validation
#   5. Mining from Ibis codebase
#   6. Final concatenation
just generate-all

# View statistics
just all-stats
```

### 6.2 Template-Based Generation

Generate examples from YAML templates:

```bash
# Generate base dataset
just generate-data

# View templates
just list-templates

# Output: data/sql2ibis/train.jsonl (~333 examples)
```

**How it works:**
1. Load YAML templates from `src/datagen/sql2ibis/templates/`
2. Expand parameter variations
3. Validate SQL and Ibis code with DuckDB
4. Save validated examples to JSONL

**Example template** (`templates/select_where.yaml`):
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

### 6.3 Repository Mining

Mine examples from GitHub repositories:

```bash
# Mine from Ibis codebase
just mine-ibis-repo

# Mine documentation
just mine-ibis-docs

# Mine specific repository
PYTHONPATH=. .venv/bin/python src/datagen/mining/github_miner.py \
  --repo https://github.com/ibis-project/ibis.git

# Output: data/mining/ibis_mined.jsonl (~93 examples)
```

**Mining strategies:**
- Direct SQL strings (`SELECT ...`)
- `.sql()` method calls
- Multiline SQL queries
- Jupyter notebook cells
- Documentation examples

### 6.4 Data Augmentation

Create variations of existing examples:

```bash
# Generate augmented dataset
just generate-augmented

# Output: data/sql2ibis/train_augmented.jsonl (~1,012 examples)
```

**Augmentation strategies:**
- Column substitution (age → score, name → title)
- Table renaming (users → customers)
- Value permutation (18 → 21, 100 → 500)
- Parameter expansion

### 6.5 Multi-Task Generation

Generate data for all 6 tasks:

```bash
# Generate all tasks
just generate-multitask

# Validate generated data
just validate-multitask

# Mine task-specific examples
just mine-multitask

# Mine specific task
just mine-task documentation

# Output: data/multitask/*.jsonl
```

**Tasks generated:**
- `code_completion.jsonl`
- `sql_to_ibis.jsonl`
- `ibis_to_sql.jsonl`
- `error_resolution.jsonl`
- `qa.jsonl`
- `documentation.jsonl`
- `train_complete.jsonl` (all combined)

---

## 7. Dataset Format

Training examples are stored in JSONL (newline-delimited JSON) format:

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "task": "sql_to_ibis",
  "dialect": "duckdb",
  "backend": "duckdb",
  "ibis_version": "9.5.0",
  "context": {
    "tables": {
      "events": {
        "schema": {
          "user_id": "int64",
          "amount": "float64",
          "event_date": "date"
        }
      }
    }
  },
  "input": {
    "sql": "SELECT user_id, COUNT(*) as cnt FROM events WHERE amount > 10 GROUP BY user_id"
  },
  "target": {
    "ibis": "events.filter(events.amount > 10).group_by('user_id').agg(cnt=events.count())",
    "expr_name": "expr"
  },
  "meta": {
    "template": "group_by_aggregate",
    "source": "synthetic",
    "features": ["select", "where", "group_by", "aggregate"],
    "difficulty": "medium"
  },
  "source_file": "data/sql2ibis/train.jsonl"
}
```

**Key fields:**
- `id` - Unique identifier (UUID)
- `task` - Task type (sql_to_ibis, code_completion, etc.)
- `context` - Schema and table information
- `input` - Input to the model
- `target` - Expected output
- `meta` - Metadata (source, difficulty, features)

---

## 8. Data Validation

Every example passes through rigorous validation:

### Validation Pipeline

```
┌─────────────────────────────────────────────────────┐
│ 1. SQL Parsing                                      │
│    - Parse with sqlglot                             │
│    - Validate syntax                                │
│    - Check for unsupported features                 │
└─────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────┐
│ 2. Ibis Compilation                                 │
│    - Compile Ibis expression                        │
│    - Type checking                                  │
│    - Validate against schema                        │
└─────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────┐
│ 3. DuckDB Execution                                 │
│    - Create test tables                             │
│    - Execute SQL query                              │
│    - Execute Ibis expression                        │
└─────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────┐
│ 4. Result Comparison                                │
│    - Compare DataFrame shapes                       │
│    - Check column names                             │
│    - Verify values (numeric tolerance: 1e-12)       │
└─────────────────────────────────────────────────────┘
```

### Validation Example

```python
from src.datagen.sql2ibis.eval.validator import Validator

validator = Validator()

example = {
    "input": {"sql": "SELECT * FROM events WHERE amount > 10"},
    "target": {"ibis": "events.filter(events.amount > 10)"},
    "context": {"tables": {"events": {...}}}
}

success, error = validator.validate_example(example)
# success = True, error = None (if valid)
```

**Only validated examples are saved to the dataset.**

---

# ⚙️ Configuration & Setup

## 9. Model Configuration

### 9.1 Local Model Setup

iBERT uses **Qwen2.5-Coder-1.5B-Instruct** for local inference:

**Benefits:**
- ✅ No API key required
- ✅ Complete privacy (no data sent externally)
- ✅ Zero costs
- ✅ Fast inference (20-60s per request on CPU)

**Configuration** (`config.yaml`):

```yaml
model:
  # Model provider
  provider: huggingface

  # HuggingFace model ID
  model_name: Qwen/Qwen2.5-Coder-1.5B-Instruct

  # Sampling temperature (0.0 = deterministic, 1.0 = creative)
  temperature: 0.2

  # Maximum tokens to generate
  max_tokens: 256

  # Device: "cpu", "cuda", "mps", "auto"
  device: cpu

  # Load in 8-bit for lower memory (CUDA only)
  load_in_8bit: false

  # Cache directory for models (~3GB)
  cache_dir: .cache

# Data directory
data_dir: data

# Logging level
log_level: INFO
```

### 9.2 Hardware Requirements

**Minimum (CPU):**
- 16GB RAM
- 10GB free disk space
- 20-60s per inference

**Recommended (GPU):**
- NVIDIA GPU with 8GB+ VRAM (RTX 3060+)
- 16GB system RAM
- 2-10s per inference

**Optimal (Apple Silicon):**
- M1/M2/M3 Mac with 16GB+ unified memory
- 5-15s per inference using MPS backend

### 9.3 GPU Acceleration

**CUDA (NVIDIA):**
```yaml
model:
  device: cuda
  load_in_8bit: true  # Reduces memory usage
```

```bash
# Verify CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

**MPS (Apple Silicon):**
```yaml
model:
  device: mps
```

```bash
# Verify MPS availability
python -c "import torch; print(torch.backends.mps.is_available())"
```

**Auto-detection:**
```yaml
model:
  device: auto  # Automatically selects best available
```

### 9.4 Model Options

**Recommended models:**

| Model | Size | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| `Qwen/Qwen2.5-Coder-1.5B-Instruct` | 1.5B | Fast | Good | Default, code-specialized |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | 1.7B | Fast | Good | General purpose |
| `meta-llama/Llama-3.2-1B-Instruct` | 1B | Very Fast | Fair | Low memory |
| `stabilityai/stable-code-3b` | 3B | Medium | Better | More capable |
| `mistralai/Mistral-7B-Instruct-v0.3` | 7B | Slow | Best | High quality |

**Changing models:**
```yaml
model:
  model_name: HuggingFaceTB/SmolLM2-1.7B-Instruct  # Just change this line
```

First run will download the new model.

---

## 10. Creating Custom Templates

Templates define SQL→Ibis translation patterns.

### Template Structure

Create `src/datagen/sql2ibis/templates/my_template.yaml`:

```yaml
# Template metadata
name: my_custom_template
description: Description of what this template covers
difficulty: medium  # easy, medium, hard
features:
  - feature1
  - feature2

# Schema for test tables
schema:
  table_name:
    column1: int64
    column2: string
    column3: float64

# SQL template with placeholders
sql_template: |
  SELECT {columns}
  FROM {table}
  WHERE {condition}
  {optional_clause}

# Ibis template
ibis_template: |
  {table}.filter({condition})[[{columns}]]

# Parameter variations
variations:
  - name: variation1
    columns: ["column1", "column2"]
    table: table_name
    condition: "column1 > 10"
    optional_clause: ""

  - name: variation2
    columns: ["column2", "column3"]
    table: table_name
    condition: "column3 < 100.0"
    optional_clause: "ORDER BY column3"
```

### Advanced Features

**Multiline Ibis:**
```yaml
ibis_template: |
  {table}.filter({condition}) \
    .group_by('{group_col}') \
    .agg({agg_func}=ibis._.{agg_col}.{agg_op}())
```

**Multiple tables:**
```yaml
schema:
  users:
    user_id: int64
    name: string
  events:
    user_id: int64
    amount: float64

sql_template: |
  SELECT u.name, SUM(e.amount)
  FROM users u
  JOIN events e ON u.user_id = e.user_id
  GROUP BY u.name
```

**See existing templates** in `src/datagen/sql2ibis/templates/` for more examples.

---

## 11. Adding New Repositories

Add repositories to mine for examples.

### Edit Repository List

Edit `src/datagen/mining/repo_urls.txt`:

```
# Format: repo_url|repo_name|optional,scan,dirs

# Ibis main repo (already included)
https://github.com/ibis-project/ibis.git|ibis|ibis/tests,docs/examples

# Add your repositories
https://github.com/your-org/your-repo.git|your-repo|src,tests
https://github.com/another-org/data-project.git|data-project|notebooks
```

### Run Mining

```bash
# Mine all repositories
just mine-ibis-repo

# Mine with custom file
PYTHONPATH=. .venv/bin/python src/datagen/mining/github_miner.py \
  --repo-file my_repos.txt
```

### Mining Configuration

**Control what gets mined:**
```python
# In src/datagen/mining/github_miner.py
MINING_CONFIG = {
    "scan_dirs": ["tests", "examples", "notebooks"],
    "file_extensions": [".py", ".ipynb", ".md"],
    "max_file_size_mb": 1,
    "exclude_patterns": ["__pycache__", ".git", "node_modules"]
}
```

---

# 🏗️ Architecture & Development

## 12. Project Structure

```
ibert/
├── bin/                          # Command-line executables
│   ├── ibert-complete           # Code completion
│   ├── ibert-to-sql             # Ibis → SQL translation
│   ├── sql-to-ibert             # SQL → Ibis translation
│   ├── ibert-fix                # Error resolution
│   ├── ibert-qa                 # Q&A system
│   └── ibert-doc                # Documentation generation
│
├── src/
│   ├── ibert/                   # Core iBERT system
│   │   ├── config/              # Configuration management
│   │   │   └── config.py        # Load/save config
│   │   ├── models/              # Model implementations
│   │   │   ├── base.py          # Abstract interface
│   │   │   ├── mistral_model.py # HuggingFace wrapper
│   │   │   └── factory.py       # Model factory
│   │   └── tasks/               # Task handlers
│   │       ├── code_completion.py
│   │       ├── ibis_to_sql.py
│   │       ├── sql_to_ibis.py
│   │       ├── error_resolution.py
│   │       ├── qa.py
│   │       └── documentation.py
│   │
│   └── datagen/                 # Data generation pipeline
│       ├── concatenate_datasets.py  # Merge all sources
│       ├── sql2ibis/           # Template-based generation
│       │   ├── templates/      # YAML definitions
│       │   ├── template_loader/ # Template parsing
│       │   ├── eval/           # DuckDB validation
│       │   ├── translator/     # SQL parsing
│       │   ├── generate_dataset.py
│       │   └── generate_augmented_dataset.py
│       ├── mining/             # Repository mining
│       │   ├── github_miner.py
│       │   ├── ibis_doc_extractor.py
│       │   ├── multitask_miner.py
│       │   └── repo_urls.txt
│       ├── multitask/          # Multi-task generation
│       │   ├── generate_multitask_data.py
│       │   ├── validate_multitask_data.py
│       │   └── templates/      # Task templates
│       └── augmentation/       # Data augmentation
│           └── augmenter.py
│
├── data/                       # Generated datasets
│   ├── sql2ibis/
│   │   ├── train.jsonl
│   │   └── train_augmented.jsonl
│   ├── mining/
│   │   ├── ibis_mined.jsonl
│   │   ├── ibis_docs_mined.jsonl
│   │   └── repos/
│   ├── multitask/
│   │   ├── code_completion.jsonl
│   │   ├── sql_to_ibis.jsonl
│   │   ├── ibis_to_sql.jsonl
│   │   ├── error_resolution.jsonl
│   │   ├── qa.jsonl
│   │   ├── documentation.jsonl
│   │   └── train_complete.jsonl
│   └── train_complete.jsonl
│
├── tests/                      # Test suite (443 tests)
│   ├── ibert/
│   │   ├── models/
│   │   ├── tasks/
│   │   └── test_config.py
│   └── datagen/
│       ├── sql2ibis/
│       ├── mining/
│       ├── multitask/
│       └── test_*.py
│
├── config.yaml.example         # Example configuration
├── justfile                    # Command definitions
├── pytest.ini                  # Test configuration
├── mypy.ini                    # Type checking config
├── requirements.txt            # Production deps
├── requirements-dev.txt        # Development deps
├── CLAUDE.md                   # AI assistant guidance
└── README.md                   # Project overview
```

---

## 13. Architecture Overview

### Data Generation Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                    Template Generation                            │
│  YAML Templates → Loader → Expander → Validator → train.jsonl   │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│                       Augmentation                                │
│  train.jsonl → Column/Table/Value Substitution → augmented.jsonl│
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│                    Repository Mining                              │
│  GitHub Repos → Pattern Matching → ibis_mined.jsonl             │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│                 Documentation Extraction                          │
│  Docs/Notebooks → Parser → ibis_docs_mined.jsonl                │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│                   Multi-Task Generation                           │
│  Templates → All 6 Tasks → multitask/*.jsonl                    │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│                      Concatenation                                │
│  All Sources → Merge + Stats → train_complete.jsonl             │
└──────────────────────────────────────────────────────────────────┘
```

### Task Execution Flow

```
┌─────────────┐
│  User Input │
└──────┬──────┘
       ↓
┌──────────────────┐
│  CLI Script      │ (bin/ibert-*)
│  - Parse args    │
│  - Load config   │
└──────┬───────────┘
       ↓
┌──────────────────┐
│  Task Handler    │ (src/ibert/tasks/*.py)
│  - Format prompt │
│  - Add context   │
└──────┬───────────┘
       ↓
┌──────────────────┐
│  Model           │ (src/ibert/models/*.py)
│  - Load weights  │
│  - Generate      │
└──────┬───────────┘
       ↓
┌──────────────────┐
│  Post-process    │
│  - Clean output  │
│  - Format result │
└──────┬───────────┘
       ↓
┌──────────────────┐
│  Output          │
└──────────────────┘
```

### Model Architecture

```
┌────────────────────────────────────────────┐
│              BaseModel (ABC)               │
│  - generate()                              │
│  - load()                                  │
└────────────────┬───────────────────────────┘
                 ↓
    ┌────────────┴────────────┐
    ↓                         ↓
┌─────────────────┐  ┌──────────────────┐
│ HuggingFaceModel│  │  Future: Others  │
│ - Qwen2.5       │  │  - OpenAI API    │
│ - Local GPU/CPU │  │  - Anthropic API │
│ - Quantization  │  │  - Custom models │
└─────────────────┘  └──────────────────┘
```

---

## 14. Testing

### 14.1 Running Tests

```bash
# Run all tests (recommended)
PYTHONPATH=. .venv/bin/python -m pytest tests/ -v

# Or with just
just test

# Run specific test file
PYTHONPATH=. .venv/bin/python -m pytest tests/datagen/test_concatenate_datasets.py -v

# Run tests matching pattern
PYTHONPATH=. .venv/bin/python -m pytest tests/ -k "test_load" -v

# Run with verbose output
PYTHONPATH=. .venv/bin/python -m pytest tests/ -vv

# Run and stop at first failure
PYTHONPATH=. .venv/bin/python -m pytest tests/ -x
```

### 14.2 Test Coverage

```bash
# Run tests with coverage report
PYTHONPATH=. .venv/bin/python -m pytest tests/ --cov=src --cov-report=term-missing

# Generate HTML coverage report
PYTHONPATH=. .venv/bin/python -m pytest tests/ --cov=src --cov-report=html

# View HTML report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

**Current coverage: 81% (1,769/2,180 statements)**

**100% coverage modules:**
- All 6 task handlers
- Template loader
- SQL translator
- Utilities (I/O, logging)
- CLI utilities

### 14.3 Writing Tests

**Test structure:**

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

    def test_edge_case(self, instance):
        """Test edge case."""
        with pytest.raises(ValueError):
            instance.method(invalid_input)
```

**Best practices:**
- One test file per source file
- Descriptive test names (`test_<functionality>_<scenario>`)
- Use fixtures for setup
- Test both success and failure cases
- Include docstrings for all tests

---

## 15. Code Quality

The codebase follows rigorous engineering standards:

### Type Hints

```python
from typing import List, Dict, Optional, Tuple

def process_data(
    input_data: List[Dict[str, Any]],
    config: Optional[Dict[str, Any]] = None
) -> Tuple[bool, str]:
    """Process input data."""
    ...
```

### Docstrings (NumPy Style)

```python
def validate_example(
    self, example: Dict[str, Any]
) -> Tuple[bool, Optional[str]]:
    """Validate a training example.

    Parameters
    ----------
    example : dict
        Training example with input/target

    Returns
    -------
    success : bool
        True if valid
    error : str or None
        Error message if validation failed

    Examples
    --------
    >>> validator.validate_example(example)
    (True, None)
    """
```

### Code Style

```bash
# Format code with black (if installed)
black src/ tests/

# Sort imports with isort (if installed)
isort src/ tests/

# Type check with mypy
.venv/bin/python -m mypy src/

# Lint with flake8
.venv/bin/python -m flake8 src/ tests/
```

### Engineering Principles

- ✅ **DRY** (Don't Repeat Yourself) - No code duplication
- ✅ **Single Responsibility** - Each function/class has one purpose
- ✅ **Type Safety** - Type hints on all public APIs
- ✅ **Error Handling** - Graceful degradation with informative errors
- ✅ **Testing** - 81% coverage, all critical paths tested
- ✅ **Documentation** - Comprehensive docstrings and comments

---

## 16. Development Workflow

### Adding a New Feature

1. **Create feature branch:**
   ```bash
   git checkout -b feature/my-new-feature
   ```

2. **Write tests first (TDD):**
   ```python
   # tests/test_my_feature.py
   def test_my_new_feature():
       result = my_new_function()
       assert result == expected
   ```

3. **Implement feature:**
   ```python
   # src/my_module.py
   def my_new_function():
       """Docstring."""
       ...
   ```

4. **Run tests:**
   ```bash
   just test
   ```

5. **Check coverage:**
   ```bash
   PYTHONPATH=. .venv/bin/python -m pytest tests/ --cov=src --cov-report=term-missing
   ```

6. **Format code:**
   ```bash
   black src/ tests/
   isort src/ tests/
   ```

7. **Commit and push:**
   ```bash
   git add .
   git commit -m "Add my new feature"
   git push origin feature/my-new-feature
   ```

8. **Create pull request**

### Release Checklist

- [ ] All tests passing (443/443)
- [ ] Coverage ≥ 80%
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version bumped
- [ ] Git tag created

---

# 📚 Reference

## 17. Command Reference

### Task Execution

| Command | Description |
|---------|-------------|
| `just complete [file]` | Complete partial Ibis code |
| `just to-sql [file] [dialect]` | Convert Ibis to SQL |
| `just from-sql [file]` | Convert SQL to Ibis |
| `just fix [file]` | Fix Ibis code errors |
| `just qa [file]` | Answer Ibis questions |
| `just doc [file]` | Generate function docs |

### Data Generation

| Command | Description |
|---------|-------------|
| `just generate-all` | **Generate everything** |
| `just generate-data` | Template-based generation |
| `just generate-augmented` | Augmented variations |
| `just generate-multitask` | Multi-task generation |
| `just validate-multitask` | Validate multi-task data |
| `just mine-ibis-repo` | Mine GitHub repositories |
| `just mine-ibis-docs` | Mine documentation |
| `just mine-multitask` | Mine all task types |
| `just concatenate-data` | Merge all datasets |

### Statistics

| Command | Description |
|---------|-------------|
| `just all-stats` | **All dataset statistics** |
| `just dataset-stats` | SQL→Ibis stats |
| `just augmented-stats` | Augmented data stats |
| `just mining-stats` | Mining results stats |
| `just complete-stats` | Combined dataset stats |

### Testing

| Command | Description |
|---------|-------------|
| `just test` | Run all tests |
| `just test-cov` | Tests with coverage |
| `just test-file FILE` | Run specific test file |
| `just test-pattern PATTERN` | Run matching tests |

### Utilities

| Command | Description |
|---------|-------------|
| `just list-templates` | List available templates |
| `just clean` | Clean generated data |

---

## 18. Dataset Statistics

**Current Statistics (as of latest generation):**

| Dataset | Examples | Source Type |
|---------|----------|-------------|
| `train.jsonl` | 333 | Template-generated |
| `train_augmented.jsonl` | 1,012 | Augmented variations |
| `ibis_mined.jsonl` | 93 | GitHub repositories |
| `ibis_docs_mined.jsonl` | 13 | Documentation |
| **`train_complete.jsonl`** | **1,451** | **All combined** |

**Multi-Task Breakdown:**

| Task | Examples | Templates |
|------|----------|-----------|
| Code Completion | ~200 | 8 templates |
| SQL→Ibis | 333 | 15 templates |
| Ibis→SQL | ~150 | 5 templates |
| Error Resolution | ~100 | 6 templates |
| Q&A | ~80 | 4 templates |
| Documentation | ~120 | 7 templates |

**Source Distribution:**
- Synthetic: 92.7%
- Direct SQL: 2.6%
- table.sql(): 2.3%
- Jupyter: 1.5%
- Quarto: 0.8%
- Markdown: 0.1%

---

## 19. Troubleshooting

### 19.1 Import Errors

#### "No module named 'transformers'" or "protobuf library not found"

**Problem:**
```
ImportError during model initialization: No module named 'transformers'
Error: Missing dependency: transformers. Install with: pip install transformers torch accelerate
# or
ImportError during model initialization: requires the protobuf library but it was not found
```

**Cause:** Scripts are using system Python instead of virtual environment, or missing dependencies.

**Solution:**

1. **Always activate your virtual environment first:**
```bash
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows
```

2. **Use `just` commands** (they automatically use the venv):
```bash
just complete  # Uses .venv/bin/python automatically
just qa
just to-sql
```

3. **If running scripts directly**, use venv Python explicitly:
```bash
.venv/bin/python bin/ibert-complete
# NOT: ./bin/ibert-complete (uses system Python)
```

4. **Verify packages are installed in venv:**
```bash
.venv/bin/python -c "import transformers; print(transformers.__version__)"
```

---

### 19.2 Script Hangs with No Output

**Problem:** Running a CLI script without input causes it to hang.

**Cause:** Missing input - scripts expect either a file argument or piped input.

**Solution:** Always provide input via pipe or file argument:

```bash
# ✅ Good - piped input
echo "table.filter(" | just complete

# ✅ Good - file input
just complete mycode.py

# ❌ Bad - no input provided
just complete  # Shows error immediately now
```

**Error message you'll see:**
```
Error: No input provided

Usage:
  echo 'your input' | ibert-complete
  ibert-complete input.txt
  cat input.txt | ibert-complete
```

---

### 19.3 Model Download Times Out or Fails

**Problem:**
```
'(ReadTimeoutError("HTTPSConnectionPool(host='huggingface.co', port=443): Read timed out.")
```

**Causes:**
- Slow internet connection
- Network firewall blocking HuggingFace
- HuggingFace servers temporarily unavailable

**Solutions:**

1. **Set longer timeout:**
```bash
export HF_HUB_DOWNLOAD_TIMEOUT=300  # 5 minutes instead of default
```

2. **Use a different mirror** (if in China or restricted region):
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

3. **Download manually** and cache:
```bash
.venv/bin/python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.3')
AutoModelForCausalLM.from_pretrained('mistralai/Mistral-7B-Instruct-v0.3')
"
```

4. **Check available disk space** (~14GB required):
```bash
df -h .cache
```

---

### 19.4 Out of Memory During Model Loading

**Problem:**
```
RuntimeError: CUDA out of memory
# or
Killed (Out of memory)
```

**Solutions:**

1. **Enable 8-bit quantization** (reduces memory by 50%):
```yaml
# config.yaml
model:
  load_in_8bit: true
```

2. **Use CPU instead of GPU** (slower but more memory):
```yaml
model:
  device: cpu
```

3. **Reduce max_tokens**:
```yaml
model:
  max_tokens: 1024  # Instead of 2048
```

4. **Close other applications** to free RAM

5. **Check memory requirements** (see [LOCAL_MODEL_SETUP.md](LOCAL_MODEL_SETUP.md))

---

### 19.5 Tests Hang or Take Forever

**Problem:** Running `just test` hangs or downloads 14GB model

**Cause:** Tests are trying to actually load the model instead of using mocks.

**Solution:** This should not happen - tests use `lazy_load=True`. If it does:

```bash
# Ensure you're running tests correctly
just test

# Or with pytest directly
PYTHONPATH=. .venv/bin/python -m pytest tests/ -v
```

Tests should complete in < 4 seconds.

---

### 19.6 "just: command not found"

**Problem:** The `just` command runner is not installed.

**Solutions:**

1. **Install just:**
```bash
# macOS
brew install just

# Linux
cargo install just
# or
wget https://github.com/casey/just/releases/download/latest/just-linux-x86_64 -O /usr/local/bin/just
chmod +x /usr/local/bin/just
```

2. **Or use Python directly:**
```bash
.venv/bin/python bin/ibert-complete
.venv/bin/python bin/ibert-qa
# etc.
```

---

### 19.7 Config File Not Found

**Problem:**
```
FileNotFoundError: config.yaml not found
```

**Solution:**

1. **Create config from example:**
```bash
cp config.yaml.example config.yaml
```

2. **Or use default config** (config file is optional):
```bash
# Scripts will use defaults if no config.yaml exists
echo "test" | just complete
```

3. **Specify config explicitly:**
```bash
./bin/ibert-complete --config /path/to/config.yaml input.txt
```

---

### 19.8 Permission Denied on Scripts

**Problem:**
```
bash: ./bin/ibert-complete: Permission denied
```

**Solution:**

1. **Make scripts executable:**
```bash
chmod +x bin/*
```

2. **Or use Python explicitly:**
```bash
.venv/bin/python bin/ibert-complete
```

---

### 19.9 Slow Inference (30+ seconds per response)

**Problem:** Model is loaded but generation is very slow.

**Solutions:**

1. **Use GPU instead of CPU:**
```yaml
model:
  device: cuda  # For NVIDIA GPU
  # or
  device: mps   # For Apple Silicon
```

2. **Enable 8-bit quantization** (faster on GPU):
```yaml
model:
  load_in_8bit: true
```

3. **Reduce max_tokens:**
```yaml
model:
  max_tokens: 512  # Generate less
```

4. **Check device actually being used:**
```bash
# Look for "✓ Model loaded successfully on cuda" (good)
# vs "✓ Model loaded successfully on cpu" (slow)
```

---

### 19.10 Wrong Python Version

**Problem:**
```
SyntaxError: invalid syntax
# or
ModuleNotFoundError: No module named 'dataclasses'
```

**Cause:** Using Python < 3.13

**Solution:**

1. **Check Python version:**
```bash
.venv/bin/python --version  # Should be 3.13+
```

2. **Recreate venv with correct Python:**
```bash
rm -rf .venv
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

### 19.11 Getting Help

If none of these solutions work:

1. **Check the logs** - errors go to stderr

2. **Run with debug info:**
```bash
PYTHONPATH=. .venv/bin/python -m pdb bin/ibert-complete
```

3. **Verify installation:**
```bash
.venv/bin/python -c "
import sys
print('Python:', sys.version)
import transformers, torch, ibis
print('transformers:', transformers.__version__)
print('torch:', torch.__version__)
print('ibis:', ibis.__version__)
"
```

4. **File an issue** with:
   - Error message (full traceback)
   - Python version
   - OS version
   - Output of verification script above

---

## 20. Contributing

We welcome contributions! Here's how:

### Getting Started

1. **Fork the repository**
2. **Clone your fork:**
   ```bash
   git clone https://github.com/your-username/ibert.git
   ```
3. **Install dev dependencies:**
   ```bash
   pip install -r requirements-dev.txt
   ```

### Making Changes

1. **Create feature branch:**
   ```bash
   git checkout -b feature/amazing-feature
   ```

2. **Make your changes**
   - Follow code style guidelines
   - Add tests for new functionality
   - Update documentation

3. **Run tests:**
   ```bash
   just test
   ```

4. **Ensure coverage:**
   ```bash
   PYTHONPATH=. .venv/bin/python -m pytest tests/ --cov=src
   ```

5. **Commit changes:**
   ```bash
   git commit -m "Add amazing feature"
   ```

6. **Push to fork:**
   ```bash
   git push origin feature/amazing-feature
   ```

7. **Create Pull Request**

### Guidelines

- ✅ Write tests for all new code
- ✅ Maintain test coverage ≥80%
- ✅ Add type hints to all functions
- ✅ Write NumPy-style docstrings
- ✅ Follow existing code style
- ✅ Update documentation
- ✅ Ensure all tests pass

---

## 21. License & Acknowledgments

### License

This project is licensed under the terms in [LICENSE.md](LICENSE.md).

### Acknowledgments

**Models & Frameworks:**
- **Alibaba Cloud** - Qwen2.5-Coder baseline model
- **HuggingFace** - Transformers library and model hub
- **DuckDB** - Fast embedded SQL execution
- **Ibis** - Lazy-evaluated DataFrame framework

**Tools & Libraries:**
- **pytest** - Testing framework
- **sqlglot** - SQL parsing and transpilation
- **pandas** - Data manipulation
- **PyYAML** - Configuration management

**Community:**
- All contributors and issue reporters
- The Ibis community for building an amazing framework

---

## 📞 Support

**Documentation:**
- This document provides comprehensive coverage of all iBERT features
- [LOCAL_MODEL_SETUP.md](LOCAL_MODEL_SETUP.md) - Detailed model setup guide
- [DATA_SCALING_GUIDE.md](DATA_SCALING_GUIDE.md) - Advanced data generation scaling
- [MULTITASK_DATA_DESIGN.md](MULTITASK_DATA_DESIGN.md) - Multi-task data format design
- [MULTITASK_MINING.md](MULTITASK_MINING.md) - Detailed mining strategies

**Issues:**
- Report bugs on [GitHub Issues](https://github.com/yourusername/ibert/issues)
- Check [Troubleshooting](#19-troubleshooting) section first

**Questions:**
- Open a [Discussion](https://github.com/yourusername/ibert/discussions)
- Check existing issues and discussions

---

**Built with ❤️ for the future of code generation**

*Generate training data. Train models. Transform SQL. All with compiler-verified correctness.*

---

**Quick Navigation:**
[↑ Back to Top](#ibert-complete-documentation) | [Overview](#1-overview) | [Installation](#3-installation) | [Quick Start](#4-quick-start) | [Task Execution](#5-task-execution) | [Data Generation](#6-data-generation-pipeline) | [Testing](#14-testing) | [Contributing](#20-contributing)
