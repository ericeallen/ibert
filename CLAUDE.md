# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
**iBERT** (name inspired by BERT but not a BERT architecture) is a code generation system for lazy-evaluated DSLs, targeting Ibis (Python→SQL framework). The project includes data generation, validation, and a multi-task baseline model implementation.

## Key Technologies
- **Baseline Model**: Qwen2.5-Coder-1.5B-Instruct - Alibaba's code-specialized 1.5B parameter model
- **Future Fine-tuning**: LoRA (Low-Rank Adaptation) for parameter-efficient training (planned)
- **Target DSL**: Ibis - lazy-evaluated Python framework that compiles to SQL
- **Validation**: DuckDB backend for executing and comparing SQL/Ibis equivalence
- **Multi-Task System**: 6 tasks (code completion, translation, error resolution, Q&A, documentation)

## Development Commands

### Python Environment
**CRITICAL**: Always use `.venv/bin/python`, never system `python3`

### Data Generation
```bash
# Generate SQL→Ibis training dataset from templates
just generate-data

# Show dataset statistics
just dataset-stats

# List available templates
just list-templates
```

### Running Tests
```bash
# Test Ibis native SQL translation capabilities
.venv/bin/python src/datagen/sql2ibis/test_ibis_sql.py
```

## Architecture

### Data Generation Pipeline (`src/datagen/sql2ibis/`)
The training data pipeline generates SQL→Ibis pairs from YAML templates:

1. **Templates** (`templates/*.yaml`): Define SQL/Ibis patterns with variations
   - Each template has `sql_template`, `ibis_template`, and `variations`
   - Templates include schema context for validation
   - 15 templates covering: SELECT/WHERE, GROUP BY, JOINs, window functions, UDFs, etc.

2. **Template Loader** (`template_loader/`): Parses YAML and generates examples
   - `loader.py`: Loads templates and applies parameter substitution

3. **Validation** (`eval/`):
   - `validator.py`: Executes both SQL and Ibis code, compares results using pandas
   - `fixtures.py`: Provides test tables (events, users, etc.)
   - Uses DuckDB backend for execution
   - Only examples that pass validation are saved to training data

4. **SQL Parsing** (`translator/`):
   - `parser.py`: Uses `sqlglot` for SQL→AST parsing
   - Supports multiple dialects (DuckDB, Postgres, MySQL, BigQuery, etc.)

### Data Generation Flow
```
YAML Templates → Template Loader → SQL+Ibis Pairs → Validator → train.jsonl
                                                          ↓
                                                    DuckDB Execution
                                                    Result Comparison
```

### Validation Strategy
- SQL and Ibis code both execute against test tables in DuckDB
- Results compared with numeric tolerance (1e-12) and sorting
- Supports multi-line Ibis code with imports and UDF decorators
- Only validated examples saved to `data/sql2ibis/train.jsonl`

## Multi-Task Training Framework
The model supports six tasks with weighted sampling:
1. Code Completion
2. Ibis→SQL Translation
3. Error Resolution
4. Q&A
5. Function Documentation
6. SQL→Ibis

## Development Principles
- **Tool-verified data only**: All training examples must pass compile/type-check
- **Use venv Python**: `.venv/bin/python`, never system `python3`
- **Semantic units over regex**: Use symbol-level chunking, not line-based
- **No substitutions**: If a specified model/tool is unavailable, STOP and explain - never use fallbacks
- **Iterative refinement**: Use compiler feedback for validation
- **Track experiments**: Use Weights & Biases for reproducibility
