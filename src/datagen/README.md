# Data Generation Pipeline

This directory contains code for generating training datasets for iBERT fine-tuning.

## Purpose

iBERT requires high-quality training data for various code generation tasks. Rather than collecting real-world examples, we use **template-based synthesis** with compiler validation to ensure correctness.

## Current Pipelines

### sql2ibis/
SQL→Ibis translation dataset generator.
- 15 YAML templates covering core SQL patterns
- Generates 46+ validated examples
- DuckDB-backed execution verification
- See `sql2ibis/README.md` for details

## Design Principles

1. **Template-based**: Parameterized YAML templates with variations
2. **Validated**: All examples must pass compile/type-check/execution
3. **Reproducible**: Deterministic generation from templates
4. **Version-controlled**: Templates tracked in git, generated data is not

## Usage

```bash
# Generate all datasets
just generate-data

# List available templates
just list-templates

# Show dataset statistics
just dataset-stats
```

## Adding New Pipelines

Future data generation pipelines (e.g., code completion, refactoring) should follow the same structure:
```
src/datagen/<task>/
├── templates/          # YAML template definitions
├── generate_dataset.py # Main generation script
├── eval/              # Validation logic
└── README.md          # Task-specific documentation
```

Output datasets go to `data/<task>/` and are gitignored.
