# SQL→Ibis Training Data Scaling Guide

This guide explains the three-phase approach to scaling from hundreds to thousands of training examples.

## Overview

We've implemented a hybrid approach to generate thousands of high-quality SQL→Ibis training examples:

1. **Phase 1: Template Parameterization** - Multiply existing templates via parameter spaces
2. **Phase 2: Real-World Mining** - Extract examples from Ibis repository and documentation
3. **Phase 3: Synthetic Augmentation** - Generate variations through systematic transformations

## Current Scale

### Before Scaling
- 36 base templates (15 original + 21 new)
- 181 manual variations

### After Parameterization (Phase 1)
- 42 templates (added 6 parameterized templates)
- **668 variations** from parameter expansion
- **3.7x increase**

### After Augmentation (Phase 3)
- Estimated **2,000-3,000+ examples** with column/table/value substitutions
- **10-15x total multiplier**

## Phase 1: Template Parameterization

### How It Works

Instead of manually writing variations, define parameter spaces that auto-generate combinations:

```yaml
variations:
  - name: numeric_filter
    params:
      op: ">"
      threshold: 10
    parameter_space:
      op: [">", "<", ">=", "<="]
      threshold: [5, 10, 15, 20, 25]
```

This single variation expands to **4 operators × 5 thresholds = 20 examples**.

### Implementation

The system is in:
- `src/datagen/sql2ibis/template_loader/expander.py` - Parameter space expansion logic
- `src/datagen/sql2ibis/template_loader/loader.py` - Updated template loader

### Parameterized Templates

Created 6 new parameterized templates:
- `37_param_filters.yaml` - Filter operators/thresholds (80 variations)
- `38_param_aggregations.yaml` - Aggregation functions (4 variations)
- `39_param_temporal.yaml` - Temporal extractions (5 variations)
- `40_param_multi_filter.yaml` - Compound filters (324 variations)
- `41_param_filter_combinations.yaml` - Systematic filter exploration (66 variations)
- `42_param_groupby_combos.yaml` - GROUP BY + filters (20 variations)

### Usage

```bash
# Count potential variations
just count-variations

# Generate with parameterization
just generate-data
```

## Phase 2: Real-World Mining

### Data Sources

1. **Ibis GitHub Repository**
   - Test files (`ibis/tests/`)
   - Documentation examples (`docs/examples/`)
   - Patterns: `con.sql()` calls, test assertions, docstrings

2. **Ibis Documentation**
   - Markdown files with SQL→Python code blocks
   - Jupyter notebooks with Ibis examples

### Implementation

- `src/datagen/mining/github_miner.py` - Clone and extract from Ibis repo
- `src/datagen/mining/ibis_doc_extractor.py` - Parse docs and notebooks

### Usage

```bash
# Mine from Ibis GitHub
just mine-ibis-repo

# Mine from documentation
just mine-ibis-docs
```

### Expected Yield

- **500-1,000 high-quality examples** from Ibis codebase
- Real-world patterns not covered by synthetic templates
- Diverse SQL dialects and Ibis idioms

## Phase 3: Synthetic Augmentation

### Augmentation Strategies

1. **Column Name Substitution**
   - `amount` → `value`, `price`, `revenue`, `cost`
   - `user_id` → `customer_id`, `account_id`, `uid`
   - `event_ts` → `timestamp`, `created_at`, `updated_at`

2. **Table Name Substitution**
   - `events` → `transactions`, `logs`, `records`
   - `users` → `customers`, `accounts`, `clients`

3. **Value Permutation**
   - Numeric thresholds: 5, 8, 10, 12, 15, 18, 20, 25, 30
   - Years: 2023, 2024, 2025

### Implementation

- `src/datagen/augmentation/augmenter.py` - Augmentation engine
- `src/datagen/sql2ibis/generate_augmented_dataset.py` - Full pipeline

### Multiplier Effect

Each base example generates:
- 3-5 column variations
- 2-4 table variations
- 3-5 value variations

**Total: ~10-15 variations per example**

From 668 base examples → **6,000-10,000+ augmented examples**

### Usage

```bash
# Generate full augmented dataset
just generate-augmented

# Check stats
just augmented-stats
```

## Validation

All generated examples pass through DuckDB validation:
1. SQL executes successfully
2. Ibis code executes successfully
3. Results match (numeric tolerance: 1e-12)

Only validated examples are saved to the training dataset.

## Expected Final Scale

| Source | Count | Notes |
|--------|-------|-------|
| Base templates | 668 | With parameterization |
| Augmented (column/table) | ~2,000 | From base templates |
| Augmented (values) | ~1,500 | From base templates |
| Mined from Ibis | ~500 | Real-world examples |
| **Total** | **~4,500-5,000** | **25x from original 181** |

## Quality vs Quantity

### High Quality (Use for validation/test)
- Original 36 hand-crafted templates
- Mined examples from Ibis repo

### Medium Quality (Bulk of training)
- Parameterized variations
- Column/table substitutions

### Lower Quality (Diversity augmentation)
- Value permutations
- May include edge cases

## Commands Reference

```bash
# Basic generation (no augmentation)
just generate-data

# Full augmented pipeline
just generate-augmented

# Mining
just mine-ibis-repo
just mine-ibis-docs

# Stats
just dataset-stats
just augmented-stats
just count-variations
```

## Next Steps for Further Scaling

1. **Add more parameterized templates** - Each template with 10-20 parameter values = 100-200 examples
2. **Mine from Stack Overflow** - SQL + Ibis tagged questions
3. **LLM-assisted generation** - Use Claude/GPT-4 to generate diverse SQL queries
4. **Compositional templates** - Combine simple templates into complex multi-step queries
5. **Multi-dialect support** - Add PostgreSQL, MySQL, BigQuery variations

## File Structure

```
src/datagen/
├── sql2ibis/
│   ├── templates/           # YAML templates (42 total)
│   ├── template_loader/
│   │   ├── loader.py        # Template loading with expansion
│   │   └── expander.py      # Parameter space expansion
│   ├── eval/                # Validation logic
│   ├── generate_dataset.py             # Basic generation
│   └── generate_augmented_dataset.py   # Full pipeline
├── mining/
│   ├── github_miner.py      # Clone and extract from GitHub
│   └── ibis_doc_extractor.py # Parse docs/notebooks
└── augmentation/
    └── augmenter.py         # Synthetic variations

data/
├── sql2ibis/
│   ├── train.jsonl          # Basic dataset
│   └── train_augmented.jsonl # Full augmented dataset
└── mining/
    ├── repos/               # Cloned repositories
    ├── ibis_mined.jsonl     # Mined from Ibis code
    └── ibis_docs_mined.jsonl # Mined from docs
```
