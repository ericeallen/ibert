# iBERT Quick Start Guide

> Get up and running with iBERT in 5 minutes

## Installation

```bash
# 1. Clone and setup
git clone https://github.com/yourusername/ibert.git
cd ibert
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Optional: for testing
```

## Essential Commands

### Data Generation

```bash
# Generate from templates (fastest)
just generate-data              # → data/sql2ibis/train.jsonl

# Generate with augmentation (more examples)
just generate-augmented         # → data/sql2ibis/train_augmented.jsonl

# Mine from GitHub (takes longer, runs once)
just mine-ibis-repo            # → data/mining/ibis_mined.jsonl

# Extract from documentation
just mine-ibis-docs            # → data/mining/ibis_docs_mined.jsonl

# Combine everything
just concatenate-data          # → data/train_complete.jsonl
```

### View Results

```bash
# See what was generated
head -n 1 data/train_complete.jsonl | python -m json.tool

# Get statistics
just all-stats

# Count examples
wc -l data/train_complete.jsonl
```

### Testing

```bash
# Run tests
just test                      # All tests
just test-cov                  # With coverage
just test-file tests/path.py   # Specific file
```

## Common Workflows

### First Time Setup

```bash
# Generate complete dataset from scratch
just generate-augmented && \
just mine-ibis-repo && \
just mine-ibis-docs && \
just concatenate-data && \
just all-stats
```

### Quick Iteration

```bash
# Regenerate template data only
just generate-data
just dataset-stats
```

### Full Rebuild

```bash
# Clean and regenerate everything
rm -rf data/sql2ibis/*.jsonl data/mining/*.jsonl data/train_complete.jsonl
just generate-augmented
just concatenate-data
```

## File Locations

```
data/
├── sql2ibis/
│   ├── train.jsonl           # Base dataset (333 examples)
│   └── train_augmented.jsonl # Augmented (1,012 examples)
├── mining/
│   ├── ibis_mined.jsonl      # GitHub (93 examples)
│   └── ibis_docs_mined.jsonl # Docs (13 examples)
└── train_complete.jsonl      # Combined (1,451 examples)
```

## Customization

### Add Repositories

Edit `src/datagen/mining/repo_urls.txt`:
```
https://github.com/org/repo.git|repo-name|optional,dirs
```

### Add Templates

Create YAML in `src/datagen/sql2ibis/templates/`:
```yaml
name: my_template
sql_template: "SELECT {col} FROM {table}"
ibis_template: "{table}[['{col}']]"
variations:
  - col: user_id
    table: users
```

## Troubleshooting

### Virtual environment not activated
```bash
source .venv/bin/activate  # Run this first!
```

### Missing dependencies
```bash
pip install -r requirements.txt
```

### Tests failing
```bash
just test-failed  # Run only failed tests
just test-verbose # See detailed output
```

### Clean start
```bash
rm -rf data/ .pytest_cache htmlcov
just generate-data
```

## Next Steps

1. ✅ Generate your first dataset: `just generate-data`
2. ✅ View results: `just dataset-stats`
3. ✅ Run tests: `just test`
4. 📖 Read full [README.md](README.md)
5. 🧪 Explore [tests/README.md](tests/README.md)
6. 🔧 Check [CLAUDE.md](CLAUDE.md) for project details

## Quick Reference

| Command | Purpose | Output |
|---------|---------|--------|
| `just generate-data` | Generate from templates | train.jsonl |
| `just generate-augmented` | Generate with variations | train_augmented.jsonl |
| `just mine-ibis-repo` | Mine GitHub repos | ibis_mined.jsonl |
| `just concatenate-data` | Combine all sources | train_complete.jsonl |
| `just all-stats` | Show statistics | Terminal output |
| `just test` | Run all tests | Test results |
| `just test-cov` | Run with coverage | Coverage report |

---

**Questions?** Check the [full README](README.md) or open an issue.
