# iBERT Training Data Generation and Task Execution

# Python interpreter
python := ".venv/bin/python"
project_root := `pwd`

# ============================================================================
# Task Execution Commands
# ============================================================================

# Complete partial Ibis code
complete INPUT='':
    #!/usr/bin/env bash
    if [ -z "{{INPUT}}" ]; then
        {{python}} {{project_root}}/bin/ibert-complete
    else
        {{python}} {{project_root}}/bin/ibert-complete "{{INPUT}}"
    fi

# Translate Ibis code to SQL
to-sql INPUT='' DIALECT='standard SQL':
    #!/usr/bin/env bash
    if [ -z "{{INPUT}}" ]; then
        {{python}} {{project_root}}/bin/ibert-to-sql --dialect "{{DIALECT}}"
    else
        {{python}} {{project_root}}/bin/ibert-to-sql "{{INPUT}}" --dialect "{{DIALECT}}"
    fi

# Translate SQL to Ibis code
from-sql INPUT='':
    #!/usr/bin/env bash
    if [ -z "{{INPUT}}" ]; then
        {{python}} {{project_root}}/bin/sql-to-ibert
    else
        {{python}} {{project_root}}/bin/sql-to-ibert "{{INPUT}}"
    fi

# Fix errors in Ibis code
fix INPUT='':
    #!/usr/bin/env bash
    if [ -z "{{INPUT}}" ]; then
        {{python}} {{project_root}}/bin/ibert-fix
    else
        {{python}} {{project_root}}/bin/ibert-fix "{{INPUT}}"
    fi

# Answer questions about Ibis
qa INPUT='':
    #!/usr/bin/env bash
    if [ -z "{{INPUT}}" ]; then
        {{python}} {{project_root}}/bin/ibert-qa
    else
        {{python}} {{project_root}}/bin/ibert-qa "{{INPUT}}"
    fi

# Generate function documentation
doc INPUT='':
    #!/usr/bin/env bash
    if [ -z "{{INPUT}}" ]; then
        {{python}} {{project_root}}/bin/ibert-doc
    else
        {{python}} {{project_root}}/bin/ibert-doc "{{INPUT}}"
    fi

# ============================================================================
# Data Generation Commands
# ============================================================================

# Generate and validate SQL→Ibis training dataset
generate-data:
    PYTHONPATH={{project_root}} {{python}} src/datagen/sql2ibis/generate_dataset.py

# Generate ALL training data (templates + augmented + multitask + mined)
generate-all:
    ./scripts/generate_all_data.sh

# Generate multi-task training data (all 6 tasks)
generate-multitask:
    PYTHONPATH={{project_root}} {{python}} src/datagen/multitask/generate_multitask_data.py

# Generate data for specific task (code_completion, ibis_to_sql, error_resolution, qa, documentation)
generate-task TASK:
    PYTHONPATH={{project_root}} {{python}} src/datagen/multitask/generate_multitask_data.py --task {{TASK}}

# Validate existing SQL→Ibis dataset
validate-data:
    PYTHONPATH={{project_root}} {{python}} src/datagen/sql2ibis/generate_dataset.py

# Validate multi-task training data (all tasks)
validate-multitask:
    PYTHONPATH={{project_root}} {{python}} src/datagen/multitask/validate_multitask_data.py

# Validate specific task data
validate-task TASK:
    PYTHONPATH={{project_root}} {{python}} src/datagen/multitask/validate_multitask_data.py --task {{TASK}}

# Validate with verbose error messages
validate-multitask-verbose:
    PYTHONPATH={{project_root}} {{python}} src/datagen/multitask/validate_multitask_data.py --verbose

# List all templates
list-templates:
    ls -1 src/datagen/sql2ibis/templates/*.yaml

# List multi-task templates
list-multitask-templates:
    find src/datagen/multitask/templates -name "*.yaml" -type f | sort

# Show multi-task data statistics
multitask-stats:
    #!/usr/bin/env bash
    echo "Multi-Task Dataset Statistics:"
    echo "=============================="
    for file in data/multitask/*.jsonl; do
        if [ -f "$file" ]; then
            count=$(wc -l < "$file")
            echo "$(basename $file): $count examples"
        fi
    done

# Show dataset stats
dataset-stats:
    @echo "Dataset: data/sql2ibis/train.jsonl"
    @wc -l data/sql2ibis/train.jsonl 2>/dev/null || echo "0 (not yet generated)"
    @echo "\nTemplate count:"
    @ls -1 src/datagen/sql2ibis/templates/*.yaml | wc -l

# Generate augmented dataset (with parameter expansion and synthetic variations)
generate-augmented:
    PYTHONPATH={{project_root}} {{python}} src/datagen/sql2ibis/generate_augmented_dataset.py

# Mine examples from Ibis GitHub repository
mine-ibis-repo:
    PYTHONPATH={{project_root}} {{python}} src/datagen/mining/github_miner.py

# Extract examples from Ibis documentation
mine-ibis-docs:
    PYTHONPATH={{project_root}} {{python}} src/datagen/mining/ibis_doc_extractor.py

# Mine multi-task examples from Ibis codebase (all 6 tasks)
mine-multitask:
    PYTHONPATH={{project_root}} {{python}} src/datagen/mining/multitask_miner.py

# Mine examples for specific task
mine-task TASK:
    PYTHONPATH={{project_root}} {{python}} src/datagen/mining/multitask_miner.py --task {{TASK}}

# Show augmented dataset stats
augmented-stats:
    @echo "Augmented dataset: data/sql2ibis/train_augmented.jsonl"
    @wc -l data/sql2ibis/train_augmented.jsonl 2>/dev/null || echo "0 (not yet generated)"
    @echo "\nTemplate count:"
    @ls -1 src/datagen/sql2ibis/templates/*.yaml | wc -l

# Count potential variations from parameterized templates
count-variations:
    @PYTHONPATH={{project_root}} {{python}} -c "from src.datagen.sql2ibis.template_loader.loader import load_templates; from pathlib import Path; templates = load_templates(Path('src/datagen/sql2ibis/templates')); total = sum(len(t.variations) for t in templates); print(f'Total potential variations: {total}')"

# ============================================================================
# Testing Commands
# ============================================================================

# Run all tests with coverage
test:
    PYTHONPATH={{project_root}} {{python}} -m pytest tests/ -v

# Run tests with coverage report
test-cov:
    PYTHONPATH={{project_root}} {{python}} -m pytest tests/ --cov=src/datagen --cov-report=term-missing --cov-report=html

# Run only unit tests
test-unit:
    PYTHONPATH={{project_root}} {{python}} -m pytest tests/ -v -m unit

# Run only integration tests
test-integration:
    PYTHONPATH={{project_root}} {{python}} -m pytest tests/ -v -m integration

# Run tests in parallel (faster)
test-parallel:
    PYTHONPATH={{project_root}} {{python}} -m pytest tests/ -v -n auto

# Run specific test file
test-file FILE:
    PYTHONPATH={{project_root}} {{python}} -m pytest {{FILE}} -v

# Run tests matching a pattern
test-pattern PATTERN:
    PYTHONPATH={{project_root}} {{python}} -m pytest tests/ -v -k "{{PATTERN}}"

# Run tests with detailed output and show print statements
test-verbose:
    PYTHONPATH={{project_root}} {{python}} -m pytest tests/ -vv -s

# Run failed tests from last run
test-failed:
    PYTHONPATH={{project_root}} {{python}} -m pytest tests/ --lf -v

# Generate HTML coverage report and open it
test-coverage-html:
    PYTHONPATH={{project_root}} {{python}} -m pytest tests/ --cov=src/datagen --cov-report=html
    @echo "\nOpening coverage report..."
    @open htmlcov/index.html || xdg-open htmlcov/index.html || echo "Please open htmlcov/index.html manually"

# Watch for file changes and re-run tests
test-watch:
    @echo "Watching for changes... (Press Ctrl+C to stop)"
    @while true; do \
        PYTHONPATH={{project_root}} {{python}} -m pytest tests/ -v; \
        inotifywait -qre close_write src/ tests/ || sleep 2; \
    done

# Clean test artifacts
test-clean:
    rm -rf .pytest_cache htmlcov .coverage
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    @echo "Test artifacts cleaned"

# Install test dependencies
install-test-deps:
    {{python}} -m pip install -r requirements-dev.txt

# ============================================================================
# Data Pipeline Commands
# ============================================================================

# Concatenate all training data (generated + mined)
concatenate-data:
    PYTHONPATH={{project_root}} {{python}} src/datagen/concatenate_datasets.py

# Show complete dataset statistics (all sources combined)
complete-stats:
    @echo "Complete Training Dataset: data/train_complete.jsonl"
    @wc -l data/train_complete.jsonl 2>/dev/null || echo "0 (not yet generated - run 'just concatenate-data')"
    @echo ""
    @if [ -f data/train_complete.jsonl ]; then \
        echo "Breakdown by source:"; \
        PYTHONPATH={{project_root}} {{python}} -c "import json; from collections import Counter; examples = [json.loads(line) for line in open('data/train_complete.jsonl')]; sources = Counter(ex.get('meta', {}).get('source') or ex.get('source', 'unknown') for ex in examples); print('\\n'.join(f'  {k}: {v}' for k, v in sorted(sources.items(), key=lambda x: -x[1])))"; \
    fi

# Show all dataset statistics (individual + complete)
all-stats:
    @echo "╔════════════════════════════════════════════════════════════════════╗"
    @echo "║                    iBERT Dataset Statistics                        ║"
    @echo "╚════════════════════════════════════════════════════════════════════╝"
    @echo ""
    @echo "📊 Generated Datasets:"
    @echo "  train.jsonl:           $(wc -l < data/sql2ibis/train.jsonl 2>/dev/null || echo 0) examples"
    @echo "  train_augmented.jsonl: $(wc -l < data/sql2ibis/train_augmented.jsonl 2>/dev/null || echo 0) examples"
    @echo ""
    @echo "⛏️  Mined Datasets:"
    @echo "  ibis_mined.jsonl:      $(wc -l < data/mining/ibis_mined.jsonl 2>/dev/null || echo 0) examples"
    @echo "  ibis_docs_mined.jsonl: $(wc -l < data/mining/ibis_docs_mined.jsonl 2>/dev/null || echo 0) examples"
    @echo ""
    @echo "🎯 Complete Dataset:"
    @echo "  train_complete.jsonl:  $(wc -l < data/train_complete.jsonl 2>/dev/null || echo 0) examples"
    @echo ""
    @echo "📝 Templates:"
    @echo "  Total templates:       $(ls -1 src/datagen/sql2ibis/templates/*.yaml 2>/dev/null | wc -l) templates"
