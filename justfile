# iBERT Training Data Generation

# Python interpreter
python := ".venv/bin/python"
project_root := `pwd`

# Generate and validate SQL→Ibis training dataset
generate-data:
    PYTHONPATH={{project_root}} {{python}} src/datagen/sql2ibis/generate_dataset.py

# Validate existing dataset
validate-data:
    PYTHONPATH={{project_root}} {{python}} src/datagen/sql2ibis/generate_dataset.py

# List all templates
list-templates:
    ls -1 src/datagen/sql2ibis/templates/*.yaml

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

# Show augmented dataset stats
augmented-stats:
    @echo "Augmented dataset: data/sql2ibis/train_augmented.jsonl"
    @wc -l data/sql2ibis/train_augmented.jsonl 2>/dev/null || echo "0 (not yet generated)"
    @echo "\nTemplate count:"
    @ls -1 src/datagen/sql2ibis/templates/*.yaml | wc -l

# Count potential variations from parameterized templates
count-variations:
    @PYTHONPATH={{project_root}} {{python}} -c "from src.datagen.sql2ibis.template_loader.loader import load_templates; from pathlib import Path; templates = load_templates(Path('src/datagen/sql2ibis/templates')); total = sum(len(t.variations) for t in templates); print(f'Total potential variations: {total}')"
