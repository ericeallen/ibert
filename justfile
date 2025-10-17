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
    @wc -l data/sql2ibis/train.jsonl
    @echo "\nTemplate count:"
    @ls -1 src/datagen/sql2ibis/templates/*.yaml | wc -l
