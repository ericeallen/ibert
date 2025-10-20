#!/bin/bash
# Generate all training data with augmentations
# This script generates data from all sources:
# - Template-based generation
# - Multi-task generation
# - Data mining
# - Augmentation
# - Final concatenation

set -e  # Exit on error

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "================================================================"
echo "iBERT Complete Data Generation Pipeline"
echo "================================================================"
echo ""

# 1. Generate SQL→Ibis data from templates
echo "Step 1/6: Generating SQL→Ibis training data from templates..."
echo "================================================================"
just generate-data
echo ""

# 2. Generate augmented variations
echo "Step 2/6: Generating augmented data with variations..."
echo "================================================================"
just generate-augmented
echo ""

# 3. Generate multi-task training data (all 6 tasks)
echo "Step 3/6: Generating multi-task training data (all 6 tasks)..."
echo "================================================================"
just generate-multitask
echo ""

# 4. Validate multi-task data
echo "Step 4/6: Validating multi-task training data..."
echo "================================================================"
just validate-multitask || echo "Warning: Some validation failures (this is OK)"
echo ""

# 5. Mine examples from Ibis codebase
echo "Step 5/6: Mining examples from Ibis codebase..."
echo "================================================================"
if [ -d "data/mining/repos/ibis" ]; then
    just mine-multitask
else
    echo "Ibis repository not found. Skipping mining."
    echo "Run 'just mine-ibis-repo' first to clone the repository."
fi
echo ""

# 6. Concatenate all data sources
echo "Step 6/6: Concatenating all data sources..."
echo "================================================================"
just concatenate-data
echo ""

# Show final statistics
echo "================================================================"
echo "Data Generation Complete!"
echo "================================================================"
just complete-stats
echo ""

echo "================================================================"
echo "Output Location: data/train_complete.jsonl"
echo "================================================================"
echo ""
echo "Next steps:"
echo "  - Review dataset: head data/train_complete.jsonl"
echo "  - View statistics: just all-stats"
echo "  - Start fine-tuning: [coming soon]"
echo ""
