#!/usr/bin/env python3
"""Generate SQL→Ibis training dataset from templates."""

import json
import os
from pathlib import Path

from src.datagen.sql2ibis.template_loader.loader import load_templates, generate_examples
from src.datagen.sql2ibis.eval.fixtures import get_test_tables
from src.datagen.sql2ibis.eval.validator import Validator


def main():
    """Generate and validate dataset."""
    # Load templates
    template_dir = Path(__file__).parent / "templates"
    templates = load_templates(template_dir)
    print(f"Loaded {len(templates)} templates")

    # Generate examples
    examples = generate_examples(templates)
    print(f"Generated {len(examples)} examples")

    # Initialize validator
    validator = Validator()
    test_tables = get_test_tables()
    validator.register_tables(test_tables)

    # Validate examples
    valid_examples = []
    failed_examples = []

    for i, example in enumerate(examples, 1):
        print(f"Validating {i}/{len(examples)}: {example['meta']['template']}:{example['meta']['variation']}...", end=" ")

        success, error = validator.validate_example(example)

        if success:
            print("✓")
            valid_examples.append(example)
        else:
            print(f"✗ {error}")
            failed_examples.append((example, error))

    # Report
    print(f"\n{'='*60}")
    print(f"Results: {len(valid_examples)}/{len(examples)} passed")
    print(f"Success rate: {100 * len(valid_examples) / len(examples):.1f}%")

    if failed_examples:
        print(f"\nFailed examples:")
        for ex, error in failed_examples:
            print(f"  - {ex['meta']['template']}:{ex['meta']['variation']}: {error}")

    # Save valid examples
    output_dir = Path(__file__).parent.parent.parent.parent / "data" / "sql2ibis"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "train.jsonl"

    with open(output_file, "w") as f:
        for example in valid_examples:
            f.write(json.dumps(example) + "\n")

    assert os.path.exists(output_file)
    print(f"\nSaved {len(valid_examples)} examples to {output_file}")

    # Print sample
    if valid_examples:
        print(f"\n{'='*60}")
        print("Sample example:")
        print(f"{'='*60}")
        sample = valid_examples[0]
        print(f"Template: {sample['meta']['template']}")
        print(f"Variation: {sample['meta']['variation']}")
        print(f"\nSQL:\n{sample['input']['sql']}")
        print(f"\nIbis:\n{sample['target']['ibis']}")


if __name__ == "__main__":
    main()
