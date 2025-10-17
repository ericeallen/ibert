#!/usr/bin/env python3
"""Generate SQL→Ibis training dataset with augmentation."""

import json
import os
from pathlib import Path

from src.datagen.sql2ibis.template_loader.loader import load_templates, generate_examples
from src.datagen.sql2ibis.eval.fixtures import get_test_tables
from src.datagen.sql2ibis.eval.validator import Validator
from src.datagen.augmentation.augmenter import augment_dataset


def main():
    """Generate and validate dataset with augmentation."""
    # Load templates
    template_dir = Path(__file__).parent / "templates"
    templates = load_templates(template_dir)
    print(f"Loaded {len(templates)} templates")

    # Generate examples from templates
    base_examples = generate_examples(templates)
    print(f"Generated {len(base_examples)} base examples from templates")

    # Initialize validator
    validator = Validator()
    test_tables = get_test_tables()
    validator.register_tables(test_tables)

    # Validate base examples first
    print("\n" + "="*60)
    print("Phase 1: Validating base examples")
    print("="*60)

    valid_base_examples = []
    failed_base_examples = []

    for i, example in enumerate(base_examples, 1):
        if i % 50 == 0:
            print(f"Validated {i}/{len(base_examples)} base examples...")

        success, error = validator.validate_example(example)

        if success:
            valid_base_examples.append(example)
        else:
            failed_base_examples.append((example, error))

    print(f"\n{'='*60}")
    print(f"Base validation: {len(valid_base_examples)}/{len(base_examples)} passed")
    print(f"Success rate: {100 * len(valid_base_examples) / len(base_examples):.1f}%")

    # Apply augmentation
    print("\n" + "="*60)
    print("Phase 2: Applying augmentation")
    print("="*60)

    print(f"Augmenting {len(valid_base_examples)} valid examples...")
    augmented_examples = augment_dataset(valid_base_examples, max_variations_per_example=3)
    print(f"Generated {len(augmented_examples)} total examples (including originals)")

    new_examples = [e for e in augmented_examples if "augmentation" in e.get("meta", {})]
    print(f"  - {len(valid_base_examples)} original")
    print(f"  - {len(new_examples)} augmented")

    # Validate augmented examples
    print("\n" + "="*60)
    print("Phase 3: Validating augmented examples")
    print("="*60)

    valid_augmented = []
    failed_augmented = []

    for i, example in enumerate(new_examples, 1):
        if i % 100 == 0:
            print(f"Validated {i}/{len(new_examples)} augmented examples...")

        success, error = validator.validate_example(example)

        if success:
            valid_augmented.append(example)
        else:
            failed_augmented.append((example, error))

    print(f"\n{'='*60}")
    print(f"Augmented validation: {len(valid_augmented)}/{len(new_examples)} passed")
    if len(new_examples) > 0:
        print(f"Success rate: {100 * len(valid_augmented) / len(new_examples):.1f}%")

    # Combine all valid examples
    all_valid_examples = valid_base_examples + valid_augmented

    # Final report
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Total valid examples: {len(all_valid_examples)}")
    print(f"  - From templates: {len(valid_base_examples)}")
    print(f"  - From augmentation: {len(valid_augmented)}")
    print(f"\nMultiplier: {len(all_valid_examples) / len(valid_base_examples):.2f}x")

    # Show failures if any
    if failed_base_examples:
        print(f"\n{'='*60}")
        print(f"Base failures: {len(failed_base_examples)}")
        print(f"{'='*60}")
        for (ex, error), _ in zip(failed_base_examples[:5], range(5)):
            print(f"  - {ex['meta']['template']}:{ex['meta']['variation']}: {error}")
        if len(failed_base_examples) > 5:
            print(f"  ... and {len(failed_base_examples) - 5} more")

    # Save valid examples
    output_dir = Path(__file__).parent.parent.parent.parent / "data" / "sql2ibis"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "train_augmented.jsonl"

    with open(output_file, "w") as f:
        for example in all_valid_examples:
            f.write(json.dumps(example) + "\n")

    assert os.path.exists(output_file)
    print(f"\n{'='*60}")
    print(f"Saved {len(all_valid_examples)} examples to {output_file}")
    print(f"{'='*60}")

    # Print sample
    if all_valid_examples:
        print(f"\n{'='*60}")
        print("Sample augmented example:")
        print(f"{'='*60}")
        # Find an augmented example
        aug_sample = next((e for e in all_valid_examples if "augmentation" in e.get("meta", {})), all_valid_examples[0])
        print(f"Template: {aug_sample['meta']['template']}")
        print(f"Variation: {aug_sample['meta']['variation']}")
        if "augmentation" in aug_sample["meta"]:
            print(f"Augmentation: {aug_sample['meta']['augmentation']}")
        print(f"\nSQL:\n{aug_sample['input']['sql']}")
        print(f"\nIbis:\n{aug_sample['target']['ibis']}")


if __name__ == "__main__":
    main()
