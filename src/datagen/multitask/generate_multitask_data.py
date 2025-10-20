#!/usr/bin/env python3
"""
Multi-Task Training Data Generator for iBERT.

Generates training examples for all 6 tasks:
1. Code Completion
2. SQL→Ibis Translation
3. Ibis→SQL Translation
4. Error Resolution
5. Q&A
6. Function Documentation

Usage:
    python generate_multitask_data.py
    python generate_multitask_data.py --task code_completion
    python generate_multitask_data.py --output data/multitask/
"""

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List

import yaml


class MultitaskDataGenerator:
    """Generate training data for all iBERT tasks."""

    def __init__(self, templates_dir: Path, output_dir: Path):
        """Initialize generator.

        Args:
            templates_dir: Directory containing task template subdirectories
            output_dir: Directory to write JSONL output files
        """
        self.templates_dir = templates_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Task-specific generators
        self.generators = {
            "code_completion": self._generate_code_completion,
            "sql_to_ibis": self._generate_sql_to_ibis,
            "ibis_to_sql": self._generate_ibis_to_sql,
            "error_resolution": self._generate_error_resolution,
            "qa": self._generate_qa,
            "documentation": self._generate_documentation,
        }

    def generate_all(self) -> Dict[str, int]:
        """Generate data for all tasks.

        Returns:
            Dictionary mapping task names to example counts
        """
        stats = {}
        for task_name, generator_func in self.generators.items():
            print(f"\n{'='*60}")
            print(f"Generating {task_name} examples...")
            print(f"{'='*60}")
            count = generator_func()
            stats[task_name] = count
            print(f"✓ Generated {count} examples for {task_name}")

        # Generate combined file
        self._combine_all_tasks()

        return stats

    def generate_task(self, task_name: str) -> int:
        """Generate data for a specific task.

        Args:
            task_name: Name of task to generate

        Returns:
            Number of examples generated

        Raises:
            ValueError: If task_name is not recognized
        """
        if task_name not in self.generators:
            raise ValueError(
                f"Unknown task: {task_name}. "
                f"Supported tasks: {list(self.generators.keys())}"
            )

        print(f"\nGenerating {task_name} examples...")
        return self.generators[task_name]()

    def _load_templates(self, task_name: str) -> List[Dict[str, Any]]:
        """Load all templates for a task.

        Args:
            task_name: Name of task directory

        Returns:
            List of loaded template dictionaries
        """
        task_dir = self.templates_dir / task_name
        if not task_dir.exists():
            print(f"Warning: No templates found for {task_name} in {task_dir}")
            return []

        templates = []
        for template_file in sorted(task_dir.glob("*.yaml")):
            try:
                with open(template_file) as f:
                    template = yaml.safe_load(f)
                    templates.append(template)
                    print(f"  Loaded: {template_file.name}")
            except Exception as e:
                print(f"  Error loading {template_file}: {e}", file=sys.stderr)

        return templates

    def _write_jsonl(self, task_name: str, examples: List[Dict[str, Any]]) -> None:
        """Write examples to JSONL file.

        Args:
            task_name: Task name for output filename
            examples: List of example dictionaries
        """
        output_file = self.output_dir / f"{task_name}.jsonl"
        with open(output_file, "w") as f:
            for example in examples:
                f.write(json.dumps(example) + "\n")
        print(f"  Wrote {len(examples)} examples to {output_file}")

    def _generate_code_completion(self) -> int:
        """Generate code completion examples."""
        templates = self._load_templates("code_completion")
        examples = []

        for template in templates:
            for variation in template.get("variations", []):
                example = {
                    "id": str(uuid.uuid4()),
                    "task": "code_completion",
                    "system_prompt": template.get("system_prompt", "").strip(),
                    "context": variation.get("context", {}),
                    "input": variation["input"],
                    "target": variation["target"],
                    "meta": {
                        "template": template["name"],
                        "variation": variation.get("name", "default"),
                        "difficulty": template.get("difficulty", "medium"),
                        "features": template.get("features", []),
                    },
                }
                examples.append(example)

        self._write_jsonl("code_completion", examples)
        return len(examples)

    def _generate_sql_to_ibis(self) -> int:
        """Generate SQL→Ibis translation examples.

        Note: This task already has data from existing templates.
        This method creates a symlink to the existing data.
        """
        existing_file = Path("data/sql2ibis/train.jsonl")
        if existing_file.exists():
            symlink_path = self.output_dir / "sql_to_ibis.jsonl"
            if symlink_path.exists():
                symlink_path.unlink()
            # Count existing examples
            with open(existing_file) as f:
                count = sum(1 for _ in f)
            print(f"  Using existing SQL→Ibis data: {existing_file}")
            print(f"  Found {count} examples")
            # Copy instead of symlink for portability
            with open(existing_file) as src, open(symlink_path, "w") as dst:
                dst.write(src.read())
            return count
        else:
            print(f"  Warning: Existing SQL→Ibis data not found at {existing_file}")
            return 0

    def _generate_ibis_to_sql(self) -> int:
        """Generate Ibis→SQL translation examples."""
        templates = self._load_templates("ibis_to_sql")
        examples = []

        for template in templates:
            for variation in template.get("variations", []):
                example = {
                    "id": str(uuid.uuid4()),
                    "task": "ibis_to_sql",
                    "dialect": variation["input"].get("dialect", "duckdb"),
                    "system_prompt": template.get("system_prompt", "").strip(),
                    "context": variation.get("context", {}),
                    "input": variation["input"],
                    "target": variation["target"],
                    "meta": {
                        "template": template["name"],
                        "variation": variation.get("name", "default"),
                        "difficulty": template.get("difficulty", "medium"),
                        "features": template.get("features", []),
                    },
                }
                examples.append(example)

        self._write_jsonl("ibis_to_sql", examples)
        return len(examples)

    def _generate_error_resolution(self) -> int:
        """Generate error resolution examples."""
        templates = self._load_templates("error_resolution")
        examples = []

        for template in templates:
            for variation in template.get("variations", []):
                example = {
                    "id": str(uuid.uuid4()),
                    "task": "error_resolution",
                    "system_prompt": template.get("system_prompt", "").strip(),
                    "context": variation.get("context", {}),
                    "input": variation["input"],
                    "target": variation["target"],
                    "meta": {
                        "template": template["name"],
                        "variation": variation.get("name", "default"),
                        "difficulty": template.get("difficulty", "medium"),
                        "features": template.get("features", []),
                        "error_type": variation.get("error_type", "unknown"),
                    },
                }
                examples.append(example)

        self._write_jsonl("error_resolution", examples)
        return len(examples)

    def _generate_qa(self) -> int:
        """Generate Q&A examples."""
        templates = self._load_templates("qa")
        examples = []

        for template in templates:
            for variation in template.get("variations", []):
                example = {
                    "id": str(uuid.uuid4()),
                    "task": "qa",
                    "system_prompt": template.get("system_prompt", "").strip(),
                    "input": variation["input"],
                    "target": variation["target"],
                    "meta": {
                        "template": template["name"],
                        "variation": variation.get("name", "default"),
                        "difficulty": template.get("difficulty", "easy"),
                        "features": template.get("features", []),
                    },
                }
                examples.append(example)

        self._write_jsonl("qa", examples)
        return len(examples)

    def _generate_documentation(self) -> int:
        """Generate documentation examples."""
        templates = self._load_templates("documentation")
        examples = []

        for template in templates:
            for variation in template.get("variations", []):
                example = {
                    "id": str(uuid.uuid4()),
                    "task": "documentation",
                    "system_prompt": template.get("system_prompt", "").strip(),
                    "input": variation["input"],
                    "target": variation["target"],
                    "meta": {
                        "template": template["name"],
                        "variation": variation.get("name", "default"),
                        "difficulty": template.get("difficulty", "medium"),
                        "features": template.get("features", []),
                        "style": variation["input"].get("style", "google"),
                    },
                }
                examples.append(example)

        self._write_jsonl("documentation", examples)
        return len(examples)

    def _combine_all_tasks(self) -> None:
        """Combine all task JSONL files into one complete file."""
        combined_path = self.output_dir / "train_complete.jsonl"
        total_count = 0

        with open(combined_path, "w") as outfile:
            for task_file in sorted(self.output_dir.glob("*.jsonl")):
                if task_file.name == "train_complete.jsonl":
                    continue

                with open(task_file) as infile:
                    for line in infile:
                        outfile.write(line)
                        total_count += 1

        print(f"\n{'='*60}")
        print(f"✓ Combined all tasks into {combined_path}")
        print(f"✓ Total examples: {total_count}")
        print(f"{'='*60}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate multi-task training data for iBERT"
    )
    parser.add_argument(
        "--task",
        choices=[
            "code_completion",
            "sql_to_ibis",
            "ibis_to_sql",
            "error_resolution",
            "qa",
            "documentation",
        ],
        help="Generate data for specific task (default: all tasks)",
    )
    parser.add_argument(
        "--templates",
        type=Path,
        default=Path("src/datagen/multitask/templates"),
        help="Templates directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/multitask"),
        help="Output directory for JSONL files",
    )

    args = parser.parse_args()

    generator = MultitaskDataGenerator(args.templates, args.output)

    if args.task:
        count = generator.generate_task(args.task)
        print(f"\n✓ Generated {count} examples for {args.task}")
    else:
        stats = generator.generate_all()
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        for task, count in stats.items():
            print(f"{task:20s}: {count:4d} examples")
        print(f"{'TOTAL':20s}: {sum(stats.values()):4d} examples")
        print("=" * 60)


if __name__ == "__main__":
    main()
