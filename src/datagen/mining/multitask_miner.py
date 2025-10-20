#!/usr/bin/env python3
"""
Multi-Task Example Miner for iBERT.

Mines training examples for all 6 tasks from the Ibis codebase:
1. Code Completion - incomplete Ibis expressions from code
2. SQL→Ibis Translation - SQL queries with Ibis equivalents
3. Ibis→SQL Translation - Ibis code that can be compiled to SQL
4. Error Resolution - broken code with fixes from git history
5. Q&A - questions and answers from issues, docs, Stack Overflow
6. Function Documentation - functions with docstrings

Usage:
    python multitask_miner.py
    python multitask_miner.py --task code_completion
    python multitask_miner.py --repo-path data/mining/repos/ibis
"""

import argparse
import ast
import json
import re
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Output directory configuration
DEFAULT_REPO_PATH = Path("data/mining/repos/ibis")
DEFAULT_OUTPUT_DIR = Path("data/mining/multitask")


class MultitaskMiner:
    """Mines training examples for all iBERT tasks from source code."""

    def __init__(self, repo_path: Path, output_dir: Path):
        """Initialize miner.

        Args:
            repo_path: Path to Ibis repository
            output_dir: Directory to write mined examples
        """
        self.repo_path = repo_path
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Task-specific miners
        self.miners = {
            "code_completion": self._mine_code_completion,
            "sql_to_ibis": self._mine_sql_to_ibis,
            "ibis_to_sql": self._mine_ibis_to_sql,
            "error_resolution": self._mine_error_resolution,
            "qa": self._mine_qa,
            "documentation": self._mine_documentation,
        }

    def mine_all(self) -> Dict[str, int]:
        """Mine examples for all tasks.

        Returns:
            Dictionary mapping task names to example counts
        """
        stats = {}
        for task_name, miner_func in self.miners.items():
            print(f"\n{'='*60}")
            print(f"Mining {task_name} examples...")
            print(f"{'='*60}")
            count = miner_func()
            stats[task_name] = count
            print(f"✓ Mined {count} examples for {task_name}")

        return stats

    def mine_task(self, task_name: str) -> int:
        """Mine examples for a specific task.

        Args:
            task_name: Name of task to mine

        Returns:
            Number of examples mined
        """
        if task_name not in self.miners:
            raise ValueError(
                f"Unknown task: {task_name}. "
                f"Supported tasks: {list(self.miners.keys())}"
            )

        print(f"\nMining {task_name} examples...")
        return self.miners[task_name]()

    def _mine_code_completion(self) -> int:
        """Mine code completion examples from partial Ibis expressions.

        Strategy:
        - Find chain expressions (table.filter().group_by()...)
        - Extract prefixes as partial code
        - Full expression is the completion
        """
        examples = []

        # Find Python files with Ibis code
        python_files = list(self.repo_path.glob("**/*.py"))
        print(f"  Scanning {len(python_files)} Python files...")

        for py_file in python_files:
            # Skip test files and generated code
            if "__pycache__" in str(py_file) or "test" in str(py_file):
                continue

            try:
                content = py_file.read_text(encoding="utf-8")
            except (UnicodeDecodeError, IOError):
                continue

            # Find method chains (likely Ibis expressions)
            # Pattern: variable.method1(...).method2(...).method3(...)
            pattern = r'(\w+\.(filter|select|group_by|order_by|aggregate|join|mutate|relabel|limit|head|tail|distinct)\([^)]*\)(?:\.\w+\([^)]*\))*)'

            matches = re.finditer(pattern, content)
            for match in matches:
                full_expr = match.group(1)

                # Skip very short expressions
                if len(full_expr) < 20:
                    continue

                # Extract chain components
                methods = re.findall(r'\.\w+\([^)]*\)', full_expr)

                # Create completion examples by taking prefixes
                if len(methods) >= 2:
                    # Get table name/variable
                    table_var = full_expr.split('.')[0]

                    # Create partial from first N methods
                    for i in range(1, len(methods)):
                        partial = table_var
                        for method in methods[:i]:
                            partial += method

                        # Don't include the closing paren of the last method
                        if ')' in partial:
                            # Find last method opening
                            last_paren = partial.rfind('(')
                            if last_paren > 0:
                                # Remove from opening paren to end
                                partial_incomplete = partial[:last_paren + 1]

                                example = {
                                    "id": str(uuid.uuid4()),
                                    "task": "code_completion",
                                    "source": "ibis_codebase",
                                    "input": {
                                        "partial_code": partial_incomplete.strip()
                                    },
                                    "target": {
                                        "completed_code": full_expr.strip()
                                    },
                                    "meta": {
                                        "file": str(py_file.relative_to(self.repo_path)),
                                        "chain_length": len(methods),
                                    }
                                }
                                examples.append(example)
                                break  # One example per full expression

        # Deduplicate
        seen = set()
        unique_examples = []
        for ex in examples:
            key = (ex["input"]["partial_code"], ex["target"]["completed_code"])
            if key not in seen:
                seen.add(key)
                unique_examples.append(ex)

        # Write to file
        output_file = self.output_dir / "code_completion_mined.jsonl"
        self._write_jsonl(output_file, unique_examples)
        return len(unique_examples)

    def _mine_sql_to_ibis(self) -> int:
        """Mine SQL→Ibis translation examples.

        Strategy:
        - Find .sql() calls with literal SQL strings
        - Extract surrounding Ibis context
        - Create SQL→Ibis pairs
        """
        examples = []

        python_files = list(self.repo_path.glob("**/*.py"))
        print(f"  Scanning {len(python_files)} Python files for SQL...")

        for py_file in python_files:
            if "__pycache__" in str(py_file):
                continue

            try:
                content = py_file.read_text(encoding="utf-8")
            except (UnicodeDecodeError, IOError):
                continue

            # Pattern: .sql("SELECT ...") or .sql('SELECT ...')
            patterns = [
                r'\.sql\(\s*["\']+(SELECT[^"\']+)["\']',
                r'\.sql\(\s*"""(SELECT.+?)"""',
                r'sql\s*=\s*["\']+(SELECT[^"\']+)["\']',
            ]

            for pattern in patterns:
                matches = re.finditer(pattern, content, re.DOTALL | re.IGNORECASE)
                for match in matches:
                    sql = match.group(1).strip()

                    # Skip very simple or very long SQL
                    if len(sql) < 20 or len(sql) > 500:
                        continue

                    example = {
                        "id": str(uuid.uuid4()),
                        "task": "sql_to_ibis",
                        "source": "ibis_codebase",
                        "input": {
                            "sql": sql
                        },
                        "target": {
                            "ibis": "# Mined example - Ibis equivalent to be inferred"
                        },
                        "meta": {
                            "file": str(py_file.relative_to(self.repo_path)),
                            "note": "SQL extracted from codebase, Ibis code needs manual review"
                        }
                    }
                    examples.append(example)

        # Deduplicate by SQL
        seen_sql = set()
        unique_examples = []
        for ex in examples:
            sql = ex["input"]["sql"]
            if sql not in seen_sql:
                seen_sql.add(sql)
                unique_examples.append(ex)

        output_file = self.output_dir / "sql_to_ibis_mined.jsonl"
        self._write_jsonl(output_file, unique_examples[:100])  # Limit to 100 examples
        return len(unique_examples[:100])

    def _mine_ibis_to_sql(self) -> int:
        """Mine Ibis→SQL translation examples.

        Strategy:
        - Find Ibis expressions
        - Use Ibis compiler to generate SQL
        - Create Ibis→SQL pairs
        """
        examples = []

        python_files = list(self.repo_path.glob("**/*.py"))
        print(f"  Scanning {len(python_files)} Python files for Ibis expressions...")

        for py_file in python_files:
            if "__pycache__" in str(py_file) or "/test" in str(py_file):
                continue

            try:
                content = py_file.read_text(encoding="utf-8")
            except (UnicodeDecodeError, IOError):
                continue

            # Find simple Ibis expressions (one-liners)
            pattern = r'(\w+\.(filter|select|group_by|order_by|aggregate)\([^)]+\))'
            matches = re.finditer(pattern, content)

            for match in matches:
                ibis_expr = match.group(1).strip()

                # Skip if too complex
                if len(ibis_expr) > 200 or ibis_expr.count('(') > 3:
                    continue

                example = {
                    "id": str(uuid.uuid4()),
                    "task": "ibis_to_sql",
                    "source": "ibis_codebase",
                    "input": {
                        "ibis": ibis_expr,
                        "dialect": "duckdb"
                    },
                    "target": {
                        "sql": "# SQL to be generated by Ibis compiler"
                    },
                    "meta": {
                        "file": str(py_file.relative_to(self.repo_path)),
                        "note": "Ibis expression extracted, SQL needs generation"
                    }
                }
                examples.append(example)

        # Limit and deduplicate
        seen = set()
        unique_examples = []
        for ex in examples[:100]:  # Limit to first 100
            ibis_code = ex["input"]["ibis"]
            if ibis_code not in seen:
                seen.add(ibis_code)
                unique_examples.append(ex)

        output_file = self.output_dir / "ibis_to_sql_mined.jsonl"
        self._write_jsonl(output_file, unique_examples)
        return len(unique_examples)

    def _mine_error_resolution(self) -> int:
        """Mine error resolution examples from git history.

        Strategy:
        - Find commits with keywords: "fix", "error", "bug"
        - Extract before/after code
        - Parse commit messages for error descriptions
        """
        examples = []

        try:
            # Get recent commits with fixes
            result = subprocess.run(
                ["git", "log", "--grep=fix", "--grep=error", "--grep=bug",
                 "-i", "--all", "--oneline", "-100"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                print(f"  Warning: Could not read git history")
                output_file = self.output_dir / "error_resolution_mined.jsonl"
                self._write_jsonl(output_file, [])
                return 0

            commits = result.stdout.strip().split('\n')
            print(f"  Found {len(commits)} fix commits")

            # For each commit, try to extract fix patterns
            for commit_line in commits[:20]:  # Limit to 20 commits
                if not commit_line:
                    continue

                commit_hash = commit_line.split()[0]

                # Get commit diff
                diff_result = subprocess.run(
                    ["git", "show", commit_hash, "--format=", "-U0"],
                    cwd=self.repo_path,
                    capture_output=True,
                    text=True,
                    timeout=10
                )

                if diff_result.returncode != 0:
                    continue

                diff = diff_result.stdout

                # Look for simple single-line fixes
                # Pattern: - old_code\n+ new_code
                fix_pattern = r'-\s*(.+?)\n\+\s*(.+?)(?:\n|$)'
                fixes = re.finditer(fix_pattern, diff)

                for fix in fixes:
                    broken = fix.group(1).strip()
                    fixed = fix.group(2).strip()

                    # Skip if not Python-like or too short
                    if len(broken) < 10 or len(fixed) < 10:
                        continue
                    if not any(kw in broken.lower() for kw in ['ibis', 'table', 'filter', 'select']):
                        continue

                    example = {
                        "id": str(uuid.uuid4()),
                        "task": "error_resolution",
                        "source": "git_history",
                        "input": {
                            "broken_code": broken,
                            "error": "Error inferred from git commit"
                        },
                        "target": {
                            "fixed_code": fixed,
                            "explanation": f"Fixed in commit {commit_hash}"
                        },
                        "meta": {
                            "commit": commit_hash,
                            "note": "Extracted from git history, needs manual review"
                        }
                    }
                    examples.append(example)

        except (subprocess.TimeoutExpired, subprocess.SubprocessError) as e:
            print(f"  Warning: Git operation failed: {e}")

        output_file = self.output_dir / "error_resolution_mined.jsonl"
        self._write_jsonl(output_file, examples[:50])  # Limit to 50
        return len(examples[:50])

    def _mine_qa(self) -> int:
        """Mine Q&A examples from documentation and docstrings.

        Strategy:
        - Extract questions from docstring examples
        - Find FAQ sections in docs
        - Parse "How to" sections
        """
        examples = []

        # Mine from Markdown docs
        doc_files = list(self.repo_path.glob("docs/**/*.md"))
        doc_files.extend(self.repo_path.glob("docs/**/*.qmd"))

        print(f"  Scanning {len(doc_files)} documentation files...")

        for doc_file in doc_files:
            try:
                content = doc_file.read_text(encoding="utf-8")
            except (UnicodeDecodeError, IOError):
                continue

            # Find FAQ or Q&A sections
            qa_section_pattern = r'##\s*(FAQ|Q&A|Questions?|How [Tt]o[^#]+?)(.+?)(?=\n##|$)'
            qa_sections = re.finditer(qa_section_pattern, content, re.DOTALL)

            for section in qa_sections:
                section_content = section.group(2)

                # Extract question-answer pairs
                # Pattern: Question followed by answer
                qa_pattern = r'\*\*Q:\*\*\s*(.+?)\n\*\*A:\*\*\s*(.+?)(?=\n\*\*Q:|$)'
                qa_pairs = re.finditer(qa_pattern, section_content, re.DOTALL)

                for qa in qa_pairs:
                    question = qa.group(1).strip()
                    answer = qa.group(2).strip()

                    if len(question) < 10 or len(answer) < 20:
                        continue

                    example = {
                        "id": str(uuid.uuid4()),
                        "task": "qa",
                        "source": "documentation",
                        "input": {
                            "question": question
                        },
                        "target": {
                            "answer": answer
                        },
                        "meta": {
                            "file": str(doc_file.relative_to(self.repo_path)),
                            "section": section.group(1).strip()
                        }
                    }
                    examples.append(example)

            # Also extract from "How to" headings
            how_to_pattern = r'###?\s+(How (?:do I|to|can I)[^#\n]+?)(.+?)(?=\n##|$)'
            how_tos = re.finditer(how_to_pattern, content, re.DOTALL | re.IGNORECASE)

            for how_to in how_tos:
                question = how_to.group(1).strip()
                answer_text = how_to.group(2).strip()

                # Take first paragraph as answer
                answer = answer_text.split('\n\n')[0]

                if len(question) < 10 or len(answer) < 30:
                    continue

                example = {
                    "id": str(uuid.uuid4()),
                    "task": "qa",
                    "source": "documentation",
                    "input": {
                        "question": question
                    },
                    "target": {
                        "answer": answer
                    },
                    "meta": {
                        "file": str(doc_file.relative_to(self.repo_path)),
                        "section": "How-to"
                    }
                }
                examples.append(example)

        output_file = self.output_dir / "qa_mined.jsonl"
        self._write_jsonl(output_file, examples[:100])  # Limit to 100
        return len(examples[:100])

    def _mine_documentation(self) -> int:
        """Mine function documentation examples.

        Strategy:
        - Find functions with docstrings
        - Extract function signature and docstring
        - Create function→docstring pairs
        """
        examples = []

        python_files = list(self.repo_path.glob("**/*.py"))
        print(f"  Scanning {len(python_files)} Python files for docstrings...")

        for py_file in python_files:
            if "__pycache__" in str(py_file) or "/test" in str(py_file):
                continue

            try:
                content = py_file.read_text(encoding="utf-8")
                tree = ast.parse(content)
            except (UnicodeDecodeError, IOError, SyntaxError):
                continue

            # Extract functions with docstrings
            for node in ast.walk(tree):
                if not isinstance(node, ast.FunctionDef):
                    continue

                docstring = ast.get_docstring(node)
                if not docstring or len(docstring) < 50:
                    continue

                # Get function source
                try:
                    func_lines = content.split('\n')[node.lineno - 1:node.end_lineno]
                    # Remove docstring from function
                    func_code_lines = []
                    in_docstring = False
                    for line in func_lines:
                        if '"""' in line or "'''" in line:
                            if not in_docstring:
                                in_docstring = True
                                # Keep the def line and signature
                                if line.strip().startswith('def '):
                                    func_code_lines.append(line)
                                continue
                            else:
                                in_docstring = False
                                continue
                        if not in_docstring and line.strip().startswith('def '):
                            func_code_lines.append(line)

                    # Build minimal function signature
                    if func_code_lines:
                        func_signature = '\n'.join(func_code_lines[:3])  # def + args
                    else:
                        func_signature = f"def {node.name}(...):"

                except Exception:
                    func_signature = f"def {node.name}(...):"

                # Determine docstring style
                style = "google" if "Args:" in docstring or "Returns:" in docstring else "numpy"

                example = {
                    "id": str(uuid.uuid4()),
                    "task": "documentation",
                    "source": "ibis_codebase",
                    "input": {
                        "code": func_signature,
                        "style": style
                    },
                    "target": {
                        "docstring": f'"""{docstring}"""'
                    },
                    "meta": {
                        "file": str(py_file.relative_to(self.repo_path)),
                        "function": node.name,
                    }
                }
                examples.append(example)

        output_file = self.output_dir / "documentation_mined.jsonl"
        self._write_jsonl(output_file, examples[:200])  # Limit to 200
        return len(examples[:200])

    def _write_jsonl(self, file_path: Path, examples: List[Dict[str, Any]]) -> None:
        """Write examples to JSONL file.

        Args:
            file_path: Output file path
            examples: List of example dictionaries
        """
        with open(file_path, "w") as f:
            for example in examples:
                f.write(json.dumps(example) + "\n")
        print(f"  Wrote {len(examples)} examples to {file_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Mine multi-task training examples from Ibis codebase"
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
        help="Mine specific task only (default: all tasks)",
    )
    parser.add_argument(
        "--repo-path",
        type=Path,
        default=DEFAULT_REPO_PATH,
        help="Path to Ibis repository",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for mined examples",
    )

    args = parser.parse_args()

    if not args.repo_path.exists():
        print(f"Error: Repository not found at {args.repo_path}")
        print("Run 'just mine-ibis-repo' first to clone the repository")
        sys.exit(1)

    miner = MultitaskMiner(args.repo_path, args.output)

    if args.task:
        count = miner.mine_task(args.task)
        print(f"\n✓ Mined {count} examples for {args.task}")
    else:
        stats = miner.mine_all()
        print("\n" + "=" * 60)
        print("MINING SUMMARY")
        print("=" * 60)
        for task, count in stats.items():
            print(f"{task:20s}: {count:4d} examples")
        print(f"{'TOTAL':20s}: {sum(stats.values()):4d} examples")
        print("=" * 60)


if __name__ == "__main__":
    main()
