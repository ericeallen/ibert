#!/usr/bin/env python3
"""
Multi-Task Data Validator for iBERT.

Validates training examples for all 6 tasks:
1. Code Completion - validates Ibis code syntax
2. SQL→Ibis Translation - validates semantic equivalence
3. Ibis→SQL Translation - validates SQL generation
4. Error Resolution - validates fixes resolve errors
5. Q&A - validates answers are informative
6. Function Documentation - validates docstring format

Usage:
    python validate_multitask_data.py
    python validate_multitask_data.py --task code_completion
    python validate_multitask_data.py --input data/multitask/
"""

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import ibis
import pandas as pd

# Import existing SQL→Ibis validator
sys.path.insert(0, str(Path(__file__).parent.parent / "sql2ibis"))
from eval.validator import Validator as SQLIbisValidator
from eval.fixtures import get_test_tables


class ValidationError(Exception):
    """Raised when validation fails."""
    pass


class MultitaskValidator:
    """Validate training examples for all iBERT tasks."""

    def __init__(self):
        """Initialize validator with Ibis connection and test data."""
        self.con = ibis.duckdb.connect()

        # Register test tables for validation
        self.test_tables = get_test_tables()
        for name, df in self.test_tables.items():
            self.con.create_table(name, df, overwrite=True)

        # SQL→Ibis validator
        self.sql_ibis_validator = SQLIbisValidator(self.con)
        self.sql_ibis_validator.register_tables(self.test_tables)

        # Task-specific validators
        self.validators = {
            "code_completion": self._validate_code_completion,
            "sql_to_ibis": self._validate_sql_to_ibis,
            "ibis_to_sql": self._validate_ibis_to_sql,
            "error_resolution": self._validate_error_resolution,
            "qa": self._validate_qa,
            "documentation": self._validate_documentation,
        }

    def _create_mock_tables(
        self, context: Dict[str, Any], namespace: Dict[str, Any]
    ) -> None:
        """Create mock tables based on schema in context.

        Args:
            context: Example context with table schemas
            namespace: Namespace dict to populate with table references
        """
        for table_name, table_info in context.get("tables", {}).items():
            schema = table_info.get("schema", {})
            if schema:
                # Create a mock DataFrame with the right schema
                mock_data = {}
                for col, dtype in schema.items():
                    if "int" in dtype:
                        mock_data[col] = [1, 2, 3, 10, 20, 30]
                    elif "float" in dtype or "double" in dtype:
                        mock_data[col] = [1.0, 2.0, 3.0, 10.5, 20.5, 30.5]
                    elif "string" in dtype or "str" in dtype:
                        mock_data[col] = ["a", "b", "c", "x", "y", "z"]
                    elif "bool" in dtype:
                        mock_data[col] = [True, False, True, False, True, False]
                    elif "date" in dtype or "time" in dtype:
                        mock_data[col] = pd.to_datetime([
                            "2024-01-01", "2024-01-02", "2024-01-03",
                            "2024-01-04", "2024-01-05", "2024-01-06"
                        ])
                    else:
                        mock_data[col] = [None, None, None, None, None, None]

                df = pd.DataFrame(mock_data)
                # Use quoted name if it's a SQL keyword
                safe_table_name = f'"{table_name}"' if table_name.lower() in [
                    "table", "select", "from", "where", "join", "order", "group"
                ] else table_name
                self.con.create_table(safe_table_name, df, overwrite=True)
                namespace[table_name] = self.con.table(safe_table_name)
            elif table_name in self.test_tables:
                # Use existing test table
                namespace[table_name] = self.con.table(table_name)

    def validate_example(
        self, example: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """Validate a single training example.

        Args:
            example: Training example dictionary

        Returns:
            Tuple of (success, error_message)
        """
        task = example.get("task")
        if task not in self.validators:
            return False, f"Unknown task: {task}"

        try:
            return self.validators[task](example)
        except Exception as e:
            return False, f"Validation exception: {str(e)}"

    def _validate_code_completion(
        self, example: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """Validate code completion example.

        Checks:
        1. Completed code is valid Python/Ibis syntax
        2. Completed code can be parsed as AST
        3. Completed code starts with partial code (if applicable)
        4. Completed code executes without errors (if has context)
        """
        partial = example["input"].get("partial_code", "")
        completed = example["target"].get("completed_code", "")

        if not completed:
            return False, "Target completed_code is empty"

        # Check 1: Valid Python syntax
        try:
            ast.parse(completed)
        except SyntaxError as e:
            return False, f"Invalid Python syntax: {e}"

        # Check 2: Completion extends partial (if partial is substantive)
        if partial and len(partial.strip()) > 3:
            # Allow for minor formatting differences
            partial_normalized = partial.strip().replace(" ", "")
            completed_normalized = completed.strip().replace(" ", "")
            if not completed_normalized.startswith(partial_normalized):
                return False, f"Completed code doesn't start with partial code"

        # Check 3: Try executing if we have table context
        context = example.get("context", {})
        if context.get("tables"):
            try:
                namespace = {"ibis": ibis}
                self._create_mock_tables(context, namespace)

                # Execute completed code
                exec(f"result = {completed}", namespace)

                # If it's an Ibis expression, try to execute it
                result = namespace.get("result")
                if hasattr(result, "execute"):
                    result.execute()

            except Exception as e:
                return False, f"Code execution failed: {e}"

        return True, None

    def _validate_sql_to_ibis(
        self, example: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """Validate SQL→Ibis translation example.

        Uses existing validator to check semantic equivalence.
        """
        # Use existing SQL→Ibis validator
        return self.sql_ibis_validator.validate_example(example)

    def _validate_ibis_to_sql(
        self, example: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """Validate Ibis→SQL translation example.

        Checks:
        1. Ibis code is valid and executable
        2. Generated SQL is valid syntax
        3. Both produce equivalent results (semantic check)
        """
        ibis_code = example["input"].get("ibis", "")
        target_sql = example["target"].get("sql", "")
        dialect = example.get("dialect", "duckdb")

        if not ibis_code or not target_sql:
            return False, "Missing ibis or sql field"

        try:
            # Build namespace with table references
            namespace = {"ibis": ibis}
            context = example.get("context", {})
            self._create_mock_tables(context, namespace)

            # Execute Ibis code
            exec(f"ibis_expr = {ibis_code}", namespace)
            ibis_result = namespace["ibis_expr"].execute()

            # Fix SQL to quote table names if they're SQL keywords
            fixed_sql = target_sql
            for table_name in context.get("tables", {}).keys():
                if table_name.lower() in ["table", "select", "from", "where", "join", "order", "group"]:
                    # Replace unquoted table name with quoted version
                    import re
                    # Match table name as a whole word
                    pattern = r'\b' + re.escape(table_name) + r'\b'
                    fixed_sql = re.sub(pattern, f'"{table_name}"', fixed_sql, flags=re.IGNORECASE)

            # Execute target SQL
            sql_result = self.con.sql(fixed_sql).execute()

            # Compare results
            if not self._results_equal(ibis_result, sql_result):
                return False, "Ibis and SQL results differ"

            return True, None

        except Exception as e:
            return False, f"Execution failed: {e}"

    def _validate_error_resolution(
        self, example: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """Validate error resolution example.

        Checks:
        1. Broken code actually produces an error
        2. Fixed code is valid Python syntax
        3. Fixed code executes without the original error
        4. Explanation is provided
        """
        broken_code = example["input"].get("broken_code", "")
        expected_error = example["input"].get("error", "")
        fixed_code = example["target"].get("fixed_code", "")
        explanation = example["target"].get("explanation", "")

        if not all([broken_code, expected_error, fixed_code, explanation]):
            return False, "Missing required fields"

        # Check 1: Broken code produces an error
        namespace = {"ibis": ibis}
        context = example.get("context", {})
        self._create_mock_tables(context, namespace)

        broken_errored = False
        try:
            exec(f"result = {broken_code}", namespace)
            result = namespace.get("result")
            if hasattr(result, "execute"):
                result.execute()
        except Exception as e:
            broken_errored = True
            # Check that error message matches expected (at least partially)
            error_str = str(e)
            if expected_error.lower() not in error_str.lower():
                # Try to match error type at least
                expected_type = expected_error.split(":")[0].strip()
                if expected_type not in str(type(e).__name__):
                    return False, f"Error mismatch: expected '{expected_error}', got '{error_str}'"

        if not broken_errored:
            return False, "Broken code didn't produce expected error"

        # Check 2: Fixed code is valid syntax
        try:
            ast.parse(fixed_code)
        except SyntaxError as e:
            return False, f"Fixed code has syntax error: {e}"

        # Check 3: Fixed code executes successfully
        try:
            namespace_fixed = {"ibis": ibis}
            self._create_mock_tables(context, namespace_fixed)

            exec(f"result = {fixed_code}", namespace_fixed)
            result = namespace_fixed.get("result")
            if hasattr(result, "execute"):
                result.execute()
        except Exception as e:
            return False, f"Fixed code still errors: {e}"

        # Check 4: Explanation exists and is meaningful
        if len(explanation.strip()) < 10:
            return False, "Explanation too short (< 10 chars)"

        return True, None

    def _validate_qa(
        self, example: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """Validate Q&A example.

        Checks:
        1. Question is not empty
        2. Answer is not empty
        3. Answer has minimum length (20+ chars)
        4. If code examples provided, they are valid syntax
        """
        question = example["input"].get("question", "")
        answer = example["target"].get("answer", "")

        if not question or not answer:
            return False, "Question or answer is empty"

        if len(answer.strip()) < 20:
            return False, "Answer too short (< 20 chars)"

        # Check for code blocks in answer
        code_blocks = re.findall(r"```(?:python)?\n(.*?)```", answer, re.DOTALL)
        for i, code_block in enumerate(code_blocks):
            code = code_block.strip()
            if not code:
                continue

            try:
                # Try to parse as Python
                ast.parse(code)
            except SyntaxError:
                # May be incomplete snippet, that's okay
                # But check it's not complete garbage
                if len(code) > 10 and not any(
                    keyword in code for keyword in ["table", "ibis", "filter", "select", "group_by"]
                ):
                    return False, f"Code block {i+1} appears invalid"

        return True, None

    def _validate_documentation(
        self, example: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """Validate documentation example.

        Checks:
        1. Function code is valid Python syntax
        2. Function has a proper signature
        3. Generated docstring follows specified style
        4. Docstring has required sections (Args, Returns, etc.)
        """
        # Try both 'code' and 'function_code' field names
        function_code = example["input"].get("code") or example["input"].get("function_code", "")
        docstring = example["target"].get("docstring", "")
        style = example["input"].get("style", "google")

        if not function_code or not docstring:
            return False, "Function code or docstring is empty"

        # Check 1: Valid Python syntax
        try:
            tree = ast.parse(function_code)
        except SyntaxError as e:
            return False, f"Function code has syntax error: {e}"

        # Check 2: Contains a function definition
        has_function = any(isinstance(node, ast.FunctionDef) for node in ast.walk(tree))
        if not has_function:
            return False, "Function code doesn't contain a function definition"

        # Check 3: Docstring format validation
        if style == "google":
            # Google style should have sections like Args:, Returns:
            required_keywords = ["Args:", "Returns:"]
            # At least one should be present
            if not any(kw in docstring for kw in required_keywords):
                return False, "Google-style docstring missing Args: or Returns: section"

        elif style == "numpy":
            # NumPy style should have sections like Parameters, Returns
            required_keywords = ["Parameters", "Returns"]
            if not any(kw in docstring for kw in required_keywords):
                return False, "NumPy-style docstring missing Parameters or Returns section"

        # Check 4: Minimum docstring length
        if len(docstring.strip()) < 30:
            return False, "Docstring too short (< 30 chars)"

        return True, None

    def _results_equal(
        self, df1: pd.DataFrame, df2: pd.DataFrame, tolerance: float = 1e-12
    ) -> bool:
        """Check if two DataFrames are equivalent.

        Args:
            df1, df2: DataFrames to compare
            tolerance: Numeric comparison tolerance

        Returns:
            True if DataFrames are equal
        """
        # Use existing validator's comparison logic
        return self.sql_ibis_validator._results_equal(df1, df2, tolerance)

    def validate_file(
        self, file_path: Path
    ) -> Tuple[int, int, List[Dict[str, Any]]]:
        """Validate all examples in a JSONL file.

        Args:
            file_path: Path to JSONL file

        Returns:
            Tuple of (total_count, valid_count, failed_examples)
        """
        total = 0
        valid = 0
        failed = []

        with open(file_path) as f:
            for line_num, line in enumerate(f, 1):
                total += 1
                try:
                    example = json.loads(line)
                    success, error = self.validate_example(example)

                    if success:
                        valid += 1
                    else:
                        failed.append({
                            "line": line_num,
                            "id": example.get("id", "unknown"),
                            "task": example.get("task", "unknown"),
                            "error": error,
                        })
                except json.JSONDecodeError as e:
                    failed.append({
                        "line": line_num,
                        "error": f"JSON decode error: {e}",
                    })

        return total, valid, failed


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate multi-task training data for iBERT"
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
        help="Validate specific task only (default: all tasks)",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/multitask"),
        help="Input directory containing JSONL files",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed error messages",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop validation on first error",
    )

    args = parser.parse_args()

    print("Initializing validator...")
    validator = MultitaskValidator()
    print("✓ Validator ready\n")

    # Determine which files to validate
    if args.task:
        files = [args.input / f"{args.task}.jsonl"]
    else:
        files = sorted(args.input.glob("*.jsonl"))
        # Exclude combined file to avoid duplication
        files = [f for f in files if f.name != "train_complete.jsonl"]

    total_all = 0
    valid_all = 0
    failed_all = []

    for file_path in files:
        if not file_path.exists():
            print(f"⚠ File not found: {file_path}")
            continue

        print(f"{'='*60}")
        print(f"Validating: {file_path.name}")
        print(f"{'='*60}")

        total, valid, failed = validator.validate_file(file_path)
        total_all += total
        valid_all += valid
        failed_all.extend(failed)

        # Print results
        pass_rate = (valid / total * 100) if total > 0 else 0
        print(f"Total:   {total:4d}")
        print(f"Valid:   {valid:4d}")
        print(f"Failed:  {len(failed):4d}")
        print(f"Pass rate: {pass_rate:.1f}%")

        if failed and args.verbose:
            print("\nFailed examples:")
            for item in failed[:10]:  # Show first 10
                print(f"  Line {item.get('line', '?')}: {item.get('error', 'Unknown error')}")
            if len(failed) > 10:
                print(f"  ... and {len(failed) - 10} more")

        if failed and args.stop_on_error:
            print("\n❌ Stopping on first error")
            sys.exit(1)

        print()

    # Summary
    print(f"{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"Files validated: {len([f for f in files if f.exists()])}")
    print(f"Total examples:  {total_all:4d}")
    print(f"Valid examples:  {valid_all:4d}")
    print(f"Failed examples: {len(failed_all):4d}")

    if total_all > 0:
        overall_pass_rate = valid_all / total_all * 100
        print(f"Overall pass rate: {overall_pass_rate:.1f}%")

    print(f"{'='*60}")

    if failed_all:
        print("\n⚠ Some examples failed validation")
        if not args.verbose:
            print("Run with --verbose to see error details")
        sys.exit(1)
    else:
        print("\n✓ All examples passed validation!")
        sys.exit(0)


if __name__ == "__main__":
    main()
