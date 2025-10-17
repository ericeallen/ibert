"""Validation logic for SQL+Ibis pairs."""

import ibis
import pandas as pd
from typing import Any, Dict, Optional, Tuple


class ValidationError(Exception):
    """Raised when validation fails."""
    pass


class Validator:
    """Validate SQL+Ibis code equivalence."""

    def __init__(self, connection=None):
        """Initialize validator.

        Parameters
        ----------
        connection : ibis.BaseBackend, optional
            Ibis connection (defaults to in-memory DuckDB)
        """
        if connection is None:
            connection = ibis.duckdb.connect()
        self.con = connection

    def register_tables(self, tables: Dict[str, pd.DataFrame]) -> None:
        """Register test tables.

        Parameters
        ----------
        tables : dict
            Dictionary of table_name -> DataFrame
        """
        for name, df in tables.items():
            self.con.create_table(name, df, overwrite=True)

    def validate_example(self, example: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate a SQL+Ibis example.

        Parameters
        ----------
        example : dict
            Example with SQL input and Ibis target

        Returns
        -------
        tuple of (bool, str or None)
            (success, error_message)
        """
        sql = example["input"]["sql"]
        ibis_code = example["target"]["ibis"]

        try:
            # Execute SQL
            sql_result = self.con.sql(sql).execute()

            # Execute Ibis code
            # Build namespace with table references
            namespace = {"ibis": ibis}
            for table_name in example.get("context", {}).get("tables", {}).keys():
                namespace[table_name] = self.con.table(table_name)

            # Eval Ibis code - handle multi-line code with imports and decorators
            lines = ibis_code.strip().split('\n')

            # Check if we have multi-line code with imports or decorators
            has_imports = any(l.strip().startswith('import ') for l in lines)
            has_decorators = any(l.strip().startswith('@') for l in lines)

            if has_imports or has_decorators:
                # Separate setup code (imports, function definitions) from expression
                import_lines = [l for l in lines if l.strip().startswith('import ')]

                # Find decorator + function definition blocks
                setup_lines = []
                expr_lines = []
                i = 0
                while i < len(lines):
                    line = lines[i]
                    if line.strip().startswith('@'):
                        # Found decorator - collect decorator + function definition
                        decorator_block = [line]
                        i += 1
                        # Collect the def line and function body
                        while i < len(lines):
                            decorator_block.append(lines[i])
                            # Function body ends when we hit a non-indented line or another decorator/import
                            if i + 1 < len(lines):
                                next_line = lines[i + 1]
                                if (next_line.strip() and
                                    not next_line.startswith('    ') and
                                    not next_line.startswith('\t')):
                                    break
                            i += 1
                        setup_lines.extend(decorator_block)
                    elif line.strip().startswith('import '):
                        setup_lines.append(line)
                        i += 1
                    elif line.strip():
                        expr_lines.append(line)
                        i += 1
                    else:
                        i += 1

                # Execute setup code (imports and function definitions)
                if setup_lines:
                    setup_code = '\n'.join(setup_lines)
                    exec(setup_code, namespace)

                # Execute expression
                if expr_lines:
                    expr_code = '\n'.join(expr_lines)
                    exec(f"expr = {expr_code}", namespace)
            else:
                # No imports/decorators - execute entire code block as expression
                exec(f"expr = {ibis_code}", namespace)

            ibis_result = namespace["expr"].execute()

            # Compare results
            if not self._results_equal(sql_result, ibis_result):
                return False, "Results differ"

            return True, None

        except Exception as e:
            return False, str(e)

    def _results_equal(self, df1: pd.DataFrame, df2: pd.DataFrame, tolerance: float = 1e-12) -> bool:
        """Check if two DataFrames are equivalent.

        Parameters
        ----------
        df1, df2 : pd.DataFrame
            DataFrames to compare
        tolerance : float
            Numeric comparison tolerance

        Returns
        -------
        bool
            True if DataFrames are equal
        """
        # Sort both by all columns for consistent comparison
        try:
            # Find sortable columns (exclude bool, object types that might not be comparable)
            sortable_cols = [
                col for col in df1.columns
                if not pd.api.types.is_bool_dtype(df1[col])
            ]
            if sortable_cols:
                df1_sorted = df1.sort_values(by=sortable_cols).reset_index(drop=True)
                df2_sorted = df2.sort_values(by=sortable_cols).reset_index(drop=True)
            else:
                df1_sorted = df1.reset_index(drop=True)
                df2_sorted = df2.reset_index(drop=True)

            # Check shapes
            if df1_sorted.shape != df2_sorted.shape:
                return False

            # Check column names
            if list(df1_sorted.columns) != list(df2_sorted.columns):
                return False

            # Compare values
            for col in df1_sorted.columns:
                if pd.api.types.is_numeric_dtype(df1_sorted[col]):
                    if not pd.Series(df1_sorted[col]).subtract(df2_sorted[col]).abs().le(tolerance).all():
                        return False
                else:
                    if not df1_sorted[col].equals(df2_sorted[col]):
                        return False

            return True

        except Exception:
            return False
