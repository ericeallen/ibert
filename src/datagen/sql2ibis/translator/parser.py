"""SQL parser using sqlglot."""

from typing import Tuple

import sqlglot
from sqlglot import Expression


class UnsupportedDialectError(Exception):
    """Raised when SQL dialect is not configured."""

    pass


class SQLParseError(Exception):
    """Raised when SQL parsing fails."""

    pass


def parse_sql(
    sql: str,
    dialect: str = "duckdb",
) -> Tuple[Expression, str]:
    """Parse SQL string to AST using sqlglot.

    Parameters
    ----------
    sql : str
        SQL query string
    dialect : str
        SQL dialect (duckdb, postgres, mysql, bigquery, snowflake, etc.)

    Returns
    -------
    tuple of (Expression, str)
        Parsed AST and normalized dialect name

    Raises
    ------
    UnsupportedDialectError
        If dialect not supported
    SQLParseError
        If parsing fails

    Examples
    --------
    >>> ast, dialect = parse_sql("SELECT * FROM events", "duckdb")
    >>> ast.sql(dialect="duckdb")
    'SELECT * FROM events'
    """
    supported_dialects = {
        "duckdb",
        "postgres",
        "postgresql",
        "mysql",
        "bigquery",
        "snowflake",
        "spark",
        "sqlite",
    }

    if dialect not in supported_dialects:
        raise UnsupportedDialectError(
            f"Dialect '{dialect}' not configured. "
            f"Supported: {', '.join(sorted(supported_dialects))}. "
            "Add config/dialects.yaml entry or choose a supported dialect."
        )

    try:
        ast = sqlglot.parse_one(sql, read=dialect)
    except Exception as e:
        raise SQLParseError(f"Failed to parse SQL: {e}\nSQL: {sql}") from e

    return ast, dialect


def normalize_sql(sql: str, dialect: str = "duckdb") -> str:
    """Parse and re-emit SQL in canonical form.

    Parameters
    ----------
    sql : str
        Input SQL
    dialect : str
        SQL dialect

    Returns
    -------
    str
        Normalized SQL string
    """
    ast, _ = parse_sql(sql, dialect)
    return str(ast.sql(dialect=dialect, pretty=True))
