"""Type system helpers for SQL→Ibis translation."""

from typing import Dict

# SQL type → Ibis dtype mapping
SQL_TO_IBIS_TYPES: Dict[str, str] = {
    # Integer types
    "TINYINT": "int8",
    "SMALLINT": "int16",
    "INTEGER": "int32",
    "INT": "int32",
    "BIGINT": "int64",
    # Floating point
    "REAL": "float32",
    "FLOAT": "float32",
    "DOUBLE": "float64",
    "DOUBLE PRECISION": "float64",
    # Decimal
    "DECIMAL": "decimal",
    "NUMERIC": "decimal",
    # String
    "VARCHAR": "string",
    "CHAR": "string",
    "TEXT": "string",
    "STRING": "string",
    # Boolean
    "BOOLEAN": "bool",
    "BOOL": "bool",
    # Temporal
    "DATE": "date",
    "TIME": "time",
    "TIMESTAMP": "timestamp",
    "TIMESTAMP WITH TIME ZONE": "timestamp",
    "TIMESTAMPTZ": "timestamp",
    # Binary
    "BINARY": "binary",
    "VARBINARY": "binary",
    "BLOB": "binary",
    # Complex types
    "ARRAY": "array",
    "STRUCT": "struct",
    "MAP": "map",
    "JSON": "json",
    "JSONB": "json",
}


def sql_type_to_ibis(sql_type: str) -> str:
    """Convert SQL type name to Ibis dtype.

    Parameters
    ----------
    sql_type : str
        SQL type name (e.g., 'INTEGER', 'VARCHAR', 'TIMESTAMP')

    Returns
    -------
    str
        Ibis dtype string

    Examples
    --------
    >>> sql_type_to_ibis('INTEGER')
    'int32'
    >>> sql_type_to_ibis('DOUBLE')
    'float64'
    """
    sql_type_upper = sql_type.upper().strip()

    # Handle parameterized types (e.g., VARCHAR(255) → VARCHAR)
    base_type = sql_type_upper.split("(")[0].strip()

    if base_type in SQL_TO_IBIS_TYPES:
        return SQL_TO_IBIS_TYPES[base_type]

    # Default fallback
    return "string"


def needs_explicit_cast(from_type: str, to_type: str, strict: bool = True) -> bool:
    """Check if explicit cast needed between types.

    Parameters
    ----------
    from_type : str
        Source type
    to_type : str
        Target type
    strict : bool
        If True, require explicit casts for all non-identical types

    Returns
    -------
    bool
        True if .cast() needed

    Examples
    --------
    >>> needs_explicit_cast('int32', 'float64', strict=True)
    True
    >>> needs_explicit_cast('int32', 'int32', strict=True)
    False
    """
    if from_type == to_type:
        return False

    if not strict:
        # Allow some implicit promotions
        safe_promotions = {
            ("int8", "int16"),
            ("int8", "int32"),
            ("int8", "int64"),
            ("int16", "int32"),
            ("int16", "int64"),
            ("int32", "int64"),
            ("float32", "float64"),
        }
        if (from_type, to_type) in safe_promotions:
            return False

    return True


def generate_cast_expr(expr: str, target_type: str) -> str:
    """Generate Ibis cast expression.

    Parameters
    ----------
    expr : str
        Expression to cast
    target_type : str
        Target Ibis dtype

    Returns
    -------
    str
        Ibis cast expression

    Examples
    --------
    >>> generate_cast_expr('events.amount', 'float64')
    'events.amount.cast("float64")'
    """
    return f'{expr}.cast("{target_type}")'


class TypeInferenceError(Exception):
    """Raised when type cannot be inferred."""

    pass
