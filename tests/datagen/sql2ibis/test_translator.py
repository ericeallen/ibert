"""Tests for SQL translation utilities (parser and type system)."""

import pytest
from sqlglot import Expression

from src.datagen.sql2ibis.translator.parser import (
    parse_sql,
    normalize_sql,
    UnsupportedDialectError,
    SQLParseError
)
from src.datagen.sql2ibis.translator.types import (
    sql_type_to_ibis,
    needs_explicit_cast,
    generate_cast_expr,
    TypeInferenceError,
    SQL_TO_IBIS_TYPES
)


class TestSQLParser:
    """Test suite for SQL parsing functions."""

    def test_parse_sql_simple_select(self):
        """Test parsing simple SELECT statement."""
        ast, dialect = parse_sql("SELECT * FROM events", "duckdb")

        assert isinstance(ast, Expression)
        assert dialect == "duckdb"
        assert "events" in ast.sql()

    def test_parse_sql_with_where_clause(self):
        """Test parsing SELECT with WHERE."""
        sql = "SELECT id, name FROM users WHERE age > 18"
        ast, dialect = parse_sql(sql, "duckdb")

        assert isinstance(ast, Expression)
        assert "users" in ast.sql()
        assert "age" in ast.sql()

    def test_parse_sql_complex_query(self):
        """Test parsing complex query with JOIN and GROUP BY."""
        sql = """
        SELECT u.name, COUNT(*) as cnt
        FROM users u
        JOIN events e ON u.id = e.user_id
        GROUP BY u.name
        """
        ast, dialect = parse_sql(sql, "duckdb")

        assert isinstance(ast, Expression)
        rendered = ast.sql()
        assert "JOIN" in rendered or "join" in rendered.lower()

    def test_parse_sql_postgres_dialect(self):
        """Test parsing with PostgreSQL dialect."""
        sql = "SELECT * FROM events LIMIT 10"
        ast, dialect = parse_sql(sql, "postgres")

        assert dialect == "postgres"
        assert isinstance(ast, Expression)

    def test_parse_sql_postgresql_not_supported(self):
        """Test 'postgresql' is not a supported alias (use 'postgres')."""
        # sqlglot doesn't accept 'postgresql', only 'postgres'
        with pytest.raises((UnsupportedDialectError, SQLParseError)):
            parse_sql("SELECT 1", "postgresql")

    def test_parse_sql_mysql_dialect(self):
        """Test parsing with MySQL dialect."""
        ast, dialect = parse_sql("SELECT 1", "mysql")

        assert dialect == "mysql"
        assert isinstance(ast, Expression)

    def test_parse_sql_bigquery_dialect(self):
        """Test parsing with BigQuery dialect."""
        ast, dialect = parse_sql("SELECT 1", "bigquery")

        assert dialect == "bigquery"
        assert isinstance(ast, Expression)

    def test_parse_sql_snowflake_dialect(self):
        """Test parsing with Snowflake dialect."""
        ast, dialect = parse_sql("SELECT 1", "snowflake")

        assert dialect == "snowflake"
        assert isinstance(ast, Expression)

    def test_parse_sql_spark_dialect(self):
        """Test parsing with Spark dialect."""
        ast, dialect = parse_sql("SELECT 1", "spark")

        assert dialect == "spark"
        assert isinstance(ast, Expression)

    def test_parse_sql_sqlite_dialect(self):
        """Test parsing with SQLite dialect."""
        ast, dialect = parse_sql("SELECT 1", "sqlite")

        assert dialect == "sqlite"
        assert isinstance(ast, Expression)

    def test_parse_sql_unsupported_dialect_raises(self):
        """Test unsupported dialect raises UnsupportedDialectError."""
        with pytest.raises(UnsupportedDialectError) as exc_info:
            parse_sql("SELECT 1", "oracle")

        assert "oracle" in str(exc_info.value).lower()
        assert "Supported:" in str(exc_info.value)

    def test_parse_sql_permissive_parsing(self):
        """Test sqlglot is permissive with invalid SQL."""
        # sqlglot doesn't raise errors for typos, it interprets them
        invalid_sql = "SELCT * FORM table"  # Typos

        # This will parse without error (sqlglot is very permissive)
        ast, dialect = parse_sql(invalid_sql, "duckdb")
        assert isinstance(ast, Expression)

    def test_parse_sql_empty_string(self):
        """Test parsing empty string raises error."""
        with pytest.raises(SQLParseError) as exc_info:
            parse_sql("", "duckdb")

        assert "No expression was parsed" in str(exc_info.value)

    def test_normalize_sql_simple(self):
        """Test SQL normalization."""
        sql = "select * from events where id=1"
        normalized = normalize_sql(sql, "duckdb")

        assert isinstance(normalized, str)
        assert len(normalized) > 0
        # Normalized SQL should be properly formatted
        assert "SELECT" in normalized or "select" in normalized

    def test_normalize_sql_formats_query(self):
        """Test normalization formats query nicely."""
        sql = "SELECT a,b,c FROM t WHERE x>5 AND y<10"
        normalized = normalize_sql(sql, "duckdb")

        # Should have newlines from pretty=True
        assert "\n" in normalized or len(normalized.split()) > 1

    def test_normalize_sql_different_dialects(self):
        """Test normalization works across dialects."""
        sql = "SELECT * FROM table1"

        norm_duck = normalize_sql(sql, "duckdb")
        norm_pg = normalize_sql(sql, "postgres")

        assert isinstance(norm_duck, str)
        assert isinstance(norm_pg, str)

    def test_normalize_sql_permissive(self):
        """Test normalizing is permissive with unusual SQL."""
        # sqlglot is very permissive
        normalized = normalize_sql("INVALID SQL", "duckdb")
        assert isinstance(normalized, str)


class TestTypeConversion:
    """Test suite for SQL type conversion functions."""

    def test_sql_type_to_ibis_integer_types(self):
        """Test integer type conversions."""
        assert sql_type_to_ibis("TINYINT") == "int8"
        assert sql_type_to_ibis("SMALLINT") == "int16"
        assert sql_type_to_ibis("INTEGER") == "int32"
        assert sql_type_to_ibis("INT") == "int32"
        assert sql_type_to_ibis("BIGINT") == "int64"

    def test_sql_type_to_ibis_float_types(self):
        """Test floating point type conversions."""
        assert sql_type_to_ibis("REAL") == "float32"
        assert sql_type_to_ibis("FLOAT") == "float32"
        assert sql_type_to_ibis("DOUBLE") == "float64"
        assert sql_type_to_ibis("DOUBLE PRECISION") == "float64"

    def test_sql_type_to_ibis_decimal_types(self):
        """Test decimal type conversions."""
        assert sql_type_to_ibis("DECIMAL") == "decimal"
        assert sql_type_to_ibis("NUMERIC") == "decimal"

    def test_sql_type_to_ibis_string_types(self):
        """Test string type conversions."""
        assert sql_type_to_ibis("VARCHAR") == "string"
        assert sql_type_to_ibis("CHAR") == "string"
        assert sql_type_to_ibis("TEXT") == "string"
        assert sql_type_to_ibis("STRING") == "string"

    def test_sql_type_to_ibis_boolean_types(self):
        """Test boolean type conversions."""
        assert sql_type_to_ibis("BOOLEAN") == "bool"
        assert sql_type_to_ibis("BOOL") == "bool"

    def test_sql_type_to_ibis_temporal_types(self):
        """Test temporal type conversions."""
        assert sql_type_to_ibis("DATE") == "date"
        assert sql_type_to_ibis("TIME") == "time"
        assert sql_type_to_ibis("TIMESTAMP") == "timestamp"
        assert sql_type_to_ibis("TIMESTAMP WITH TIME ZONE") == "timestamp"
        assert sql_type_to_ibis("TIMESTAMPTZ") == "timestamp"

    def test_sql_type_to_ibis_binary_types(self):
        """Test binary type conversions."""
        assert sql_type_to_ibis("BINARY") == "binary"
        assert sql_type_to_ibis("VARBINARY") == "binary"
        assert sql_type_to_ibis("BLOB") == "binary"

    def test_sql_type_to_ibis_complex_types(self):
        """Test complex type conversions."""
        assert sql_type_to_ibis("ARRAY") == "array"
        assert sql_type_to_ibis("STRUCT") == "struct"
        assert sql_type_to_ibis("MAP") == "map"
        assert sql_type_to_ibis("JSON") == "json"
        assert sql_type_to_ibis("JSONB") == "json"

    def test_sql_type_to_ibis_case_insensitive(self):
        """Test type conversion is case insensitive."""
        assert sql_type_to_ibis("integer") == "int32"
        assert sql_type_to_ibis("Integer") == "int32"
        assert sql_type_to_ibis("INTEGER") == "int32"

    def test_sql_type_to_ibis_with_whitespace(self):
        """Test type conversion handles whitespace."""
        assert sql_type_to_ibis("  INTEGER  ") == "int32"
        assert sql_type_to_ibis("DOUBLE PRECISION") == "float64"

    def test_sql_type_to_ibis_parameterized_types(self):
        """Test conversion strips parameters."""
        assert sql_type_to_ibis("VARCHAR(255)") == "string"
        assert sql_type_to_ibis("DECIMAL(10,2)") == "decimal"
        assert sql_type_to_ibis("CHAR(10)") == "string"

    def test_sql_type_to_ibis_unknown_type_defaults(self):
        """Test unknown types default to string."""
        assert sql_type_to_ibis("UNKNOWN_TYPE") == "string"
        assert sql_type_to_ibis("CUSTOM") == "string"

    def test_sql_to_ibis_types_constant_exists(self):
        """Test SQL_TO_IBIS_TYPES constant is defined."""
        assert isinstance(SQL_TO_IBIS_TYPES, dict)
        assert len(SQL_TO_IBIS_TYPES) > 0
        assert "INTEGER" in SQL_TO_IBIS_TYPES
        assert "VARCHAR" in SQL_TO_IBIS_TYPES


class TestTypeCasting:
    """Test suite for type casting utilities."""

    def test_needs_explicit_cast_same_type(self):
        """Test no cast needed for same type."""
        assert not needs_explicit_cast("int32", "int32", strict=True)
        assert not needs_explicit_cast("float64", "float64", strict=True)
        assert not needs_explicit_cast("string", "string", strict=False)

    def test_needs_explicit_cast_different_types_strict(self):
        """Test cast needed for different types in strict mode."""
        assert needs_explicit_cast("int32", "float64", strict=True)
        assert needs_explicit_cast("string", "int32", strict=True)
        assert needs_explicit_cast("bool", "int8", strict=True)

    def test_needs_explicit_cast_safe_promotions_non_strict(self):
        """Test safe promotions allowed in non-strict mode."""
        # int8 → int16/int32/int64
        assert not needs_explicit_cast("int8", "int16", strict=False)
        assert not needs_explicit_cast("int8", "int32", strict=False)
        assert not needs_explicit_cast("int8", "int64", strict=False)

        # int16 → int32/int64
        assert not needs_explicit_cast("int16", "int32", strict=False)
        assert not needs_explicit_cast("int16", "int64", strict=False)

        # int32 → int64
        assert not needs_explicit_cast("int32", "int64", strict=False)

        # float32 → float64
        assert not needs_explicit_cast("float32", "float64", strict=False)

    def test_needs_explicit_cast_unsafe_promotions_non_strict(self):
        """Test unsafe promotions still need cast in non-strict mode."""
        # Reverse direction not safe
        assert needs_explicit_cast("int64", "int32", strict=False)
        assert needs_explicit_cast("float64", "float32", strict=False)

        # Cross-category not safe
        assert needs_explicit_cast("int32", "float32", strict=False)
        assert needs_explicit_cast("float64", "int64", strict=False)

    def test_generate_cast_expr_simple(self):
        """Test generating simple cast expression."""
        result = generate_cast_expr("events.amount", "float64")

        assert result == 'events.amount.cast("float64")'

    def test_generate_cast_expr_various_types(self):
        """Test cast expression for various types."""
        assert generate_cast_expr("col", "int32") == 'col.cast("int32")'
        assert generate_cast_expr("t.x", "string") == 't.x.cast("string")'
        assert generate_cast_expr("expr", "timestamp") == 'expr.cast("timestamp")'

    def test_generate_cast_expr_complex_expression(self):
        """Test cast for complex expression."""
        expr = "table.filter(table.age > 18).amount"
        result = generate_cast_expr(expr, "decimal")

        expected = 'table.filter(table.age > 18).amount.cast("decimal")'
        assert result == expected

    def test_type_inference_error_exception(self):
        """Test TypeInferenceError can be raised."""
        with pytest.raises(TypeInferenceError):
            raise TypeInferenceError("Cannot infer type")

    def test_type_inference_error_is_exception(self):
        """Test TypeInferenceError is an Exception."""
        assert issubclass(TypeInferenceError, Exception)


class TestExceptionClasses:
    """Test suite for custom exception classes."""

    def test_unsupported_dialect_error_is_exception(self):
        """Test UnsupportedDialectError is an Exception."""
        assert issubclass(UnsupportedDialectError, Exception)

    def test_unsupported_dialect_error_can_raise(self):
        """Test UnsupportedDialectError can be raised with message."""
        with pytest.raises(UnsupportedDialectError) as exc_info:
            raise UnsupportedDialectError("Test dialect not supported")

        assert "Test dialect" in str(exc_info.value)

    def test_sql_parse_error_is_exception(self):
        """Test SQLParseError is an Exception."""
        assert issubclass(SQLParseError, Exception)

    def test_sql_parse_error_can_raise(self):
        """Test SQLParseError can be raised with message."""
        with pytest.raises(SQLParseError) as exc_info:
            raise SQLParseError("Failed to parse query")

        assert "Failed to parse" in str(exc_info.value)

    def test_type_inference_error_can_be_caught(self):
        """Test TypeInferenceError can be caught."""
        try:
            raise TypeInferenceError("Test error")
        except TypeInferenceError as e:
            assert "Test error" in str(e)
        except Exception:
            pytest.fail("Should catch as TypeInferenceError")


class TestIntegration:
    """Integration tests for translator modules."""

    def test_parse_and_extract_table_names(self):
        """Test parsing SQL and extracting table info."""
        sql = "SELECT * FROM users JOIN events ON users.id = events.user_id"
        ast, dialect = parse_sql(sql, "duckdb")

        rendered = ast.sql(dialect=dialect)
        assert "users" in rendered
        assert "events" in rendered

    def test_type_conversion_pipeline(self):
        """Test complete type conversion pipeline."""
        # SQL type → Ibis type
        sql_type = "INTEGER"
        ibis_type = sql_type_to_ibis(sql_type)
        assert ibis_type == "int32"

        # Check if cast needed
        if needs_explicit_cast(ibis_type, "float64", strict=True):
            cast_expr = generate_cast_expr("column", "float64")
            assert ".cast(" in cast_expr

    def test_normalize_complex_query(self):
        """Test normalizing a complex query."""
        sql = """
        SELECT
            user_id,
            COUNT(*) as event_count,
            AVG(amount) as avg_amount
        FROM events
        WHERE timestamp > '2024-01-01'
        GROUP BY user_id
        HAVING COUNT(*) > 5
        """

        normalized = normalize_sql(sql, "duckdb")

        assert isinstance(normalized, str)
        assert "user_id" in normalized
        assert "event_count" in normalized or "COUNT" in normalized

    def test_all_sql_types_have_mappings(self):
        """Test all SQL types in constant can be converted."""
        for sql_type in SQL_TO_IBIS_TYPES.keys():
            ibis_type = sql_type_to_ibis(sql_type)
            assert isinstance(ibis_type, str)
            assert len(ibis_type) > 0

    def test_dialect_specific_parsing(self):
        """Test parsing works across different dialects."""
        sql = "SELECT * FROM events LIMIT 10"

        dialects_to_test = ["duckdb", "postgres", "mysql", "sqlite"]

        for dialect in dialects_to_test:
            ast, returned_dialect = parse_sql(sql, dialect)
            assert returned_dialect == dialect
            assert isinstance(ast, Expression)
