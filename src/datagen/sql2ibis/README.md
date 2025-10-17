# SQL→Ibis Training Dataset

Template-based synthetic data generation for training iBERT on SQL to Ibis DSL translation.

## Overview

This directory contains a complete pipeline for generating validated SQL→Ibis translation pairs. The system uses YAML-based templates with parameterized variations to create high-quality training examples where SQL and Ibis code produce provably identical results.

**Current Status**: 15 templates, 46 validated examples, 100% validation success rate

## Architecture

```
ibert-dev/
├── src/sql2ibis/            # SQL→Ibis pipeline
│   ├── templates/           # YAML template definitions (01-15)
│   ├── eval/                # Validation logic
│   │   └── validator.py
│   ├── template_loader/     # Template loading
│   └── generate_dataset.py  # Main generation script
└── data/sql2ibis/           # Generated datasets
    └── train.jsonl          # Training data (gitignored)
```

## Template System

### Template Structure

Each template is a YAML file with:

```yaml
name: template_name
description: What this template covers
difficulty: easy|medium|hard
features:
  - feature1
  - feature2

sql_template: |
  SQL query template with {placeholders}

ibis_template: |
  Ibis code template with {placeholders}

variations:
  - name: variation_name
    params:
      param1: value1
      param2: value2

context:
  tables:
    table_name:
      schema:
        column_name: dtype
```

### Validation Process

1. **Table Registration**: Creates test tables from schema definitions
2. **SQL Execution**: Runs SQL query against DuckDB backend
3. **Ibis Execution**: Evaluates Ibis expression with table references
4. **Result Comparison**: Compares DataFrames with numeric tolerance
5. **Multi-line Handling**: Separates imports/decorators from expressions

The validator handles:
- Single-line expressions
- Multi-line code with imports (`import ibis.selectors as s`)
- Decorator-based function definitions (`@ibis.udf.scalar.builtin`)

## Templates (1-15)

### Core SQL Operations

| Template | Features | Variations | Difficulty |
|----------|----------|------------|------------|
| **01_select_where** | SELECT, WHERE, filtering | 4 | Easy |
| **02_groupby_aggregate** | GROUP BY, aggregations | 4 | Easy |
| **03_filter_groupby** | Combined filter + aggregate | 3 | Medium |
| **04_order_limit** | ORDER BY, LIMIT, sorting | 4 | Easy |
| **05_join** | INNER/LEFT JOIN | 2 | Medium |
| **06_having_clause** | HAVING, post-agg filtering | 3 | Medium |
| **07_case_when** | CASE/WHEN conditionals | 1 | Medium |
| **08_window_functions** | ROW_NUMBER, RANK, windows | 2 | Hard |
| **09_distinct** | DISTINCT, unique values | 3 | Easy |
| **10_null_handling** | NULL checks, COALESCE | 2 | Easy |

### Advanced Features

| Template | Features | Variations | Difficulty |
|----------|----------|------------|------------|
| **11_temporal** | Date/time extraction | 6 | Medium |
| **12_selectors** | Dynamic column selection | 2 | Medium |
| **13_collections** | Array operations | 1 | Hard |
| **14_udf_scalar_builtin** | Scalar UDFs (abs, round, upper, lower) | 4 | Medium |
| **15_udf_agg_builtin** | Aggregate UDFs (avg, sum, count, max, min) | 5 | Medium |

**Total**: 46 validated examples

## Dataset Format

Generated examples are saved as JSONL with this structure:

```json
{
  "id": "uuid",
  "task": "sql_to_ibis",
  "dialect": "duckdb",
  "backend": "duckdb",
  "ibis_version": "9.5.0",
  "context": {
    "tables": {
      "events": {
        "schema": {
          "user_id": "int64",
          "event_ts": "timestamp",
          "amount": "float64"
        }
      }
    }
  },
  "input": {
    "sql": "SELECT user_id, amount FROM events WHERE amount > 10"
  },
  "target": {
    "ibis": "events.filter(events.amount > 10)[[\"user_id\", \"amount\"]]",
    "expr_name": "expr"
  },
  "meta": {
    "template": "select_where",
    "variation": "simple_numeric",
    "features": ["select", "where", "filter"],
    "source": "synthetic",
    "difficulty": "easy"
  }
}
```

### Fields

- **id**: Unique identifier (UUID)
- **task**: Always `sql_to_ibis`
- **dialect/backend**: DuckDB (could extend to other backends)
- **ibis_version**: Version used for generation
- **context**: Table schemas and metadata
- **input**: SQL query string
- **target**: Ibis expression code + variable name
- **meta**: Template info, features, difficulty

## Usage

### Generate Dataset

```bash
# Using justfile (from project root)
just generate-data

# Direct Python
.venv/bin/python src/sql2ibis/generate_dataset.py
```

### Show Statistics

```bash
just dataset-stats
```

Output:
```
Loaded 15 templates
Generated 46 examples
Results: 46/46 passed
Success rate: 100.0%
```

### Validation Output

The generator shows progress for each example:

```
Validating 1/46: select_where:simple_numeric... ✓
Validating 2/46: select_where:timestamp_filter... ✓
...
Validating 46/46: udf_agg_builtin:min_builtin... ✓
```

Failed examples show error details:
```
Validating 10/46: example:variation... ✗ Results differ
```

## Example Templates

### Simple Filter (Template 01)

```yaml
name: select_where
sql_template: |
  SELECT {columns}
  FROM {table}
  WHERE {condition}

ibis_template: |
  {table}.filter({ibis_condition})[{ibis_select_cols}]

variations:
  - name: simple_numeric
    params:
      table: events
      columns: user_id, amount
      condition: amount > 10
      ibis_condition: events.amount > 10
      ibis_select_cols: '["user_id", "amount"]'
```

### Window Function (Template 08)

```yaml
name: window_functions
sql_template: |
  SELECT {columns}, {window_expr}
  FROM {table}

ibis_template: |
  {table}.mutate({ibis_window})[{ibis_select_cols}]

variations:
  - name: row_number_by_user
    params:
      table: events
      columns: user_id, event_ts, amount
      window_expr: ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY event_ts) AS row_num
      ibis_window: row_num=(ibis.row_number().over(ibis.window(group_by="user_id", order_by="event_ts")) + 1)
      ibis_select_cols: '["user_id", "event_ts", "amount", "row_num"]'
```

### Built-in UDF (Template 14)

```yaml
name: udf_scalar_builtin
sql_template: |
  SELECT {columns}, {udf_expr}
  FROM {table}

ibis_template: |
  import ibis
  {udf_definition}
  {table}.mutate({ibis_udf})[{ibis_select_cols}]

variations:
  - name: abs_builtin
    params:
      table: events
      columns: user_id, amount
      udf_expr: ABS(amount) AS abs_amount
      udf_definition: |
        @ibis.udf.scalar.builtin(name="abs")
        def abs_val(x: float) -> float:
            """Absolute value."""
      ibis_udf: abs_amount=abs_val(events.amount)
      ibis_select_cols: '["user_id", "amount", "abs_amount"]'
```

## Validator Details

Located in `src/sql2ibis/eval/validator.py`

### Key Features

**Multi-line Code Handling** (lines 65-118):
- Detects imports and decorators
- Separates setup code from expression code
- Executes in proper order: imports → decorators/functions → expression

**Result Comparison** (lines 95-144):
- Sorts DataFrames for consistent comparison
- Handles numeric tolerance (1e-12)
- Type-aware comparison (numeric vs string)

### Example Validation Flow

```python
# 1. Register tables
validator.register_tables({
    "events": pd.DataFrame({
        "user_id": [1, 2, 3],
        "amount": [10.5, 20.3, 15.7]
    })
})

# 2. Validate example
success, error = validator.validate_example({
    "input": {"sql": "SELECT user_id FROM events WHERE amount > 15"},
    "target": {"ibis": "events.filter(events.amount > 15)[[\"user_id\"]]"},
    "context": {"tables": {"events": {"schema": {...}}}}
})

# 3. Returns (True, None) if results match
```

## Common Schemas

### events table
```python
{
  "user_id": "int64",
  "event_ts": "timestamp",
  "amount": "float64"
}
```

### users table
```python
{
  "user_id": "int64",
  "name": "string"
}
```

### labels table
```python
{
  "user_id": "int64",
  "label": "int8"
}
```

## Adding New Templates

1. **Create YAML file** in `templates/` (e.g., `16_your_template.yaml`)

2. **Define structure**:
```yaml
name: your_template
description: What it does
difficulty: easy|medium|hard
features: [feature1, feature2]

sql_template: |
  SQL with {placeholders}

ibis_template: |
  Ibis with {placeholders}

variations:
  - name: variation1
    params:
      param1: value1

context:
  tables:
    table_name:
      schema:
        col: dtype
```

3. **Test**:
```bash
just generate-data
```

4. **Validate**: Ensure 100% pass rate

### Template Best Practices

- **Use existing tables**: Reuse `events`, `users`, `labels` schemas
- **Start simple**: Add one variation, validate, then add more
- **Test edge cases**: NULL values, empty results, type mismatches
- **Document features**: Tag with relevant SQL/Ibis features
- **Match output**: Ensure SQL and Ibis column order matches

## Troubleshooting

### "Results differ" Error

**Cause**: SQL and Ibis produce different DataFrames

**Debug**:
1. Check column names match exactly
2. Verify column order (use explicit column lists)
3. Check for floating-point precision issues
4. Ensure both queries use same tables

### "invalid syntax" Error

**Cause**: Python code won't parse

**Debug**:
1. Check indentation in YAML (use `|` for multi-line)
2. Verify decorator syntax is correct
3. Test Ibis code in isolation
4. Check for missing imports

### Import/Decorator Issues

**Solution**: Validator automatically handles:
- `import ibis.selectors as s`
- `@ibis.udf.scalar.builtin(name="func")`
- `@ibis.udf.agg.builtin(name="func")`

Just ensure decorators come before function definitions.

## Performance

- **Generation**: ~1-2 seconds for 46 examples
- **Validation**: ~20ms per example average
- **Memory**: Minimal (test data is small)

## Future Work

### Missing SQL Features

- [ ] UNION/INTERSECT/EXCEPT
- [ ] CTEs (WITH clause)
- [ ] Subqueries (correlated and uncorrelated)
- [ ] CROSS JOIN, FULL OUTER JOIN
- [ ] More window functions (LAG, LEAD, NTILE)
- [ ] String functions (CONCAT, SUBSTRING, REGEXP)
- [ ] Math functions (CEIL, FLOOR, MOD)
- [ ] PIVOT/UNPIVOT operations

### Template Expansion

- [ ] Add 5+ variations per template
- [ ] Multi-table joins (3+ tables)
- [ ] Complex WHERE clauses (OR, NOT, IN, BETWEEN)
- [ ] Nested aggregations
- [ ] Mixed data types

### Validation Enhancements

- [ ] Support for approximate results (statistical tests)
- [ ] Multi-backend validation (Postgres, Snowflake, BigQuery)
- [ ] Performance benchmarking
- [ ] Error injection for negative examples

### Dataset Growth

**Current**: 46 examples
**Target**: 500+ examples covering 90% of common SQL patterns

## Technical Notes

### Why DuckDB?

- Fast in-process SQL engine
- No server setup required
- Excellent SQL dialect coverage
- Native Ibis backend support

### Why Template-Based?

- **Scalability**: Add variations without writing full examples
- **Validation**: Automatic correctness checking
- **Consistency**: Standardized format and metadata
- **Coverage**: Systematic exploration of SQL feature space

### Built-in UDF Insight

`@ibis.udf.scalar.builtin(name="abs")` is a **mapping**, not registration:
- Maps Python wrapper to existing backend function
- No DB-side registration needed
- Single-phase validation works
- Both SQL and Ibis compile to same backend call

This allows UDF templates to validate like any other template.

## References

- [Ibis Documentation](https://ibis-project.org)
- [DuckDB SQL Reference](https://duckdb.org/docs/sql/introduction)
- [Ibis Selectors](https://ibis-project.org/reference/selectors)
- [Ibis UDFs](https://ibis-project.org/reference/scalar-udfs)

## License

Apache 2.0 (matches iBERT project license)
