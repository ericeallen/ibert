# SQL→Ibis Translation: Tool Research Findings

## Executive Summary

After investigating existing tools, here's what we found:

### ✅ What Exists

1. **Ibis Native `con.sql()` method**
   - Converts SQL string → Ibis expression object
   - Roundtrip: SQL → Ibis expr → `ibis.to_sql()` → SQL
   - **Problem**: No way to get "Ibis Python code" from expression

2. **sql-to-ibis package** (zbrookle)
   - Dedicated SQL→Ibis translator using Lark grammar
   - **Problem**: Python 3.13 incompatibility (`SafeConfigParser` removed)
   - **Problem**: Not actively maintained (last update 2020)

3. **sqlglot** (underlying Ibis compiler)
   - SQL parser/transpiler
   - Ibis uses this internally
   - **Problem**: Generates SQL, not Ibis Python code

### ❌ What Doesn't Exist

**No tool generates "Ibis Python code strings" from SQL.**

All existing tools produce:
- Ibis expression **objects** (runtime)
- SQL strings (via compilation)

But NOT:
- Ibis **source code** (for training data)

## The Core Problem

For iBERT training, we need pairs like:

```json
{
  "input": "SELECT user_id, COUNT(*) FROM events GROUP BY user_id",
  "target": "events.group_by('user_id').aggregate(n=events.user_id.count())"
}
```

Existing tools give us:
```python
con.sql("SELECT ...") #  → <ibis.expr.types.Table object>
ibis.to_sql(expr)      # → "SELECT ..." (SQL string)
```

But NOT:
```python
get_ibis_code(expr) # → "events.group_by(...)" ❌ DOESN'T EXIST
```

## Options Moving Forward

### Option A: Synthetic Generation (RECOMMENDED)
**Manually create (SQL, Ibis code) pairs programmatically**

```python
# Template-based generation
templates = [
    {
        "sql": "SELECT {cols} FROM {table} WHERE {pred}",
        "ibis": "{table}.filter({ibis_pred})[{ibis_cols}]",
        "params": {
            "cols": ["user_id", "amount"],
            "table": "events",
            "pred": "amount > 10",
            "ibis_pred": "events.amount > 10",
            "ibis_cols": '["user_id", "amount"]'
        }
    }
]
```

**Pros:**
- Full control over quality
- Can target specific operators/patterns
- Built-in validation (we write both sides)
- No dependency issues

**Cons:**
- Manual effort to create templates
- Covers common cases, not edge cases

### Option B: Expression Introspection
**Build code generator from Ibis expression AST**

```python
def expr_to_code(expr: ibis.expr.types.Table) -> str:
    # Introspect expression graph
    # Walk nodes, emit code
    # Similar to what we were building before
    pass
```

**Pros:**
- Reuses `con.sql()` for parsing
- Generates "real" Ibis code

**Cons:**
- Still building AST walker (complex)
- Ibis expression API is internal/unstable

### Option C: Use sql-to-ibis with Python 3.10
**Downgrade Python to use existing package**

**Pros:**
- Leverage existing translator
- Proven grammar

**Cons:**
- Forces Python 3.10 constraint
- Package unmaintained
- Still doesn't give us code strings (returns objects)

### Option D: Hybrid Approach (PRAGMATIC)
**Synthetic generation + validation via Ibis**

1. **Generate SQL** using templates
2. **Hand-write Ibis code** for each template
3. **Validate equivalence**:
   ```python
   sql_expr = con.sql(sql_template)
   ibis_expr = eval(ibis_code_template)
   assert ibis.to_sql(sql_expr) == ibis.to_sql(ibis_expr)
   ```
4. **Mutate parameters** to scale dataset

**Pros:**
- Best of both worlds
- Validation ensures correctness
- Scalable via mutations
- No fragile dependencies

**Cons:**
- Initial template creation effort

## Recommendation

**Go with Option D: Hybrid Approach**

### Implementation Plan

1. **Create 20-30 core templates** covering:
   - SELECT/WHERE/LIMIT
   - GROUP BY + aggregates
   - JOINs
   - Window functions
   - CTEs
   - CASE/WHEN

2. **For each template, manually write**:
   - SQL pattern with placeholders
   - Ibis code pattern with placeholders
   - Parameter variations

3. **Generate 1000+ examples** via:
   - Different table/column names
   - Different predicates
   - Different aggregation functions
   - Different join types

4. **Validate every pair**:
   - SQL → Ibis expr (via `con.sql()`)
   - Ibis code → Ibis expr (via `eval()`)
   - Compare: `ibis.to_sql(expr1) == ibis.to_sql(expr2)`
   - Execute both on fixtures, compare results

5. **Package as JSONL** with metadata

### Estimated Effort

- **Templates creation**: 2-3 days (20-30 templates)
- **Validation harness**: 1 day
- **Generation pipeline**: 1 day
- **Quality checking**: 1 day

**Total: ~1 week to 1000+ validated examples**

### Why This Works

1. **Quality over quantity**: Hand-crafted templates ensure idiomatic Ibis
2. **Validation**: Every pair proven equivalent
3. **Scalable**: Mutations amplify templates 50x-100x
4. **Maintainable**: No complex dependencies
5. **Extensible**: Add templates for edge cases incrementally

## Next Steps

1. Create `templates/` directory with YAML definitions
2. Build template renderer + mutator
3. Build validation harness (DuckDB execution)
4. Generate initial 100 examples from 5 templates
5. Measure quality metrics
6. Scale to 1000+ examples

---

**Decision**: Proceed with Hybrid Synthetic Generation approach.
