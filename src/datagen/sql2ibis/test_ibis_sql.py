"""Test Ibis native SQL→Ibis translation capabilities."""

import ibis
import pandas as pd

# Create test data
df = pd.DataFrame(
    {
        "user_id": [1, 2, 3, 1, 2],
        "event_ts": pd.to_datetime(
            ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02", "2024-01-03"]
        ),
        "amount": [10.0, 20.0, 30.0, 15.0, 25.0],
    }
)

# Connect to DuckDB and register table
con = ibis.duckdb.connect()
events = con.create_table("events", df)

print("=" * 60)
print("TEST 1: Simple SELECT with WHERE")
print("=" * 60)

sql1 = "SELECT user_id, amount FROM events WHERE amount > 15"
expr1 = con.sql(sql1)
print(f"SQL: {sql1}")
print(f"\nIbis expression type: {type(expr1)}")
print(f"\nResult:\n{expr1.execute()}")
print(f"\nCompiled back to SQL:\n{ibis.to_sql(expr1)}")

print("\n" + "=" * 60)
print("TEST 2: GROUP BY with aggregation")
print("=" * 60)

sql2 = "SELECT user_id, COUNT(*) as n, SUM(amount) as total FROM events GROUP BY user_id"
expr2 = con.sql(sql2)
print(f"SQL: {sql2}")
print(f"\nResult:\n{expr2.execute()}")
print(f"\nCompiled back to SQL:\n{ibis.to_sql(expr2)}")

print("\n" + "=" * 60)
print("TEST 3: Can we extract Ibis code from expression?")
print("=" * 60)

# Try to get a code representation
print(f"repr(expr2):\n{repr(expr2)}")
print("\ndir(expr2) [relevant methods]:")
for attr in dir(expr2):
    if (
        not attr.startswith("_")
        and "sql" in attr.lower()
        or "code" in attr.lower()
        or "str" in attr.lower()
    ):
        print(f"  - {attr}")

print("\n" + "=" * 60)
print("TEST 4: Check if expressions are equivalent")
print("=" * 60)

# Build same query with Ibis API
expr2_ibis = events.group_by("user_id").aggregate(n=events.count(), total=events.amount.sum())

print("SQL-generated SQL:")
print(ibis.to_sql(expr2))
print("\nIbis-API-generated SQL:")
print(ibis.to_sql(expr2_ibis))
print("\nResults match:", expr2.execute().equals(expr2_ibis.execute()))

print("\n" + "=" * 60)
print("CONCLUSION")
print("=" * 60)
print("✓ con.sql() works: Converts SQL string to Ibis expression")
print("✓ ibis.to_sql() works: Converts Ibis expression back to SQL")
print("✓ Expressions execute correctly")
print("✗ No direct way to get 'Ibis code string' from expression")
print("\nIMPLICATION:")
print("We need to build our own code generator from the Ibis expression object,")
print("OR use sql-to-ibis (if we can fix Python 3.13 compat),")
print("OR generate synthetic pairs: SQL template → hand-write Ibis code")
