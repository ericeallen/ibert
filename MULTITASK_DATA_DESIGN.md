# Multi-Task Training Data Generation Design

## Overview

This document outlines the design for generating training data across all 6 tasks in iBERT's multi-task system.

## Task Types and Data Format

### 1. Code Completion (`code_completion`)
**Input**: Partial Ibis code
**Output**: Completed Ibis code
**System Prompt**: "Complete the partial Ibis expression. Output only the completed code."

**Example**:
```json
{
  "id": "uuid",
  "task": "code_completion",
  "context": {"tables": {"table": {"schema": {"age": "int64"}}}},
  "input": {"partial_code": "table.filter(table.age >"},
  "target": {"completed_code": "table.filter(table.age > 18)"},
  "meta": {"template": "filter_completion", "difficulty": "easy"}
}
```

### 2. SQL→Ibis Translation (`sql_to_ibis`)
**Input**: SQL query
**Output**: Equivalent Ibis code
**System Prompt**: "Translate the SQL query to equivalent Ibis code."

**Example** (already exists):
```json
{
  "id": "uuid",
  "task": "sql_to_ibis",
  "input": {"sql": "SELECT * FROM table WHERE age > 18"},
  "target": {"ibis": "table.filter(table.age > 18)"},
  ...
}
```

### 3. Ibis→SQL Translation (`ibis_to_sql`)
**Input**: Ibis code + optional dialect
**Output**: SQL query
**System Prompt**: "Translate the Ibis expression to SQL for the specified dialect."

**Example**:
```json
{
  "id": "uuid",
  "task": "ibis_to_sql",
  "dialect": "duckdb",
  "context": {"tables": {"table": {"schema": ...}}},
  "input": {"ibis": "table.filter(table.age > 18)", "dialect": "duckdb"},
  "target": {"sql": "SELECT *\\nFROM table\\nWHERE age > 18"},
  "meta": {"template": "filter_to_sql", "difficulty": "easy"}
}
```

### 4. Error Resolution (`error_resolution`)
**Input**: Broken Ibis code + error message (optional) + context
**Output**: Fixed Ibis code + explanation
**System Prompt**: "Fix the compilation or type error in the Ibis code."

**Example**:
```json
{
  "id": "uuid",
  "task": "error_resolution",
  "context": {"tables": {"table": {"schema": {"age": "int64"}}}},
  "input": {
    "broken_code": "table.filter(table.age > '18')",
    "error": "TypeError: '>' not supported between int and str",
    "context_info": "age column is int64 type"
  },
  "target": {
    "fixed_code": "table.filter(table.age > 18)",
    "explanation": "Removed quotes from 18 to compare as integer"
  },
  "meta": {"template": "type_error", "error_type": "type_mismatch"}
}
```

### 5. Q&A (`qa`)
**Input**: Question about Ibis
**Output**: Answer with optional code examples
**System Prompt**: "Answer questions about Ibis clearly and concisely."

**Example**:
```json
{
  "id": "uuid",
  "task": "qa",
  "input": {"question": "How do I filter rows where age is greater than 18?"},
  "target": {
    "answer": "Use the filter() method with a comparison expression",
    "code_example": "table.filter(table.age > 18)"
  },
  "meta": {"template": "how_to_filter", "topic": "filtering"}
}
```

### 6. Function Documentation (`documentation`)
**Input**: Ibis code (function or expression)
**Output**: Docstring in specified style
**System Prompt**: "Generate a docstring for the Ibis function in {style} format."

**Example**:
```json
{
  "id": "uuid",
  "task": "documentation",
  "input": {
    "code": "def get_active_users(table):\\n    return table.filter(table.is_active)",
    "style": "google"
  },
  "target": {
    "docstring": "\"\"\"Get active users from table.\\n\\nArgs:\\n    table: Input Ibis table\\n\\nReturns:\\n    Filtered table with only active users\"\"\""
  },
  "meta": {"template": "simple_filter_func", "style": "google"}
}
```

## Template Structure

Each task type has its own template directory:
```
src/datagen/multitask/templates/
├── code_completion/
│   ├── 01_filter_completion.yaml
│   ├── 02_aggregation_completion.yaml
│   ├── 03_join_completion.yaml
│   └── ...
├── ibis_to_sql/
│   ├── 01_filter_to_sql.yaml
│   ├── 02_groupby_to_sql.yaml
│   └── ...
├── error_resolution/
│   ├── 01_type_errors.yaml
│   ├── 02_syntax_errors.yaml
│   ├── 03_logical_errors.yaml
│   └── ...
├── qa/
│   ├── 01_basic_operations.yaml
│   ├── 02_aggregations.yaml
│   ├── 03_joins.yaml
│   └── ...
└── documentation/
    ├── 01_filter_functions.yaml
    ├── 02_aggregation_functions.yaml
    └── ...
```

## Data Generation Pipeline

### Phase 1: Template Loading
1. Load all templates from each task directory
2. Parse YAML and extract variations
3. Validate template structure

### Phase 2: Example Generation
For each template variation:
1. Generate input based on task type
2. Generate target output
3. Add context (table schemas, etc.)
4. Add metadata (difficulty, features, etc.)
5. Assign unique ID

### Phase 3: Validation
1. **Code Completion**: Validate completed code compiles
2. **SQL→Ibis**: Execute both and compare results (existing)
3. **Ibis→SQL**: Parse SQL and validate syntax
4. **Error Resolution**: Verify fixed code compiles
5. **Q&A**: Manual review or LLM-assisted validation
6. **Documentation**: Verify docstring format

### Phase 4: Output Generation
1. Write validated examples to task-specific JSONL files:
   - `data/multitask/code_completion.jsonl`
   - `data/multitask/sql_to_ibis.jsonl` (symlink to existing)
   - `data/multitask/ibis_to_sql.jsonl`
   - `data/multitask/error_resolution.jsonl`
   - `data/multitask/qa.jsonl`
   - `data/multitask/documentation.jsonl`

2. Concatenate all into `data/multitask/train_complete.jsonl`

## Template Coverage Goals

| Task | Target Templates | Target Examples |
|------|------------------|-----------------|
| Code Completion | 15-20 | 500-1000 |
| SQL→Ibis | 42 (existing) | 1000+ (existing) |
| Ibis→SQL | 15-20 | 500-1000 |
| Error Resolution | 10-15 | 300-500 |
| Q&A | 20-30 | 200-400 |
| Documentation | 10-15 | 200-400 |
| **Total** | **112-142** | **2700-4300+** |

## Key Principles

1. **Task Clarity**: Each example clearly indicates task type
2. **System Prompts**: Include task-specific instructions
3. **Output Format**: Specify expected output format in prompts
4. **Validation**: All examples must pass automated validation
5. **Diversity**: Cover easy/medium/hard difficulty levels
6. **Realism**: Examples should reflect real-world use cases

## Implementation Priority

1. ✅ Code Completion templates (highest ROI)
2. ✅ Ibis→SQL templates (reverse of existing)
3. Error Resolution templates (high value for users)
4. Q&A templates (knowledge transfer)
5. Documentation templates (code quality)

## Future Enhancements

- **Data Augmentation**: Use LLM to generate variations
- **Mining**: Extract examples from real codebases
- **Difficulty Balancing**: Ensure even distribution
- **Multi-turn Q&A**: Conversational examples
- **Chain-of-thought**: Add reasoning steps for complex examples
