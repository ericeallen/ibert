"""Tests for SQL→Ibis template loading and rendering system."""

import pytest

from src.datagen.sql2ibis.template_loader.expander import (
    ParameterSpaceConfig,
    apply_substitutions,
    create_column_variations,
    expand_parameter_space,
    expand_template_variations,
)
from src.datagen.sql2ibis.template_loader.loader import Template, generate_examples, load_templates


class TestParameterSpaceExpander:
    """Test suite for parameter space expansion functions."""

    def test_expand_parameter_space_simple(self):
        """Test expanding simple parameter space."""
        param_space = {"col": ["age", "price"], "op": [">", "<"]}

        combinations = list(expand_parameter_space(param_space))

        assert len(combinations) == 4  # 2 x 2
        assert {"col": "age", "op": ">"} in combinations
        assert {"col": "age", "op": "<"} in combinations
        assert {"col": "price", "op": ">"} in combinations
        assert {"col": "price", "op": "<"} in combinations

    def test_expand_parameter_space_empty(self):
        """Test expanding empty parameter space."""
        combinations = list(expand_parameter_space({}))

        assert len(combinations) == 1
        assert combinations[0] == {}

    def test_expand_parameter_space_single_param(self):
        """Test expanding parameter space with single parameter."""
        param_space = {"value": [10, 20, 30]}

        combinations = list(expand_parameter_space(param_space))

        assert len(combinations) == 3
        assert all("value" in c for c in combinations)

    def test_expand_parameter_space_three_dimensions(self):
        """Test expanding multi-dimensional parameter space."""
        param_space = {"col": ["a", "b"], "op": [">", "<", "=="], "val": [10, 20]}

        combinations = list(expand_parameter_space(param_space))

        assert len(combinations) == 12  # 2 x 3 x 2

    def test_expand_template_variations_simple(self):
        """Test expanding template variations."""
        base_variation = {"name": "filter", "params": {"table": "events"}}
        param_space = {"col": ["age", "amount"], "threshold": [10, 20]}

        variations = expand_template_variations(base_variation, param_space)

        assert len(variations) == 4
        for v in variations:
            assert "name" in v
            assert "params" in v
            assert "table" in v["params"]  # Base param preserved
            assert "col" in v["params"]
            assert "threshold" in v["params"]

    def test_expand_template_variations_name_pattern(self):
        """Test variation naming with custom pattern."""
        base_variation = {"name": "simple_filter", "params": {}}
        param_space = {"op": [">", "<"]}

        variations = expand_template_variations(
            base_variation, param_space, name_pattern="{base_name}_v{idx}"
        )

        assert variations[0]["name"] == "simple_filter_v0"
        assert variations[1]["name"] == "simple_filter_v1"

    def test_expand_template_variations_no_base_name(self):
        """Test variation expansion without base name."""
        base_variation = {"params": {}}
        param_space = {"val": [1, 2]}

        variations = expand_template_variations(base_variation, param_space)

        assert all("variation" in v["name"] for v in variations)

    def test_apply_substitutions_simple(self):
        """Test applying simple substitutions."""
        template = "SELECT {col} FROM {table}"
        subs = {"{col}": "age", "{table}": "users"}

        result = apply_substitutions(template, subs)

        assert result == "SELECT age FROM users"

    def test_apply_substitutions_multiple_occurrences(self):
        """Test substituting repeated placeholders."""
        template = "{x} + {x} = {y}"
        subs = {"{x}": "5", "{y}": "10"}

        result = apply_substitutions(template, subs)

        assert result == "5 + 5 = 10"

    def test_apply_substitutions_no_match(self):
        """Test substitutions with no matches."""
        template = "SELECT * FROM table"
        subs = {"{col}": "age"}

        result = apply_substitutions(template, subs)

        assert result == template  # Unchanged

    def test_create_column_variations(self):
        """Test creating column name variations."""
        base_params = {"threshold": 10}
        column_mapping = {"amount": ["revenue", "cost"], "user": ["customer_id", "account_id"]}

        variations = create_column_variations(base_params, column_mapping)

        assert len(variations) == 4  # 2 + 2
        assert all("threshold" in v for v in variations)


class TestParameterSpaceConfig:
    """Test suite for ParameterSpaceConfig."""

    def test_get_filter_space(self):
        """Test filter parameter space."""
        space = ParameterSpaceConfig.get_filter_space()

        assert "numeric_op" in space
        assert "threshold" in space
        assert len(space["numeric_op"]) > 0
        assert len(space["threshold"]) > 0

    def test_get_aggregation_space(self):
        """Test aggregation parameter space."""
        space = ParameterSpaceConfig.get_aggregation_space()

        assert "agg_func" in space
        assert "SUM" in space["agg_func"]
        assert "AVG" in space["agg_func"]

    def test_get_temporal_space(self):
        """Test temporal parameter space."""
        space = ParameterSpaceConfig.get_temporal_space()

        assert "date_part" in space
        assert "YEAR" in space["date_part"]
        assert "MONTH" in space["date_part"]

    def test_constants_defined(self):
        """Test that all constant lists are defined."""
        assert len(ParameterSpaceConfig.NUMERIC_OPS) > 0
        assert len(ParameterSpaceConfig.NUMERIC_THRESHOLDS) > 0
        assert len(ParameterSpaceConfig.AMOUNT_COLUMNS) > 0
        assert len(ParameterSpaceConfig.USER_COLUMNS) > 0
        assert len(ParameterSpaceConfig.EVENT_TABLES) > 0


class TestTemplate:
    """Test suite for Template class."""

    def test_template_initialization_simple(self):
        """Test template initialization with minimal data."""
        data = {
            "name": "simple_filter",
            "sql_template": "SELECT * FROM {table}",
            "ibis_template": "{table}",
            "variations": [{"name": "v1", "params": {"table": "events"}}],
        }

        template = Template(data)

        assert template.name == "simple_filter"
        assert template.sql_template == "SELECT * FROM {table}"
        assert template.ibis_template == "{table}"
        assert len(template.variations) == 1

    def test_template_initialization_with_defaults(self):
        """Test template initialization uses defaults."""
        data = {
            "name": "test",
            "sql_template": "SELECT 1",
            "ibis_template": "expr",
        }

        template = Template(data)

        assert template.description == ""
        assert template.difficulty == "medium"
        assert template.features == []
        assert template.variations == []
        assert template.context == {}

    def test_template_initialization_with_all_fields(self):
        """Test template initialization with all fields."""
        data = {
            "name": "complex",
            "description": "A complex template",
            "difficulty": "hard",
            "features": ["filter", "group_by"],
            "sql_template": "SELECT * FROM {table}",
            "ibis_template": "{table}",
            "variations": [{"name": "v1", "params": {}}],
            "context": {"tables": {"events": {}}},
        }

        template = Template(data)

        assert template.name == "complex"
        assert template.description == "A complex template"
        assert template.difficulty == "hard"
        assert template.features == ["filter", "group_by"]
        assert template.context == {"tables": {"events": {}}}

    def test_template_render_basic(self):
        """Test rendering a basic template variation."""
        data = {
            "name": "filter_template",
            "sql_template": "SELECT * FROM {table} WHERE {col} > {threshold}",
            "ibis_template": "{table}.filter({table}.{col} > {threshold})",
            "features": ["filter"],
        }

        template = Template(data)
        variation = {
            "name": "age_filter",
            "params": {"table": "users", "col": "age", "threshold": "18"},
        }

        example = template.render(variation)

        assert example["task"] == "sql_to_ibis"
        assert example["dialect"] == "duckdb"
        assert "id" in example
        assert example["input"]["sql"] == "SELECT * FROM users WHERE age > 18"
        assert example["target"]["ibis"] == "users.filter(users.age > 18)"
        assert example["meta"]["template"] == "filter_template"
        assert example["meta"]["variation"] == "age_filter"
        assert example["meta"]["features"] == ["filter"]

    def test_template_render_with_context(self):
        """Test rendering includes context."""
        data = {
            "name": "test",
            "sql_template": "SELECT 1",
            "ibis_template": "expr",
            "context": {"tables": {"events": {"schema": {"age": "int64"}}}},
        }

        template = Template(data)
        variation = {"name": "v1", "params": {}}

        example = template.render(variation)

        assert example["context"] == data["context"]

    def test_template_expansion_with_parameter_space(self):
        """Test template-level parameter space expansion."""
        data = {
            "name": "expandable",
            "sql_template": "SELECT {col}",
            "ibis_template": "{table}.{col}",
            "variations": [{"name": "base", "params": {"table": "events"}}],
            "parameter_space": {"col": ["age", "amount"]},
        }

        template = Template(data)

        # Should expand 1 variation into 2 (one per col value)
        assert len(template.variations) == 2

    def test_template_expansion_variation_level(self):
        """Test variation-level parameter space expansion."""
        data = {
            "name": "test",
            "sql_template": "SELECT {col}",
            "ibis_template": "{table}.{col}",
            "variations": [
                {
                    "name": "v1",
                    "params": {"table": "events"},
                    "parameter_space": {"col": ["a", "b"]},
                },
                {"name": "v2", "params": {"table": "users"}},
            ],
        }

        template = Template(data)

        # v1 expands to 2, v2 stays as 1 = 3 total
        assert len(template.variations) == 3

    def test_template_render_strips_whitespace(self):
        """Test render strips extra whitespace."""
        data = {
            "name": "test",
            "sql_template": "  SELECT 1  ",
            "ibis_template": "  expr  ",
        }

        template = Template(data)
        variation = {"name": "v1", "params": {}}

        example = template.render(variation)

        assert example["input"]["sql"] == "SELECT 1"
        assert example["target"]["ibis"] == "expr"


class TestTemplateLoading:
    """Test suite for template loading functions."""

    @pytest.fixture
    def temp_templates_dir(self, tmp_path):
        """Create temporary templates directory."""
        templates_dir = tmp_path / "templates"
        templates_dir.mkdir()

        # Template 1
        (templates_dir / "template1.yaml").write_text(
            """
name: simple_select
sql_template: "SELECT * FROM {table}"
ibis_template: "{table}"
features: ["select"]
variations:
  - name: events_table
    params:
      table: events
"""
        )

        # Template 2
        (templates_dir / "template2.yaml").write_text(
            """
name: filter_query
sql_template: "SELECT * FROM {table} WHERE {col} > {val}"
ibis_template: "{table}.filter({table}.{col} > {val})"
features: ["filter"]
difficulty: easy
variations:
  - name: age_filter
    params:
      table: users
      col: age
      val: 18
  - name: amount_filter
    params:
      table: transactions
      col: amount
      val: 100
"""
        )

        return templates_dir

    def test_load_templates_success(self, temp_templates_dir):
        """Test loading templates from directory."""
        templates = load_templates(temp_templates_dir)

        assert len(templates) == 2
        assert all(isinstance(t, Template) for t in templates)
        assert templates[0].name == "simple_select"
        assert templates[1].name == "filter_query"

    def test_load_templates_empty_directory(self, tmp_path):
        """Test loading from empty directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        templates = load_templates(empty_dir)

        assert templates == []

    def test_load_templates_sorted_by_filename(self, tmp_path):
        """Test templates are loaded in sorted order."""
        templates_dir = tmp_path / "templates"
        templates_dir.mkdir()

        (templates_dir / "z_last.yaml").write_text(
            """
name: last
sql_template: "SELECT 1"
ibis_template: "expr"
"""
        )

        (templates_dir / "a_first.yaml").write_text(
            """
name: first
sql_template: "SELECT 1"
ibis_template: "expr"
"""
        )

        templates = load_templates(templates_dir)

        assert templates[0].name == "first"
        assert templates[1].name == "last"

    def test_generate_examples_simple(self, temp_templates_dir):
        """Test generating examples from templates."""
        templates = load_templates(temp_templates_dir)
        examples = generate_examples(templates)

        # Template 1 has 1 variation, Template 2 has 2 variations
        assert len(examples) == 3
        assert all(isinstance(ex, dict) for ex in examples)
        assert all("id" in ex for ex in examples)
        assert all(ex["task"] == "sql_to_ibis" for ex in examples)

    def test_generate_examples_empty_templates(self):
        """Test generating examples from empty template list."""
        examples = generate_examples([])

        assert examples == []

    def test_generate_examples_no_variations(self, tmp_path):
        """Test generating examples from template with no variations."""
        templates_dir = tmp_path / "templates"
        templates_dir.mkdir()

        (templates_dir / "no_vars.yaml").write_text(
            """
name: no_variations
sql_template: "SELECT 1"
ibis_template: "expr"
"""
        )

        templates = load_templates(templates_dir)
        examples = generate_examples(templates)

        # No variations = no examples
        assert examples == []

    def test_generate_examples_preserves_metadata(self, temp_templates_dir):
        """Test examples preserve template metadata."""
        templates = load_templates(temp_templates_dir)
        examples = generate_examples(templates)

        for ex in examples:
            assert "meta" in ex
            assert "template" in ex["meta"]
            assert "variation" in ex["meta"]
            assert "features" in ex["meta"]
            assert "source" in ex["meta"]
            assert ex["meta"]["source"] == "synthetic"


class TestIntegration:
    """Integration tests for template loading and rendering."""

    def test_end_to_end_template_processing(self, tmp_path):
        """Test complete template processing workflow."""
        # Create comprehensive template
        templates_dir = tmp_path / "templates"
        templates_dir.mkdir()

        (templates_dir / "comprehensive.yaml").write_text(
            """
name: group_by_aggregate
description: "GROUP BY with aggregation"
difficulty: medium
features: ["group_by", "aggregate"]
sql_template: "SELECT {group_col}, {agg_func}({agg_col}) FROM {table} GROUP BY {group_col}"
ibis_template: "{table}.group_by({table}.{group_col}).aggregate({agg_col}_{agg_func}={table}.{agg_col}.{ibis_agg}())"
context:
  tables:
    events:
      schema:
        user_id: int64
        amount: float64
variations:
  - name: user_sum
    params:
      table: events
      group_col: user_id
      agg_col: amount
      agg_func: SUM
      ibis_agg: sum
"""
        )

        # Load and generate
        templates = load_templates(templates_dir)
        examples = generate_examples(templates)

        assert len(examples) == 1
        ex = examples[0]

        # Verify structure
        assert ex["task"] == "sql_to_ibis"
        assert "id" in ex
        assert "input" in ex
        assert "target" in ex
        assert "meta" in ex
        assert "context" in ex

        # Verify content
        assert "GROUP BY" in ex["input"]["sql"]
        assert "group_by" in ex["target"]["ibis"]
        assert ex["meta"]["difficulty"] == "medium"
        assert "group_by" in ex["meta"]["features"]
        assert "aggregate" in ex["meta"]["features"]

    def test_parameter_space_expansion_integration(self, tmp_path):
        """Test parameter space expansion with rendering."""
        templates_dir = tmp_path / "templates"
        templates_dir.mkdir()

        (templates_dir / "expandable.yaml").write_text(
            """
name: filter_variations
sql_template: "SELECT * FROM events WHERE {col} {op} {val}"
ibis_template: "events.filter(events.{col} {op} {val})"
parameter_space:
  col: ["age", "amount"]
  op: [">", "<"]
  val: [10, 20]
variations:
  - name: base
    params: {}
"""
        )

        templates = load_templates(templates_dir)
        examples = generate_examples(templates)

        # Should expand: 2 cols x 2 ops x 2 vals = 8 examples
        assert len(examples) == 8

        # Verify diversity
        sql_queries = [ex["input"]["sql"] for ex in examples]
        assert len(set(sql_queries)) == 8  # All unique

    def test_multiple_templates_with_expansion(self, tmp_path):
        """Test loading multiple templates with mixed expansion."""
        templates_dir = tmp_path / "templates"
        templates_dir.mkdir()

        # Template with expansion
        (templates_dir / "t1.yaml").write_text(
            """
name: t1
sql_template: "SELECT {col}"
ibis_template: "table.{col}"
variations:
  - name: v1
    params: {}
    parameter_space:
      col: [a, b]
"""
        )

        # Template without expansion
        (templates_dir / "t2.yaml").write_text(
            """
name: t2
sql_template: "SELECT 1"
ibis_template: "expr"
variations:
  - name: static
    params: {}
"""
        )

        templates = load_templates(templates_dir)
        examples = generate_examples(templates)

        # t1 expands to 2, t2 has 1 = 3 total
        assert len(examples) == 3
