"""Tests for synthetic data augmentation."""

import pytest
import copy

from src.datagen.augmentation.augmenter import (
    SyntheticAugmenter,
    augment_dataset
)


class TestSyntheticAugmenter:
    """Test suite for SyntheticAugmenter class."""

    @pytest.fixture
    def augmenter(self):
        """Create augmenter instance."""
        return SyntheticAugmenter()

    @pytest.fixture
    def base_example(self):
        """Create base example for testing."""
        return {
            "input": {"sql": "SELECT user_id, amount FROM events WHERE amount > 10"},
            "target": {"ibis": "events.filter(events.amount > 10)[['user_id', 'amount']]"},
            "meta": {"source": "test"},
            "context": {"tables": {"events": {"schema": {"user_id": "int64", "amount": "float64"}}}},
        }

    def test_augmenter_initialization(self, augmenter):
        """Test augmenter initializes with substitution dictionaries."""
        assert hasattr(augmenter, 'COLUMN_SUBSTITUTIONS')
        assert hasattr(augmenter, 'TABLE_SUBSTITUTIONS')
        assert isinstance(augmenter.COLUMN_SUBSTITUTIONS, dict)
        assert isinstance(augmenter.TABLE_SUBSTITUTIONS, dict)

    def test_augment_by_column_substitution_basic(self, augmenter, base_example):
        """Test column substitution creates variations."""
        variations = augmenter.augment_by_column_substitution(base_example, max_variations=2)

        assert len(variations) > 0
        # Should have variations for both user_id and amount
        assert all("input" in v for v in variations)
        assert all("target" in v for v in variations)
        assert all("meta" in v for v in variations)

    def test_augment_by_column_substitution_user_id(self, augmenter, base_example):
        """Test user_id column substitution."""
        variations = augmenter.augment_by_column_substitution(base_example)

        # Check that user_id was replaced
        user_id_vars = [v for v in variations if "user_id" not in v["input"]["sql"].lower()]
        assert len(user_id_vars) > 0

        # Verify expected replacements
        for var in user_id_vars:
            sql = var["input"]["sql"].lower()
            ibis = var["target"]["ibis"].lower()
            # Should contain one of the alternatives
            assert any(alt in sql for alt in ["customer_id", "account_id", "client_id", "uid"])
            assert any(alt in ibis for alt in ["customer_id", "account_id", "client_id", "uid"])

    def test_augment_by_column_substitution_amount(self, augmenter, base_example):
        """Test amount column substitution."""
        variations = augmenter.augment_by_column_substitution(base_example)

        # Check that amount was replaced
        amount_vars = [v for v in variations if "amount" not in v["input"]["sql"].lower()]
        assert len(amount_vars) > 0

        # Verify expected replacements
        for var in amount_vars:
            sql = var["input"]["sql"].lower()
            # Should contain one of the alternatives
            assert any(alt in sql for alt in ["value", "price", "revenue", "cost", "total"])

    def test_augment_by_column_substitution_max_variations(self, augmenter, base_example):
        """Test max_variations parameter limits output."""
        variations_1 = augmenter.augment_by_column_substitution(base_example, max_variations=1)
        variations_3 = augmenter.augment_by_column_substitution(base_example, max_variations=3)

        # More variations with higher limit (though depends on how many columns match)
        assert len(variations_3) >= len(variations_1)

    def test_augment_by_column_substitution_preserves_metadata(self, augmenter, base_example):
        """Test variations preserve original metadata."""
        variations = augmenter.augment_by_column_substitution(base_example)

        for var in variations:
            assert var["meta"]["source"] == "test"
            assert "augmentation" in var["meta"]
            assert var["meta"]["augmentation"].startswith("column_sub_")

    def test_augment_by_column_substitution_no_match(self, augmenter):
        """Test examples with no matching columns return empty list."""
        example = {
            "input": {"sql": "SELECT id FROM table1"},
            "target": {"ibis": "table1"},
            "meta": {},
        }

        variations = augmenter.augment_by_column_substitution(example)

        # No recognized columns, so no variations
        assert len(variations) == 0

    def test_augment_by_table_substitution_basic(self, augmenter, base_example):
        """Test table substitution creates variations."""
        variations = augmenter.augment_by_table_substitution(base_example)

        assert len(variations) > 0
        # All variations should have 'events' replaced
        for var in variations:
            assert "events" not in var["input"]["sql"].lower()

    def test_augment_by_table_substitution_events(self, augmenter, base_example):
        """Test events table substitution."""
        variations = augmenter.augment_by_table_substitution(base_example)

        # Check expected replacements
        for var in variations:
            sql = var["input"]["sql"].lower()
            ibis = var["target"]["ibis"].lower()
            assert any(alt in sql for alt in ["transactions", "logs", "records", "activities"])
            assert any(alt in ibis for alt in ["transactions", "logs", "records", "activities"])

    def test_augment_by_table_substitution_updates_context(self, augmenter, base_example):
        """Test table substitution updates context."""
        variations = augmenter.augment_by_table_substitution(base_example)

        for var in variations:
            # Context should have new table name, not old
            assert "events" not in var["context"]["tables"]
            # Should have one of the alternatives
            table_names = list(var["context"]["tables"].keys())
            assert len(table_names) == 1
            assert table_names[0] in ["transactions", "logs", "records", "activities"]

    def test_augment_by_table_substitution_preserves_schema(self, augmenter, base_example):
        """Test table substitution preserves schema."""
        variations = augmenter.augment_by_table_substitution(base_example)

        original_schema = base_example["context"]["tables"]["events"]["schema"]

        for var in variations:
            new_table_name = list(var["context"]["tables"].keys())[0]
            new_schema = var["context"]["tables"][new_table_name]["schema"]
            assert new_schema == original_schema

    def test_augment_by_table_substitution_metadata(self, augmenter, base_example):
        """Test table substitution adds metadata."""
        variations = augmenter.augment_by_table_substitution(base_example)

        for var in variations:
            assert "augmentation" in var["meta"]
            assert var["meta"]["augmentation"].startswith("table_sub_events_")

    def test_augment_by_table_substitution_no_match(self, augmenter):
        """Test example with no matching tables returns empty list."""
        example = {
            "input": {"sql": "SELECT * FROM unknown_table"},
            "target": {"ibis": "unknown_table"},
            "meta": {},
        }

        variations = augmenter.augment_by_table_substitution(example)

        assert len(variations) == 0

    def test_augment_by_value_permutation_basic(self, augmenter, base_example):
        """Test value permutation creates variations."""
        variations = augmenter.augment_by_value_permutation(base_example)

        assert len(variations) > 0
        # Should replace the value 10
        for var in variations:
            assert "10" not in var["input"]["sql"] or var["input"]["sql"] != base_example["input"]["sql"]

    def test_augment_by_value_permutation_default_ranges(self, augmenter, base_example):
        """Test value permutation with default value ranges."""
        variations = augmenter.augment_by_value_permutation(base_example)

        # Should have multiple variations with different values
        values_found = set()
        for var in variations:
            # Extract numeric value from SQL
            import re
            match = re.search(r'> (\d+)', var["input"]["sql"])
            if match:
                values_found.add(match.group(1))

        # Should have replaced 10 with various values
        assert len(values_found) > 0
        assert "10" not in values_found  # Original value should be replaced

    def test_augment_by_value_permutation_custom_ranges(self, augmenter, base_example):
        """Test value permutation with custom value ranges."""
        custom_ranges = {
            "numeric": [100, 200, 300],
            "year": [2020, 2021],
        }

        variations = augmenter.augment_by_value_permutation(base_example, value_ranges=custom_ranges)

        # Should use custom values
        assert len(variations) > 0
        for var in variations:
            sql = var["input"]["sql"]
            # Should contain one of the custom values
            assert any(str(val) in sql for val in [100, 200, 300])

    def test_augment_by_value_permutation_metadata(self, augmenter, base_example):
        """Test value permutation adds metadata."""
        variations = augmenter.augment_by_value_permutation(base_example)

        for var in variations:
            assert "augmentation" in var["meta"]
            assert var["meta"]["augmentation"].startswith("value_perm_")

    def test_augment_by_value_permutation_no_numeric(self, augmenter):
        """Test example with no numeric literals returns empty list."""
        example = {
            "input": {"sql": "SELECT * FROM events"},
            "target": {"ibis": "events"},
            "meta": {},
        }

        variations = augmenter.augment_by_value_permutation(example)

        # No numeric literals, so no variations
        assert len(variations) == 0

    def test_replace_identifier_basic(self, augmenter):
        """Test identifier replacement with word boundaries."""
        code = "SELECT user_id FROM users WHERE user_id > 10"

        result = augmenter._replace_identifier(code, "user_id", "customer_id")

        assert "customer_id" in result
        assert "user_id" not in result
        # Should replace all occurrences
        assert result.count("customer_id") == 2

    def test_replace_identifier_case_insensitive(self, augmenter):
        """Test identifier replacement is case insensitive."""
        code = "SELECT User_ID, user_id, USER_ID FROM table"

        result = augmenter._replace_identifier(code, "user_id", "customer_id")

        assert "customer_id" in result
        # Should replace all case variations
        assert "User_ID" not in result
        assert "user_id" not in result
        assert "USER_ID" not in result

    def test_replace_identifier_word_boundaries(self, augmenter):
        """Test identifier replacement respects word boundaries."""
        code = "SELECT user_id, user_id_prefix, prefix_user_id FROM table"

        result = augmenter._replace_identifier(code, "user_id", "customer_id")

        # Should only replace exact matches
        assert "customer_id," in result
        assert "user_id_prefix" in result  # Partial match not replaced
        assert "prefix_user_id" in result  # Partial match not replaced

    def test_replace_identifier_with_special_chars(self, augmenter):
        """Test identifier replacement escapes special regex chars."""
        code = "SELECT col FROM table WHERE col > 10"

        # Even with dots (special in regex), should work
        result = augmenter._replace_identifier(code, "col", "new.col")

        assert "new.col" in result

    def test_variations_are_independent(self, augmenter, base_example):
        """Test variations are deep copies and independent."""
        variations = augmenter.augment_by_column_substitution(base_example, max_variations=2)

        # Modify one variation
        if len(variations) > 0:
            variations[0]["meta"]["test"] = "modified"

            # Other variations should not be affected
            if len(variations) > 1:
                assert "test" not in variations[1]["meta"]

        # Original should not be affected
        assert "test" not in base_example["meta"]


class TestAugmentDataset:
    """Test suite for augment_dataset function."""

    @pytest.fixture
    def example_dataset(self):
        """Create example dataset."""
        return [
            {
                "input": {"sql": "SELECT user_id FROM events WHERE amount > 10"},
                "target": {"ibis": "events.filter(events.amount > 10).user_id"},
                "meta": {"source": "test"},
                "context": {"tables": {"events": {}}},
            },
            {
                "input": {"sql": "SELECT name FROM users"},
                "target": {"ibis": "users.name"},
                "meta": {"source": "test"},
                "context": {"tables": {"users": {}}},
            },
        ]

    def test_augment_dataset_basic(self, example_dataset):
        """Test dataset augmentation creates variations."""
        augmented = augment_dataset(example_dataset, max_variations_per_example=2)

        # Should include originals plus variations
        assert len(augmented) > len(example_dataset)

    def test_augment_dataset_includes_originals(self, example_dataset):
        """Test augmented dataset includes all originals."""
        augmented = augment_dataset(example_dataset)

        # First examples should be the originals
        assert example_dataset[0] in augmented
        assert example_dataset[1] in augmented

    def test_augment_dataset_applies_all_augmentations(self, example_dataset):
        """Test all augmentation types are applied."""
        augmented = augment_dataset(example_dataset, max_variations_per_example=5)

        # Find augmented examples (not originals)
        augmented_only = [ex for ex in augmented if "augmentation" in ex.get("meta", {})]

        # Should have variations from different augmentation types
        aug_types = set(ex["meta"]["augmentation"].split("_")[0] for ex in augmented_only)
        # Expect column, table, and value augmentations
        assert "column" in aug_types or "table" in aug_types or "value" in aug_types

    def test_augment_dataset_respects_max_variations(self, example_dataset):
        """Test max_variations_per_example limits output."""
        augmented_low = augment_dataset(example_dataset, max_variations_per_example=1)
        augmented_high = augment_dataset(example_dataset, max_variations_per_example=10)

        # Higher limit should produce more examples
        assert len(augmented_high) >= len(augmented_low)

    def test_augment_dataset_empty_input(self):
        """Test augmenting empty dataset."""
        augmented = augment_dataset([])

        assert augmented == []

    def test_augment_dataset_single_example(self):
        """Test augmenting single example."""
        single = [{
            "input": {"sql": "SELECT user_id, amount FROM events"},
            "target": {"ibis": "events[['user_id', 'amount']]"},
            "meta": {},
            "context": {"tables": {"events": {}}},
        }]

        augmented = augment_dataset(single, max_variations_per_example=3)

        # Should have original + variations
        assert len(augmented) > 1
        assert single[0] in augmented

    def test_augment_dataset_preserves_structure(self, example_dataset):
        """Test augmented examples have same structure as originals."""
        augmented = augment_dataset(example_dataset)

        for ex in augmented:
            assert "input" in ex
            assert "target" in ex
            assert "meta" in ex
            assert "sql" in ex["input"]
            assert "ibis" in ex["target"]


class TestIntegration:
    """Integration tests for augmentation pipeline."""

    def test_full_augmentation_pipeline(self):
        """Test complete augmentation workflow."""
        base = {
            "input": {"sql": "SELECT user_id, amount FROM events WHERE amount > 10"},
            "target": {"ibis": "events.filter(events.amount > 10)[['user_id', 'amount']]"},
            "meta": {"source": "synthetic"},
            "context": {"tables": {"events": {"schema": {}}}},
        }

        augmenter = SyntheticAugmenter()

        # Apply all augmentation types
        col_vars = augmenter.augment_by_column_substitution(base, max_variations=2)
        table_vars = augmenter.augment_by_table_substitution(base)
        value_vars = augmenter.augment_by_value_permutation(base)

        total_vars = len(col_vars) + len(table_vars) + len(value_vars)

        # Should have generated multiple variations
        assert total_vars > 5

        # All should have proper structure
        all_vars = col_vars + table_vars + value_vars
        for var in all_vars:
            assert "augmentation" in var["meta"]
            assert var["input"]["sql"] != base["input"]["sql"]  # Should be different

    def test_augmentation_maintains_validity(self):
        """Test augmented examples maintain structural validity."""
        example = {
            "input": {"sql": "SELECT amount FROM events WHERE amount > 100"},
            "target": {"ibis": "events.filter(events.amount > 100).amount"},
            "meta": {"difficulty": "easy"},
            "context": {"tables": {"events": {}}},
        }

        augmented = augment_dataset([example], max_variations_per_example=10)

        # All examples should be structurally valid
        for ex in augmented:
            assert isinstance(ex["input"]["sql"], str)
            assert isinstance(ex["target"]["ibis"], str)
            assert len(ex["input"]["sql"]) > 0
            assert len(ex["target"]["ibis"]) > 0
            assert "meta" in ex

    def test_augmentation_diversity(self):
        """Test augmentation creates diverse variations."""
        example = {
            "input": {"sql": "SELECT user_id, amount FROM events WHERE amount > 10"},
            "target": {"ibis": "events.filter(events.amount > 10)[['user_id', 'amount']]"},
            "meta": {},
            "context": {"tables": {"events": {}}},
        }

        augmented = augment_dataset([example], max_variations_per_example=20)

        # Collect all SQL queries
        sql_queries = [ex["input"]["sql"] for ex in augmented]

        # Should have multiple unique queries
        unique_queries = set(sql_queries)
        assert len(unique_queries) > 5  # Should have good diversity
