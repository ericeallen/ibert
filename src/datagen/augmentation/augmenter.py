"""Synthetic data augmentation for SQL→Ibis examples."""

import re
from typing import List, Dict, Any
import copy


class SyntheticAugmenter:
    """Augment examples through systematic variations."""

    # Column name substitutions by category
    COLUMN_SUBSTITUTIONS = {
        "amount": ["value", "price", "revenue", "cost", "total"],
        "user_id": ["customer_id", "account_id", "client_id", "uid"],
        "event_ts": ["timestamp", "created_at", "updated_at", "event_time"],
        "name": ["username", "full_name", "display_name"],
    }

    # Table name substitutions
    TABLE_SUBSTITUTIONS = {
        "events": ["transactions", "logs", "records", "activities"],
        "users": ["customers", "accounts", "clients"],
        "labels": ["tags", "categories", "classifications"],
    }

    def augment_by_column_substitution(
        self,
        example: Dict[str, Any],
        max_variations: int = 3
    ) -> List[Dict[str, Any]]:
        """Create variations by substituting column names.

        Parameters
        ----------
        example : dict
            Base example with SQL and Ibis code
        max_variations : int
            Maximum variations per column

        Returns
        -------
        list of dict
            Augmented examples
        """
        variations = []

        sql = example["input"]["sql"]
        ibis_code = example["target"]["ibis"]

        # Find columns in the SQL
        for original_col, alternatives in self.COLUMN_SUBSTITUTIONS.items():
            if original_col in sql.lower():
                for i, alt_col in enumerate(alternatives[:max_variations]):
                    # Create variation
                    new_sql = self._replace_identifier(sql, original_col, alt_col)
                    new_ibis = self._replace_identifier(ibis_code, original_col, alt_col)

                    if new_sql != sql:  # Only add if something changed
                        variation = copy.deepcopy(example)
                        variation["input"]["sql"] = new_sql
                        variation["target"]["ibis"] = new_ibis
                        variation["meta"]["augmentation"] = f"column_sub_{original_col}_{alt_col}"
                        variations.append(variation)

        return variations

    def augment_by_table_substitution(
        self,
        example: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Create variations by substituting table names.

        Parameters
        ----------
        example : dict
            Base example

        Returns
        -------
        list of dict
            Augmented examples
        """
        variations = []

        sql = example["input"]["sql"]
        ibis_code = example["target"]["ibis"]

        for original_table, alternatives in self.TABLE_SUBSTITUTIONS.items():
            if original_table in sql.lower():
                for alt_table in alternatives:
                    # Replace table name
                    new_sql = self._replace_identifier(sql, original_table, alt_table)
                    new_ibis = self._replace_identifier(ibis_code, original_table, alt_table)

                    # Also update context if present
                    if new_sql != sql:
                        variation = copy.deepcopy(example)
                        variation["input"]["sql"] = new_sql
                        variation["target"]["ibis"] = new_ibis

                        # Update context table names
                        if "context" in variation and "tables" in variation["context"]:
                            if original_table in variation["context"]["tables"]:
                                table_schema = variation["context"]["tables"].pop(original_table)
                                variation["context"]["tables"][alt_table] = table_schema

                        variation["meta"]["augmentation"] = f"table_sub_{original_table}_{alt_table}"
                        variations.append(variation)

        return variations

    def augment_by_value_permutation(
        self,
        example: Dict[str, Any],
        value_ranges: Dict[str, List[Any]] = None
    ) -> List[Dict[str, Any]]:
        """Create variations by changing literal values.

        Parameters
        ----------
        example : dict
            Base example
        value_ranges : dict
            Mapping of value types to ranges

        Returns
        -------
        list of dict
            Augmented examples
        """
        if value_ranges is None:
            value_ranges = {
                "numeric": [5, 8, 10, 12, 15, 18, 20, 25, 30],
                "year": [2023, 2024, 2025],
            }

        variations = []

        sql = example["input"]["sql"]
        ibis_code = example["target"]["ibis"]

        # Find numeric literals in SQL
        numeric_pattern = r'\b(\d+)\b'
        matches = list(re.finditer(numeric_pattern, sql))

        for match in matches[:1]:  # Limit to first numeric literal
            original_value = match.group(1)

            for new_value in value_ranges["numeric"]:
                if str(new_value) != original_value:
                    new_sql = sql.replace(str(original_value), str(new_value), 1)
                    new_ibis = ibis_code.replace(str(original_value), str(new_value), 1)

                    variation = copy.deepcopy(example)
                    variation["input"]["sql"] = new_sql
                    variation["target"]["ibis"] = new_ibis
                    variation["meta"]["augmentation"] = f"value_perm_{original_value}_{new_value}"
                    variations.append(variation)

        return variations

    def _replace_identifier(self, code: str, old_name: str, new_name: str) -> str:
        """Replace identifier in code, preserving word boundaries.

        Parameters
        ----------
        code : str
            Source code
        old_name : str
            Identifier to replace
        new_name : str
            Replacement identifier

        Returns
        -------
        str
            Code with replacements
        """
        # Use word boundaries to avoid partial matches
        pattern = r'\b' + re.escape(old_name) + r'\b'
        return re.sub(pattern, new_name, code, flags=re.IGNORECASE)


def augment_dataset(
    examples: List[Dict[str, Any]],
    max_variations_per_example: int = 5
) -> List[Dict[str, Any]]:
    """Augment an entire dataset.

    Parameters
    ----------
    examples : list of dict
        Base examples
    max_variations_per_example : int
        Maximum augmentations per example

    Returns
    -------
    list of dict
        Original + augmented examples
    """
    augmenter = SyntheticAugmenter()
    all_examples = list(examples)  # Start with originals

    for example in examples:
        # Apply column substitution
        col_vars = augmenter.augment_by_column_substitution(example, max_variations=2)
        all_examples.extend(col_vars[:max_variations_per_example])

        # Apply table substitution
        table_vars = augmenter.augment_by_table_substitution(example)
        all_examples.extend(table_vars[:max_variations_per_example])

        # Apply value permutation
        value_vars = augmenter.augment_by_value_permutation(example)
        all_examples.extend(value_vars[:max_variations_per_example])

    return all_examples


if __name__ == "__main__":
    # Example usage
    example = {
        "input": {"sql": "SELECT user_id, amount FROM events WHERE amount > 10"},
        "target": {"ibis": "events.filter(events.amount > 10)[['user_id', 'amount']]"},
        "meta": {"source": "test"},
        "context": {"tables": {"events": {}}},
    }

    augmenter = SyntheticAugmenter()

    col_variations = augmenter.augment_by_column_substitution(example)
    print(f"Column variations: {len(col_variations)}")

    table_variations = augmenter.augment_by_table_substitution(example)
    print(f"Table variations: {len(table_variations)}")

    value_variations = augmenter.augment_by_value_permutation(example)
    print(f"Value variations: {len(value_variations)}")

    print(f"\nTotal: {len(col_variations) + len(table_variations) + len(value_variations)} variations from 1 example")
