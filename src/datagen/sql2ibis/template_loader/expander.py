"""Parameter space expansion for generating variations."""

import itertools
from typing import Any, Dict, List, Iterator


def expand_parameter_space(param_space: Dict[str, List[Any]]) -> Iterator[Dict[str, Any]]:
    """Generate all combinations from a parameter space.

    Parameters
    ----------
    param_space : dict
        Dictionary mapping parameter names to lists of possible values

    Returns
    -------
    Iterator[dict]
        Iterator of parameter combinations

    Examples
    --------
    >>> space = {"col": ["amount", "price"], "op": [">", "<"], "val": [10, 20]}
    >>> list(expand_parameter_space(space))
    [
        {"col": "amount", "op": ">", "val": 10},
        {"col": "amount", "op": ">", "val": 20},
        {"col": "amount", "op": "<", "val": 10},
        ...
    ]
    """
    if not param_space:
        yield {}
        return

    keys = list(param_space.keys())
    values = [param_space[k] for k in keys]

    for combination in itertools.product(*values):
        yield dict(zip(keys, combination))


def expand_template_variations(
    base_variation: Dict[str, Any],
    param_space: Dict[str, List[Any]],
    name_pattern: str = "{base_name}_{idx}"
) -> List[Dict[str, Any]]:
    """Expand a single variation with parameter space.

    Parameters
    ----------
    base_variation : dict
        Base variation with template parameters
    param_space : dict
        Parameter space for expansion
    name_pattern : str
        Pattern for naming expanded variations (supports {base_name} and {idx})

    Returns
    -------
    list of dict
        Expanded variations
    """
    base_name = base_variation.get("name", "variation")
    base_params = base_variation.get("params", {})

    variations = []

    for idx, param_combo in enumerate(expand_parameter_space(param_space)):
        # Merge base params with expanded params
        merged_params = {**base_params, **param_combo}

        variation = {
            "name": name_pattern.format(base_name=base_name, idx=idx),
            "params": merged_params,
        }
        variations.append(variation)

    return variations


def apply_substitutions(
    template_str: str,
    substitutions: Dict[str, str]
) -> str:
    """Apply string substitutions to a template.

    Parameters
    ----------
    template_str : str
        Template string
    substitutions : dict
        Mapping of old -> new strings

    Returns
    -------
    str
        Template with substitutions applied
    """
    result = template_str
    for old, new in substitutions.items():
        result = result.replace(old, new)
    return result


class ParameterSpaceConfig:
    """Configuration for common parameter spaces."""

    # Numeric comparison operators
    NUMERIC_OPS = [">", "<", ">=", "<=", "==", "!="]

    # Numeric threshold values
    NUMERIC_THRESHOLDS = [5, 10, 15, 20, 25, 30]

    # Column name variations
    AMOUNT_COLUMNS = ["amount", "value", "price", "revenue", "cost"]
    USER_COLUMNS = ["user_id", "customer_id", "account_id"]
    TIMESTAMP_COLUMNS = ["event_ts", "created_at", "updated_at", "timestamp"]

    # Table name variations
    EVENT_TABLES = ["events", "transactions", "logs", "records", "activities"]
    USER_TABLES = ["users", "customers", "accounts"]

    # Aggregation functions
    AGG_FUNCTIONS = ["SUM", "AVG", "MIN", "MAX", "COUNT"]

    # Date parts
    DATE_PARTS = ["YEAR", "MONTH", "DAY", "QUARTER", "WEEK"]

    @classmethod
    def get_filter_space(cls, table: str = "events") -> Dict[str, List[Any]]:
        """Get parameter space for filter operations."""
        return {
            "numeric_op": cls.NUMERIC_OPS,
            "threshold": cls.NUMERIC_THRESHOLDS,
        }

    @classmethod
    def get_aggregation_space(cls) -> Dict[str, List[Any]]:
        """Get parameter space for aggregation operations."""
        return {
            "agg_func": cls.AGG_FUNCTIONS,
        }

    @classmethod
    def get_temporal_space(cls) -> Dict[str, List[Any]]:
        """Get parameter space for temporal operations."""
        return {
            "date_part": cls.DATE_PARTS,
        }


def create_column_variations(
    base_params: Dict[str, Any],
    column_mapping: Dict[str, List[str]]
) -> List[Dict[str, Any]]:
    """Create variations by substituting column names.

    Parameters
    ----------
    base_params : dict
        Base parameter set
    column_mapping : dict
        Mapping of column categories to alternative names

    Returns
    -------
    list of dict
        Variations with different column names
    """
    variations = []

    # Get all column substitutions
    for category, alternatives in column_mapping.items():
        for alt_col in alternatives:
            params = base_params.copy()
            # This is a simple example - you'd need more sophisticated logic
            # to handle column references in SQL/Ibis templates
            params[f"{category}_col"] = alt_col
            variations.append(params)

    return variations
