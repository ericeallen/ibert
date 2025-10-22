"""Test fixtures for validation."""

import pandas as pd


def get_test_tables() -> dict:
    """Create test tables for validation.

    Returns
    -------
    dict
        Dictionary of table_name -> DataFrame
    """
    # Events table
    events_df = pd.DataFrame(
        {
            "user_id": [1, 2, 3, 1, 2, 3, 1, 2],
            "event_ts": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-04",
                ]
            ),
            "amount": [10.0, 20.0, 30.0, 15.0, 25.0, 35.0, 5.0, 50.0],
        }
    )

    # Labels table
    labels_df = pd.DataFrame({"user_id": [1, 2, 3], "label": [0, 1, 1]})

    # Users table
    users_df = pd.DataFrame({"user_id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"]})

    return {
        "events": events_df,
        "labels": labels_df,
        "users": users_df,
    }
