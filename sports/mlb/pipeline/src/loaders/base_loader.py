"""
Base Loader
Generic database loading functions
"""
import sys
import os

# Add project root to path for generic imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from generic.db import get_engine, upsert_via_staging
from ..utils.helpers import get_primary_keys, get_data_columns
from ..utils.constants import DEFAULT_SCHEMA


def load_table_to_database(df, table_name, engine=None, schema=DEFAULT_SCHEMA, dry_run=False):
    """
    Generic function to load any table to database

    Args:
        df (pd.DataFrame): Data to load
        table_name (str): Target table name
        engine: SQLAlchemy engine (if None, creates new one)
        schema (str): Database schema
        dry_run (bool): If True, don't actually load data

    Returns:
        dict: Result from upsert operation
    """
    if engine is None:
        engine = get_engine('MLB')

    # Get configuration from constants
    pks = get_primary_keys(table_name)
    data_columns = get_data_columns(df, table_name)

    print(f"  Loading {len(df)} rows to {table_name}...")

    result = upsert_via_staging(
        df=df,
        table_name=table_name,
        primary_keys=pks,
        data_columns=data_columns,
        engine=engine,
        schema=schema,
        dry_run=dry_run
    )

    print(f"  ✓ {table_name} loaded successfully")
    return result
