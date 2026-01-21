"""
Helper Utilities
Shared utility functions used across the pipeline
"""
from datetime import datetime
from .constants import DATE_FORMAT, DATETIME_FORMAT, DATETIME_FORMAT_UTC


def parse_date(date_string):
    """
    Parse date string to datetime.date object

    Args:
        date_string (str): Date string in YYYY-MM-DD format

    Returns:
        datetime.date or None: Parsed date object
    """
    if not date_string:
        return None
    try:
        return datetime.strptime(date_string, DATE_FORMAT).date()
    except:
        return None


def parse_datetime(datetime_string):
    """
    Parse datetime string to datetime.datetime object

    Args:
        datetime_string (str): Datetime string in ISO format

    Returns:
        datetime.datetime or None: Parsed datetime object
    """
    if not datetime_string:
        return None
    try:
        # Handle ISO format with Z (UTC)
        if datetime_string.endswith('Z'):
            return datetime.strptime(datetime_string, DATETIME_FORMAT_UTC)
        else:
            return datetime.strptime(datetime_string, DATETIME_FORMAT)
    except:
        return None


def to_string_id(value):
    """
    Convert value to string ID (handles None)

    Args:
        value: Value to convert

    Returns:
        str or None: String representation or None
    """
    if value is None:
        return None
    return str(value)


def get_table_config(table_name):
    """
    Get configuration for a table

    Args:
        table_name (str): Table name

    Returns:
        dict: Table configuration

    Raises:
        ValueError: If table not found in configuration
    """
    from .constants import TABLE_CONFIGS

    if table_name not in TABLE_CONFIGS:
        raise ValueError(f"Table '{table_name}' not found in TABLE_CONFIGS")

    return TABLE_CONFIGS[table_name]


def get_primary_keys(table_name):
    """
    Get primary keys for a table

    Args:
        table_name (str): Table name

    Returns:
        list: List of primary key column names
    """
    config = get_table_config(table_name)
    return config['primary_keys']


def get_data_columns(df, table_name):
    """
    Get data columns for a table (excludes PKs and metadata)

    Args:
        df (pd.DataFrame): DataFrame
        table_name (str): Table name

    Returns:
        list: List of data column names
    """
    pks = get_primary_keys(table_name)
    excluded = pks + ['insert_date', 'update_date']
    return [c for c in df.columns if c not in excluded]