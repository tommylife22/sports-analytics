"""
Base Cleaners
Generic cleaning and validation functions
"""
import pandas as pd


def validate_dataframe(df, required_columns, name="DataFrame"):
    """
    Validate that a DataFrame has required columns

    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
        name (str): Name for error messages

    Raises:
        ValueError: If required columns are missing
    """
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"{name} is missing columns: {missing_cols}")


def check_duplicates(df, subset_columns, name="DataFrame"):
    """
    Check for duplicate rows based on subset of columns

    Args:
        df (pd.DataFrame): DataFrame to check
        subset_columns (list): Columns to check for duplicates
        name (str): Name for logging

    Returns:
        pd.DataFrame: DataFrame with duplicates info
    """
    duplicates = df[df.duplicated(subset=subset_columns, keep=False)]

    if len(duplicates) > 0:
        print(f"⚠ Warning: {name} has {len(duplicates)} duplicate rows")
        return duplicates
    else:
        print(f"✓ {name} has no duplicates")
        return pd.DataFrame()


def remove_nulls_in_required_columns(df, required_columns, name="DataFrame"):
    """
    Remove rows with null values in required columns

    Args:
        df (pd.DataFrame): DataFrame to clean
        required_columns (list): Columns that cannot be null
        name (str): Name for logging

    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    original_len = len(df)
    df = df.dropna(subset=required_columns)
    removed = original_len - len(df)

    if removed > 0:
        print(f"⚠ Removed {removed} rows from {name} due to null values in required columns")

    return df
