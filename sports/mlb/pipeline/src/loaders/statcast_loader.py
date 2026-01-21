"""
Statcast Data Loader
Load Statcast pitch data into database
"""
import sys
import os

# Add project root to path for generic imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from generic.db import get_engine, upsert_via_staging
from ..utils.constants import DEFAULT_SCHEMA


def load_statcast_pitches(df, engine=None, schema=DEFAULT_SCHEMA, dry_run=False, batch_size=10000):
    """
    Load Statcast pitch-level data to StatcastPitches table

    Args:
        df (pd.DataFrame): Cleaned Statcast data
        engine: SQLAlchemy engine (if None, creates new one)
        schema (str): Database schema
        dry_run (bool): If True, don't actually load data
        batch_size (int): Number of rows per batch (Statcast data can be huge)

    Returns:
        dict: Result from upsert operation

    Note:
        StatcastPitches uses auto-increment pitch_id as primary key
        We don't upsert based on natural keys since pitch data shouldn't change
        Instead, we can use game_pk + at_bat_number + pitch_number to avoid duplicates
    """
    if engine is None:
        engine = get_engine('MLB')

    if len(df) == 0:
        print("  ⚠ No data to load")
        return {'inserted': 0, 'updated': 0}

    # Note: pitch_id is auto-increment, so we don't include it in the data
    # We'll use a composite natural key to avoid duplicates
    primary_keys = ['game_pk', 'at_bat_number', 'pitch_number']

    # Get all columns except pitch_id (auto-generated) and primary keys (to avoid duplication)
    data_columns = [col for col in df.columns
                    if col != 'pitch_id' and col not in primary_keys]

    print(f"  Loading {len(df):,} pitches to StatcastPitches...")

    # For very large datasets, process in batches
    if len(df) > batch_size:
        print(f"  Processing in batches of {batch_size:,} rows...")

        total_inserted = 0
        total_updated = 0

        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(df) + batch_size - 1) // batch_size

            print(f"    Batch {batch_num}/{total_batches}: rows {i:,} to {min(i+batch_size, len(df)):,}")

            result = upsert_via_staging(
                df=batch,
                table_name='StatcastPitches',
                primary_keys=primary_keys,
                data_columns=data_columns,
                engine=engine,
                schema=schema,
                dry_run=dry_run
            )

            total_inserted += result.get('inserted', 0)
            total_updated += result.get('updated', 0)

        print(f"  ✓ StatcastPitches loaded: {total_inserted:,} inserted, {total_updated:,} updated")
        return {'inserted': total_inserted, 'updated': total_updated}

    else:
        # Small dataset, load all at once
        result = upsert_via_staging(
            df=df,
            table_name='StatcastPitches',
            primary_keys=primary_keys,
            data_columns=data_columns,
            engine=engine,
            schema=schema,
            dry_run=dry_run
        )

        print(f"  ✓ StatcastPitches loaded: {result.get('inserted', 0):,} inserted, {result.get('updated', 0):,} updated")
        return result


def check_existing_data(start_date, end_date, engine=None):
    """
    Check if data already exists for a date range

    Args:
        start_date (str): Start date 'YYYY-MM-DD'
        end_date (str): End date 'YYYY-MM-DD'
        engine: SQLAlchemy engine

    Returns:
        dict: Summary of existing data
    """
    if engine is None:
        engine = get_engine('MLB')

    query = f"""
        SELECT
            MIN(game_date) as min_date,
            MAX(game_date) as max_date,
            COUNT(*) as pitch_count,
            COUNT(DISTINCT game_pk) as game_count
        FROM StatcastPitches
        WHERE game_date BETWEEN '{start_date}' AND '{end_date}'
    """

    try:
        import pandas as pd
        result = pd.read_sql(query, engine)

        if len(result) > 0 and result['pitch_count'].iloc[0] > 0:
            return {
                'exists': True,
                'min_date': result['min_date'].iloc[0],
                'max_date': result['max_date'].iloc[0],
                'pitch_count': result['pitch_count'].iloc[0],
                'game_count': result['game_count'].iloc[0]
            }
        else:
            return {'exists': False}

    except Exception as e:
        print(f"  ⚠ Could not check existing data: {e}")
        return {'exists': False}
