"""
CBB Data Migration Script
Migrates data from nhlpipe-sqlsvr.CBB schema to cbb.dbo schema
"""

import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
from sqlalchemy import text
from generic.db import get_engine

def migrate_table(source_engine, target_engine, table_name, source_schema='CBB', target_schema='dbo',
                  primary_keys=None, skip_identity_cols=None):
    """
    Migrate a single table from source to target database

    Args:
        source_engine: SQLAlchemy engine for source database
        target_engine: SQLAlchemy engine for target database
        table_name: Name of the table to migrate
        source_schema: Schema in source database
        target_schema: Schema in target database
        primary_keys: List of primary key columns for duplicate checking
        skip_identity_cols: List of identity columns to skip (auto-generated)
    """
    print(f"\n{'='*50}")
    print(f"Migrating {table_name}...")
    print('='*50)

    # Read from source
    query = f"SELECT * FROM [{source_schema}].[{table_name}]"
    try:
        df = pd.read_sql(query, source_engine)
        print(f"  Source rows: {len(df)}")
    except Exception as e:
        print(f"  ERROR reading source: {e}")
        return 0

    if len(df) == 0:
        print("  No data to migrate")
        return 0

    # Skip identity columns if specified
    if skip_identity_cols:
        for col in skip_identity_cols:
            if col in df.columns:
                df = df.drop(columns=[col])

    # Check for existing records in target
    if primary_keys:
        pk_cols = ', '.join([f"[{pk}]" for pk in primary_keys])
        existing_query = f"SELECT {pk_cols} FROM [{target_schema}].[{table_name}]"
        try:
            existing_df = pd.read_sql(existing_query, target_engine)
            print(f"  Existing rows in target: {len(existing_df)}")

            # Filter out existing records
            if len(existing_df) > 0:
                if len(primary_keys) == 1:
                    pk = primary_keys[0]
                    df = df[~df[pk].isin(existing_df[pk])]
                else:
                    # Multi-column primary key
                    existing_set = set(existing_df[primary_keys].apply(tuple, axis=1))
                    df = df[~df[primary_keys].apply(tuple, axis=1).isin(existing_set)]

            print(f"  New rows to insert: {len(df)}")
        except Exception as e:
            print(f"  Warning checking existing: {e}")

    if len(df) == 0:
        print("  All records already exist")
        return 0

    # Ensure timestamp columns exist
    if 'insert_date' not in df.columns:
        df['insert_date'] = pd.Timestamp.now()
    if 'update_date' not in df.columns:
        df['update_date'] = pd.Timestamp.now()

    # Insert into target
    try:
        df.to_sql(
            name=table_name,
            con=target_engine,
            schema=target_schema,
            if_exists='append',
            index=False
        )
        print(f"  Successfully inserted {len(df)} rows")
        return len(df)
    except Exception as e:
        print(f"  ERROR inserting: {e}")
        return 0


def run_migration():
    """Run the full migration"""
    print("\n" + "="*60)
    print("CBB DATA MIGRATION")
    print("="*60)
    print("Source: nhlpipe-sqlsvr database, CBB schema")
    print("Target: cbb database, dbo schema")
    print("="*60)

    # Get engines - source is the old NHL database with CBB schema
    # We need to create a custom engine for the source since get_engine('CBB') now points to new db
    from sqlalchemy import create_engine
    from urllib.parse import quote_plus
    from dotenv import load_dotenv

    load_dotenv()

    server = os.getenv("SPORTS_SERVER_NAME")
    username = os.getenv("AZURE_USERNAME")
    password = os.getenv("AZURE_PASSWORD")
    source_db = os.getenv("NHL_DB", "nhlpipe-sqlsvr")  # Old database name

    encoded_username = quote_plus(username)
    encoded_password = quote_plus(password)

    # Source engine (old database)
    source_url = f"mssql+pymssql://{encoded_username}:{encoded_password}@{server}/{source_db}"
    source_engine = create_engine(
        source_url,
        pool_pre_ping=True,
        pool_recycle=3600,
        connect_args={"timeout": 30, "login_timeout": 30}
    )

    # Target engine (new cbb database)
    target_engine = get_engine('CBB')

    print(f"\nSource: {source_db}")
    print(f"Target: cbb")

    # Migration order (respects foreign key dependencies)
    # Full list:
    # ("ConferenceInfo", ["conferenceId"], None),
    # ("VenueInfo", ["venueId"], None),
    # ("TeamInfo", ["teamId"], None),
    # ("PlayerInfo", ["playerId"], None),
    # ("GameInfo", ["gameId"], None),
    # ("GameBoxscoreTeam", ["gameId", "teamId"], ["boxscoreId"]),
    migrations = [
        # Table name, primary keys, identity columns to skip
        ("ConferenceInfo", ["conferenceId"], None),
        ("VenueInfo", ["venueId"], None),
        ("TeamInfo", ["teamId"], None),
        ("PlayerInfo", ["playerId"], None),
        ("GameInfo", ["gameId"], None),
        ("GameBoxscoreTeam", ["gameId", "teamId"], ["boxscoreId"]),
        ("GameBoxscorePlayer", ["gameId", "athleteId"], ["boxscoreId"]),
        ("GameLines", ["gameId"], ["linesId"]),
    ]

    results = {}

    for table_name, primary_keys, skip_identity in migrations:
        try:
            rows_inserted = migrate_table(
                source_engine=source_engine,
                target_engine=target_engine,
                table_name=table_name,
                source_schema='CBB',
                target_schema='dbo',
                primary_keys=primary_keys,
                skip_identity_cols=skip_identity
            )
            results[table_name] = {'success': True, 'rows': rows_inserted}
        except Exception as e:
            print(f"  FAILED: {e}")
            results[table_name] = {'success': False, 'error': str(e)}

    # Summary
    print("\n" + "="*60)
    print("MIGRATION SUMMARY")
    print("="*60)

    total_rows = 0
    for table_name, result in results.items():
        if result['success']:
            print(f"  {table_name}: {result['rows']} rows inserted")
            total_rows += result['rows']
        else:
            print(f"  {table_name}: FAILED - {result['error']}")

    print(f"\nTotal rows migrated: {total_rows}")
    print("="*60)

    return results


def verify_migration(target_engine=None):
    """Verify the migration by checking row counts"""
    if target_engine is None:
        target_engine = get_engine('CBB')

    print("\n" + "="*60)
    print("VERIFICATION - Target Database Row Counts")
    print("="*60)

    tables = [
        "ConferenceInfo", "VenueInfo", "TeamInfo", "PlayerInfo",
        "GameInfo", "GameBoxscoreTeam", "GameBoxscorePlayer", "GameLines"
    ]

    for table in tables:
        try:
            query = f"SELECT COUNT(*) as cnt FROM [dbo].[{table}]"
            result = pd.read_sql(query, target_engine)
            print(f"  {table}: {result['cnt'].iloc[0]} rows")
        except Exception as e:
            print(f"  {table}: ERROR - {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Migrate CBB data between databases')
    parser.add_argument('--verify-only', action='store_true', help='Only verify, do not migrate')
    args = parser.parse_args()

    if args.verify_only:
        verify_migration()
    else:
        run_migration()
        verify_migration()
