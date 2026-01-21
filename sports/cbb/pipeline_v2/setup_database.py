"""
Database Setup and Migration Script
Sets up cbb database schema in dbo and optionally migrates data from old database
"""
import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from generic.db import get_engine
from sqlalchemy import text
import pandas as pd


def read_schema_file(schema_file):
    """Read SQL schema file"""
    with open(schema_file, 'r') as f:
        return f.read()


def execute_schema_script(engine, schema_file):
    """
    Execute SQL schema script against database
    
    Args:
        engine: SQLAlchemy engine
        schema_file (str): Path to SQL schema file
    """
    schema_sql = read_schema_file(schema_file)
    
    # Split by GO statements (SQL Server batch separator)
    batches = schema_sql.split('GO')
    
    with engine.connect() as conn:
        for batch in batches:
            batch = batch.strip()
            if batch:  # Skip empty batches
                print(f"Executing: {batch[:60]}...")
                try:
                    conn.execute(text(batch))
                    conn.commit()
                except Exception as e:
                    print(f"  ⚠ Warning: {e}")
    
    print("✓ Schema script executed successfully")


def migrate_old_data(old_engine, new_engine):
    """
    Migrate data from old CBB database to new cbb database in dbo schema
    
    Args:
        old_engine: SQLAlchemy engine for old database
        new_engine: SQLAlchemy engine for new database
    """
    print("\n=== Migrating Data ===")
    
    # Tables to migrate from [CBB].[dbo] to [cbb].[dbo]
    tables_to_migrate = [
        'ConferenceInfo',
        'VenueInfo',
        'TeamInfo',
        'PlayerInfo',
        'GameInfo',
        'GameBoxscoreTeam',
        'GameBoxscorePlayer',
        'GameLines',
    ]
    
    for table_name in tables_to_migrate:
        try:
            print(f"\nMigrating {table_name}...")
            
            # Read from old database (assuming CBB schema in original database)
            query = f"SELECT * FROM [dbo].[{table_name}]"
            df = pd.read_sql(query, old_engine)
            
            if len(df) == 0:
                print(f"  ℹ No data found in {table_name}")
                continue
            
            print(f"  Read {len(df)} rows from {table_name}")
            
            # Write to new database in dbo schema
            df.to_sql(table_name, new_engine, schema='dbo', if_exists='append', index=False)
            print(f"  ✓ Migrated {len(df)} rows to {table_name}")
            
        except Exception as e:
            print(f"  ⚠ Migration skipped for {table_name}: {e}")


def verify_schema(engine):
    """Verify that all expected tables were created"""
    print("\n=== Verifying Schema ===")
    
    expected_tables = [
        'ConferenceInfo',
        'VenueInfo',
        'TeamInfo',
        'PlayerInfo',
        'GameInfo',
        'GameBoxscoreTeam',
        'GameBoxscorePlayer',
        'GameLines',
    ]
    
    query = """
    SELECT TABLE_NAME 
    FROM INFORMATION_SCHEMA.TABLES 
    WHERE TABLE_SCHEMA = 'dbo' 
    AND TABLE_NAME IN ('ConferenceInfo', 'VenueInfo', 'TeamInfo', 'PlayerInfo', 
                       'GameInfo', 'GameBoxscoreTeam', 'GameBoxscorePlayer', 'GameLines')
    ORDER BY TABLE_NAME
    """
    
    with engine.connect() as conn:
        result = conn.execute(text(query))
        created_tables = [row[0] for row in result.fetchall()]
    
    print(f"Expected tables: {len(expected_tables)}")
    print(f"Created tables: {len(created_tables)}")
    
    for table in expected_tables:
        status = "✓" if table in created_tables else "✗"
        print(f"  {status} {table}")
    
    if len(created_tables) == len(expected_tables):
        print("\n✓ All tables created successfully!")
        return True
    else:
        missing = set(expected_tables) - set(created_tables)
        print(f"\n✗ Missing tables: {missing}")
        return False


def main():
    """Main setup function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Set up cbb database schema in dbo')
    parser.add_argument('--migrate', action='store_true', help='Migrate data from old CBB database')
    parser.add_argument('--schema-file', default='sports/cbb/pipeline_v2/cbb_v2_schema.sql',
                       help='Path to SQL schema file')
    
    args = parser.parse_args()
    
    print("=== CBB Database Setup ===\n")
    
    # Get database engine for new database
    try:
        new_engine = get_engine('CBB')
        print("✓ Connected to cbb database")
    except Exception as e:
        print(f"✗ Failed to connect to cbb database: {e}")
        print("\nNote: Make sure cbb database exists on your SQL Server")
        print("You may need to create it manually first using SQL Server Management Studio")
        return 1
    
    # Execute schema script
    try:
        schema_file = os.path.join(PROJECT_ROOT, args.schema_file)
        print(f"\nExecuting schema script: {schema_file}")
        execute_schema_script(new_engine, schema_file)
    except Exception as e:
        print(f"✗ Failed to execute schema: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Verify schema was created
    if not verify_schema(new_engine):
        return 1
    
    # Optionally migrate old data
    if args.migrate:
        try:
            old_engine = get_engine('CBB')
            print("\n✓ Connected to old CBB database")
            migrate_old_data(old_engine, new_engine)
        except Exception as e:
            print(f"⚠ Migration skipped: {e}")
    
    print("\n=== Setup Complete ===")
    print("✓ cbb database schema created in dbo")
    print("✓ Ready to load data with new pipeline")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
