"""
Database Setup and Migration Script
Sets up CBB_V2 database schema and optionally migrates data from old database
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


def create_database_if_not_exists(server, username, password, database_name):
    """
    Create a new database if it doesn't exist
    
    Args:
        server (str): SQL Server name
        username (str): Username
        password (str): Password
        database_name (str): Database name to create
    """
    from urllib.parse import quote_plus
    from sqlalchemy import create_engine, text
    
    # Connect to master to create new database
    connection_string = f"mssql+pyodbc://{username}:{quote_plus(password)}@{server}/master?driver=ODBC+Driver+17+for+SQL+Server"
    engine = create_engine(connection_string)
    
    with engine.connect() as conn:
        # Check if database exists
        check_db = f"SELECT database_id FROM sys.databases WHERE name = '{database_name}'"
        result = conn.execute(text(check_db))
        
        if result.fetchone() is None:
            print(f"✓ Creating database: {database_name}")
            conn.execute(text(f"CREATE DATABASE [{database_name}]"))
            conn.commit()
            print(f"✓ Database created successfully")
        else:
            print(f"ℹ Database {database_name} already exists")
    
    engine.dispose()


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
    Migrate data from old CBB database to new CBB_V2 database
    
    Args:
        old_engine: SQLAlchemy engine for old database
        new_engine: SQLAlchemy engine for new database
    """
    print("\n=== Migrating Data ===")
    
    tables_to_migrate = [
        ('TeamInfo', 'Team'),
        ('GameInfo', 'Game'),
        ('PlayerInfo', 'Player'),
    ]
    
    for old_table, new_table in tables_to_migrate:
        try:
            print(f"\nMigrating {old_table} → {new_table}...")
            
            # Read from old database
            query = f"SELECT * FROM [CBB].[{old_table}]"
            df = pd.read_sql(query, old_engine)
            print(f"  Read {len(df)} rows from {old_table}")
            
            # Write to new database
            df.to_sql(new_table, new_engine, schema='CBB_V2', if_exists='append', index=False)
            print(f"  ✓ Migrated {len(df)} rows to {new_table}")
            
        except Exception as e:
            print(f"  ⚠ Migration skipped: {e}")


def main():
    """Main setup function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Set up CBB_V2 database schema')
    parser.add_argument('--migrate', action='store_true', help='Migrate data from old CBB database')
    parser.add_argument('--schema-file', default='sports/cbb/pipeline_v2/cbb_v2_schema.sql',
                       help='Path to SQL schema file')
    
    args = parser.parse_args()
    
    print("=== CBB Pipeline V2 Database Setup ===\n")
    
    # Get database engine
    try:
        new_engine = get_engine('CBB_V2')
        print("✓ Connected to CBB_V2 database")
    except Exception as e:
        print(f"✗ Failed to connect to CBB_V2: {e}")
        print("\nNote: Make sure CBB_V2 database exists on your SQL Server")
        print("You may need to create it manually first using SQL Server Management Studio")
        return 1
    
    # Execute schema script
    try:
        schema_file = os.path.join(PROJECT_ROOT, args.schema_file)
        print(f"\nExecuting schema script: {schema_file}")
        execute_schema_script(new_engine, schema_file)
    except Exception as e:
        print(f"✗ Failed to execute schema: {e}")
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
    print("✓ CBB_V2 database schema created")
    print("✓ Ready to load data with new pipeline")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
