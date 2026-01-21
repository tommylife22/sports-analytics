import os
from typing import List
import time
from urllib.parse import quote_plus

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from dotenv import load_dotenv

load_dotenv()

def clean_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Replace NaN / NaT with None for SQL NULLs."""
    return df.replace({np.nan: None, pd.NaT: None})

def retry_on_connection_error(func, max_retries=3, delay=2):
    """
    Retry a database operation on connection failures.

    Args:
        func: Function to execute
        max_retries: Maximum number of retry attempts
        delay: Seconds to wait between retries

    Returns:
        Result of the function call
    """
    last_error = None
    for attempt in range(max_retries):
        try:
            return func()
        except OperationalError as e:
            last_error = e
            if attempt < max_retries - 1:
                print(f"Connection failed (attempt {attempt + 1}/{max_retries}). Retrying in {delay}s...")
                time.sleep(delay)
            else:
                print(f"Connection failed after {max_retries} attempts.")
    raise last_error

def get_engine(sport: str):

    server      = os.environ.get('SPORTS_SERVER_NAME')
    username    = os.environ.get('AZURE_USERNAME')
    password    = os.environ.get('AZURE_PASSWORD')
    
    cbb_database    = os.environ.get('CBB_DB')
    mlb_database    = os.environ.get('MLB_DB')
    nhl_database    = os.environ.get('NHL_DB')

    # URL-encode username and password to handle special characters like @, :, etc.
    encoded_username = quote_plus(username)
    encoded_password = quote_plus(password)

    # pymssql connection string - much simpler than pyodbc
    
    if sport == 'MLB':
        sqlalchemy_url = f"mssql+pymssql://{encoded_username}:{encoded_password}@{server}/{mlb_database}"
    elif sport == 'NHL':
        sqlalchemy_url = f"mssql+pymssql://{encoded_username}:{encoded_password}@{server}/{nhl_database}"
    elif sport == 'CBB':
        sqlalchemy_url = f"mssql+pymssql://{encoded_username}:{encoded_password}@{server}/{cbb_database}"

    # Add connection pool settings to handle transient failures
    return create_engine(
        sqlalchemy_url,
        pool_pre_ping=True,  # Verify connections before using them
        pool_recycle=3600,   # Recycle connections after 1 hour
        connect_args={
            "timeout": 30,
            "login_timeout": 30
        }
    )

def ensure_staging_table(engine, table_name: str, staging_table: str, schema: str = "dbo") -> None:
    """
    Ensure a staging table exists with the same schema as the target table.
    If it doesn't exist, create it as a TOP 0 clone of the target.
    If it does exist, TRUNCATE it.

    This guarantees target and staging schemas always match.
    """
    sql = f"""
    IF OBJECT_ID('{schema}.{staging_table}', 'U') IS NULL
    BEGIN
        SELECT TOP 0 *
        INTO [{schema}].[{staging_table}]
        FROM [{schema}].[{table_name}];
    END
    ELSE
    BEGIN
        TRUNCATE TABLE [{schema}].[{staging_table}];
    END
    """
    with engine.begin() as conn:
        conn.execute(text(sql))
        
def upload_to_staging(
    df: pd.DataFrame,
    staging_table: str,
    engine,
    schema: str = None,
    if_exists: str = "append",  # default append now
):
    """
    Upload df to an existing staging table.
    Assumes schema already matches the target (created via ensure_staging_table).
    """
    df = df.copy()

    # ensure timestamp columns exist
    if "insert_date" not in df.columns:
        df["insert_date"] = pd.NaT
    if "update_date" not in df.columns:
        df["update_date"] = pd.NaT

    df = clean_nulls(df)

    to_sql_kwargs = dict(
        name=staging_table,
        con=engine,
        if_exists=if_exists,
        index=False,
    )
    if schema:
        to_sql_kwargs["schema"] = schema

    df.to_sql(**to_sql_kwargs)
    
def run_dynamic_merge_sql(engine, table_name: str, primary_keys: List[str], data_columns: List[str],
                         staging_table: str = None, schema: str = "dbo"):
   """
   Generates and runs a dynamic MERGE statement for SQL Server using a staging table.

   - engine: SQLAlchemy engine
   - table_name: target table name (no schema)
   - primary_keys: list of PK column names
   - data_columns: list of non-key data columns (exclude insert_date/update_date)
   - staging_table: name of staging table (if None, 'Staging_' + table_name is used)
   - schema: schema name, default 'dbo'
   """
   if staging_table is None:
       staging_table = f"Staging_{table_name}"

   # quoted identifier helpers
   def q(col): return f"[{col}]"
   def q_table(name): return f"[{schema}].[{name}]" if schema else f"{name}"

   match_condition = " AND ".join([f"T.{q(pk)} = S.{q(pk)}" for pk in primary_keys])

   # Build update-check: detect inequality including NULL/NOT NULL transitions
   update_check = " OR ".join([
       f"(T.{q(col)} != S.{q(col)} OR (T.{q(col)} IS NULL AND S.{q(col)} IS NOT NULL) OR (T.{q(col)} IS NOT NULL AND S.{q(col)} IS NULL))"
       for col in data_columns
   ]) or "1=0"  # fallback if no data columns

   update_set = ", ".join([f"T.{q(col)} = S.{q(col)}" for col in data_columns] + ["T.update_date = CURRENT_TIMESTAMP"])

   insert_columns = primary_keys + data_columns + ["insert_date", "update_date"]
   insert_values = [f"S.{q(col)}" for col in primary_keys + data_columns] + ["CURRENT_TIMESTAMP", "CURRENT_TIMESTAMP"]

   merge_sql = f"""
   MERGE {q_table(table_name)} AS T
   USING {q_table(staging_table)} AS S
   ON {match_condition}
   WHEN MATCHED AND (
       {update_check}
   )
   THEN UPDATE SET
       {update_set}
   WHEN NOT MATCHED BY TARGET THEN
   INSERT ({', '.join([q(col) for col in insert_columns])})
   VALUES ({', '.join(insert_values)});
   """

   # Execute merge within transaction
   with engine.begin() as conn:
       conn.execute(text(merge_sql))
       


def upsert_via_staging(
    df: pd.DataFrame,
    table_name: str,
    primary_keys: List[str],
    data_columns: List[str],
    engine,
    staging_prefix: str = "Staging_",
    schema: str = "dbo",
    drop_stage: bool = False,   # change default: we now usually KEEP staging
    dry_run: bool = False,
):
    """
    Orchestrator: ensure staging schema matches target, upload data, run MERGE.

    - Staging table is created (if needed) as a clone of the target table.
    - Data is loaded with to_sql(if_exists='append') to a fixed schema.
    """
    staging_table = f"{staging_prefix}{table_name}"

    if dry_run:
        return {"action": "dry_run", "staging_table": staging_table}

    # 1) Ensure staging exists with the same schema as target, and is empty
    ensure_staging_table(engine, table_name, staging_table, schema=schema)

    # 2) Upload df into staging (append to existing structure)
    upload_to_staging(
        df=df,
        staging_table=staging_table,
        engine=engine,
        schema=schema,
        if_exists="append",   # <-- append into fixed schema
    )

    # 3) Merge from staging into target
    try:
        run_dynamic_merge_sql(
            engine,
            table_name,
            primary_keys,
            data_columns,
            staging_table=staging_table,
            schema=schema,
        )
    except Exception:
        # Leave staging table in place for debugging
        raise

    # 4) Optional: drop staging table (I'd keep it by default for troubleshooting)
    dropped = False
    if drop_stage:
        drop_sql = (
            f"IF OBJECT_ID('{schema}.{staging_table}', 'U') IS NOT NULL "
            f"DROP TABLE [{schema}].[{staging_table}];"
        )
        with engine.begin() as conn:
            conn.execute(text(drop_sql))
        dropped = True

    return {"staging_table": staging_table, "dropped": dropped}

def generate_create_sqlserver(df, table_name, schema="dbo", primary_keys=None):
    """
    Generate a SQL Server CREATE TABLE statement from a Pandas DataFrame.
    """

    # SQL Server dtype mapping
    dtype_mapping = {
        "object": "VARCHAR(255)",
        "int64": "INT",
        "int32": "INT",
        "float64": "DECIMAL(19,4)",
        "float32": "DECIMAL(19,4)",
        "bool": "BIT",
        "datetime64": "DATETIME",
        "datetime64[ns]": "DATETIME2",
        "datetime64[ns, UTC]": "DATETIME2",
    }

    # Build column definitions
    col_defs = []
    for col in df.columns:
        pd_type = str(df[col].dtype)
        sql_type = dtype_mapping.get(pd_type, "NVARCHAR(255)")
        col_defs.append(f"    [{col}] {sql_type}")

    # Add timestamp columns at the end
    col_defs.append("    [insert_date] DATETIME DEFAULT GETDATE()")
    col_defs.append("    [update_date] DATETIME DEFAULT GETDATE()")

    # Join with newlines
    col_defs_sql = ",\n".join(col_defs)

    # Primary key clause
    if primary_keys:
        if isinstance(primary_keys, str):
            pk_cols = f"[{primary_keys}]"
        else:
            pk_cols = ", ".join(f"[{pk}]" for pk in primary_keys)

        pk_clause = f",\n    CONSTRAINT PK_{table_name} PRIMARY KEY ({pk_cols})"
    else:
        pk_clause = ""

    # Final statement
    create_stmt = (
        f"CREATE TABLE [{schema}].[{table_name}] (\n"
        f"{col_defs_sql}"
        f"{pk_clause}\n"
        ");"
    )

    return print(create_stmt)