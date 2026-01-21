import os
import re
import time
import json
import unicodedata
import urllib.parse
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Tuple, List
from zoneinfo import ZoneInfo

import dataclasses
import numpy as np
import pandas as pd
from pandas import json_normalize
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from dotenv import load_dotenv

load_dotenv()



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

    return create_stmt

# === General cleaning functions ===

def clean_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Replace NaN / NaT with None for SQL NULLs."""
    return df.replace({np.nan: None, pd.NaT: None})

def convert_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Converts all columns containing 'date' in their name to datetime dtype."""
    for col in df.columns:
        if "date" in col.lower():
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def clean_column(name: object, ascii_only: bool = False) -> str:
    """Normalize a single column name to lowercase snake_case."""
    s = unicodedata.normalize("NFKC", str(name))

    if ascii_only:
        s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")

    # CamelCase / PascalCase splitting
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", s)
    s = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", s)

    # Replace non-alphanumeric with underscores
    s = re.sub(r"[^0-9A-Za-z]+", "_", s)

    # Lowercase
    s = s.lower()

    # Collapse multiple underscores and trim
    s = re.sub(r"_+", "_", s).strip("_")

    # Ensure doesn't start with digit
    if s and s[0].isdigit():
        s = "_" + s

    return s or "col"

def clean_dataframe_columns(df: pd.DataFrame, ascii_only: bool = False) -> pd.DataFrame:
    """Clean all column names in a dataframe to snake_case."""
    df = df.copy()
    df.columns = [clean_column(col, ascii_only=ascii_only) for col in df.columns]
    return df


# === Upsert helpers and function ===

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



# === Get engine function ===

def get_engine() -> Engine:

    server      = os.environ.get('SPORTS_SERVER_NAME')
    database    = os.environ.get('SPORTS_DB_NAME')
    username    = os.environ.get('AZURE_USERNAME')
    password    = os.environ.get('AZURE_PASSWORD')

    # ODBC connection string for Driver 18
    odbc_str = (
        "Driver={ODBC Driver 18 for SQL Server};"
        f"Server=tcp:{server},1433;"
        f"Database={database};"
        f"Uid={username};"
        f"Pwd={password};"
        "Encrypt=yes;"
        "TrustServerCertificate=no;"
        "Connection Timeout=30;"
    )

    params = urllib.parse.quote_plus(odbc_str)
    sqlalchemy_url = f"mssql+pyodbc:///?odbc_connect={params}"


    return create_engine(sqlalchemy_url)


# === API Helper Functions ===

def est_date_range_to_utc(start_date: date, end_date: date) -> tuple[str, str]:
    
    est = ZoneInfo("America/New_York")
    utc = ZoneInfo("UTC")
    
    # Midnight at start_date in EST
    start_est = datetime.combine(start_date, datetime.min.time(), tzinfo=est)
    # Midnight after end_date in EST (i.e., end_date + 1)
    end_est = datetime.combine(end_date + timedelta(days=1), datetime.min.time(), tzinfo=est)

    start_utc = start_est.astimezone(utc)
    end_utc = end_est.astimezone(utc)
        
    return (
        start_utc.isoformat().replace("+00:00", "Z"),
        end_utc.isoformat().replace("+00:00", "Z"),
    )
    
def fix_col_name(col: str) -> str:
    """
    Convert names like 'points.byPeriod' → 'pointsByPeriod'.
    Removes dots and capitalizes the next character.
    """
    if "." not in col:
        return col

    parts = col.split(".")
    # First segment unchanged, capitalize each next segment
    return parts[0] + "".join(p[:1].upper() + p[1:] for p in parts[1:])