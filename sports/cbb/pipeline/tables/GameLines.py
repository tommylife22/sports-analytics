from datetime import date
import pandas as pd
import requests
import os
from dotenv import load_dotenv
from pandas import json_normalize

from .functions import (
    upsert_via_staging,
    est_date_range_to_utc
)

load_dotenv()

API_KEY = os.environ.get('CBB_API_KEY')
BASE_URL = "https://api.collegebasketballdata.com/lines"

def getGameLines(start_date: date, end_date: date) -> pd.DataFrame:
    
    start_utc, end_utc = est_date_range_to_utc(start_date, end_date)
    
    params = {
        "startDateRange": start_utc,
        "endDateRange": end_utc
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Accept": "application/json",
    }

    resp = requests.get(BASE_URL, params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    
    return pd.DataFrame(resp.json())

import pandas as pd
from pandas import json_normalize

def cleanGameLines(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten the /GameLines response.
    """

    df = df.copy()

    # Keep just gameId + lines
    base = df[["gameId", "lines"]].copy()

    # If lines is a list, explode so each element becomes its own row
    exploded = base.explode("lines", ignore_index=True)

    # Normalize the dicts in 'lines' into columns
    lines_flat = json_normalize(exploded["lines"])

    # Combine gameId + flattened line fields
    out = pd.concat(
        [
            exploded[["gameId"]].reset_index(drop=True),
            lines_flat.reset_index(drop=True),
        ],
        axis=1,
    )

    # Keep only the fields you care about
    keep_cols = [
        "gameId",
        "provider",
        "spread",
        "overUnder",
        "homeMoneyline",
        "awayMoneyline",
        "spreadOpen",
        "overUnderOpen",
    ]
    out = out[keep_cols]

    # (Optional) drop rows missing provider if you want strictly unique (gameId, provider)
    out = out.dropna(subset=["provider"])

    # (Optional) enforce uniqueness on (gameId, provider)
    out = out.drop_duplicates(subset=["gameId", "provider"])

    return out

import pandas as pd

def coerceGameLinesDtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce dtypes for GameLines to match SQL Server DDL.

    SQL Types:
      - gameId: VARCHAR(25)
      - provider: VARCHAR(255)
      - spread, overUnder, homeMoneyline, awayMoneyline,
        spreadOpen, overUnderOpen: DECIMAL(10,2)
      - PK = gameId
    """

    df = df.copy()

    # ---- VARCHAR columns ----
    varchar_cols = ["gameId", "provider"]

    for col in varchar_cols:
        if col in df.columns:
            # Ensure numeric IDs like 1234.0 become "1234"
            if col == "gameId":
                df[col] = pd.to_numeric(df[col], errors="coerce") \
                              .astype("Int64") \
                              .astype("string")
            else:
                df[col] = df[col].astype("string")

    # ---- DECIMAL columns → numeric ----
    decimal_cols = [
        "spread",
        "overUnder",
        "homeMoneyline",
        "awayMoneyline",
        "spreadOpen",
        "overUnderOpen",
    ]

    for col in decimal_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ---- Drop rows missing gameId (PK constraint) ----
    if "gameId" in df.columns:
        df = df[df["gameId"].notna()]

    return df

def loadGameLines(engine, startDate, endDate):
    
    df = coerceGameLinesDtypes(cleanGameLines(getGameLines(startDate, endDate)))
    
    df.rename(columns={'spread':'homeSpread'},inplace=True)
    
    pks = ['gameId','provider']
    data_columns = [c for c in df.columns if c not in pks + ["insert_date", "update_date"]]

    upsert_via_staging(
        df              = df,
        table_name      = "GameLines",
        primary_keys    = pks,
        data_columns    = data_columns,
        engine          = engine,
        schema          = 'dbo',
        dry_run         = False
    )