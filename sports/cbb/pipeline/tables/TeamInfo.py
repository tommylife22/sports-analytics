import pandas as pd
import requests
import os
from dotenv import load_dotenv

from .functions import upsert_via_staging

load_dotenv()

API_KEY = os.environ.get('CBB_API_KEY')
BASE_URL = "https://api.collegebasketballdata.com/teams"


def getTeamInfo(season: int) -> pd.DataFrame:
    
    params = {
        "season": season
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Accept": "application/json",
    }

    resp = requests.get(BASE_URL, params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    
    return pd.DataFrame(resp.json())

def coerce_team_info_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce all TeamInfo columns to string.
    Ensures ID-like fields lose any '.0' decimals by converting
    numeric → Int64 → string.
    """

    df = df.copy()

    # Columns that must be treated as numeric IDs cleanly
    id_like_cols = ["id", "sourceId", "conferenceId", "currentVenueId"]

    for col in df.columns:
        if col in id_like_cols:
            # Convert to numeric to strip decimals, then back to string
            df[col] = pd.to_numeric(df[col], errors="coerce")   # float -> int
            df[col] = df[col].astype("Int64")                   # support NA
            df[col] = df[col].astype("string")                  # final type
        else:
            # Plain string columns
            df[col] = df[col].astype("string")

    return df

def cleanTeamInfo(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.copy()
    
    df.rename(columns={'id':'teamId'},inplace=True)

    # Columns that must be treated as numeric IDs cleanly
    id_like_cols = ["teamId", "sourceId", "conferenceId", "currentVenueId"]

    for col in df.columns:
        if col in id_like_cols:
            # Convert to numeric to strip decimals, then back to string
            df[col] = pd.to_numeric(df[col], errors="coerce")   # float -> int
            df[col] = df[col].astype("Int64")                   # support NA
            df[col] = df[col].astype("string")                  # final type
        else:
            # Plain string columns
            df[col] = df[col].astype("string")

    return df

def loadTeamInfo(engine,season):
    
    df = cleanTeamInfo(getTeamInfo(season))

    pks = ["teamId"]
    data_columns = [c for c in df.columns if c not in pks + ["insert_date", "update_date"]]

    upsert_via_staging(
        df              = df,
        table_name      = "TeamInfo",
        primary_keys    = pks,
        data_columns    = data_columns,
        engine          = engine,
        schema          = 'CBB',
        dry_run         = False
    )
    
#loadTeamInfo(engine, season)