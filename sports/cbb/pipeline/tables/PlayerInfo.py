import pandas as pd
import requests
import os
from dotenv import load_dotenv
from pandas import json_normalize

from .functions import upsert_via_staging

load_dotenv()

API_KEY = os.environ.get('CBB_API_KEY')
BASE_URL = "https://api.collegebasketballdata.com/teams/roster"

def getPlayerInfo(season: int) -> pd.DataFrame:
    
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

def cleanPlayerInfo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Take the raw team rosters DataFrame from the API and return
    a flattened PlayerInfo DataFrame.

    Output grain: one row per (season, teamId, playerId/athleteId).
    Keeps teamId + season, expands each player dict into columns.
    """

    df = df.copy()

    # Keep just the keys we care about + the players list
    base_cols = ["teamId", "season", "players"]
    missing = [c for c in base_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    # 1) Explode the players list so each row has a single player dict
    exploded = df[base_cols].explode("players", ignore_index=True)

    # 2) Flatten each player dict into columns
    exploded = exploded[exploded["players"].notna()]
    players_flat = json_normalize(exploded["players"], sep=".")

    # Optional: clean column names like "bio.height" → "bioHeight"
    try:
        from pipeline.tables.functions import fix_col_name
        players_flat.columns = [fix_col_name(c) for c in players_flat.columns]
    except ImportError:
        # fallback if fix_col_name isn't available in this context
        players_flat.columns = players_flat.columns.str.replace(r"\.", "_", regex=True)

    # 3) Combine team/season keys with player columns
    out = pd.concat(
        [
            exploded[["teamId", "season"]].reset_index(drop=True),
            players_flat.reset_index(drop=True),
        ],
        axis=1,
    )
    
    out.rename(columns={'id':'playerId'},inplace=True)
    
    for col in ['dateOfBirth','hometownCity','hometownCountry','hometownLatitude','hometownLongitude','hometownCountyFips']:
        if col in out.columns:
            out.drop([col],inplace=True,axis=1)

    return out

def coercePlayerInfoDtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce dtypes for PlayerInfo to match SQL Server DDL.

    SQL Types:
      - teamId, playerId, sourceId, name, firstName, lastName,
        jersey, position, hometownCity, hometownState → VARCHAR → string
      - season, startSeason, endSeason → INT
      - height, weight → INT
      - PK = playerId → must not be null
    """

    df = df.copy()

    # ---- ID-like columns → convert numeric → remove decimals → string ----
    id_like_cols = ["teamId", "playerId", "sourceId"]

    for col in id_like_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].astype("Int64")        # supports NA
            df[col] = df[col].astype("string")

    # ---- INT columns (true numeric) ----
    int_cols = ["season", "startSeason", "endSeason", "height", "weight"]

    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    # ---- VARCHAR columns (everything else) ----
    varchar_cols = [
        "name", "firstName", "lastName",
        "jersey", "position",
        "hometownCity", "hometownState"
    ]

    for col in varchar_cols:
        if col in df.columns:
            df[col] = df[col].astype("string")

    # ---- Drop rows missing playerId (cannot insert into PK) ----
    if "playerId" in df.columns:
        df = df[df["playerId"].notna()]

    return df

def loadPlayerInfo(engine,season):
    
    df = coercePlayerInfoDtypes(cleanPlayerInfo(getPlayerInfo(season)))
    
    pks = ['playerId']
    data_columns = [c for c in df.columns if c not in pks + ["insert_date", "update_date"]]

    upsert_via_staging(
        df              = df,
        table_name      = "PlayerInfo",
        primary_keys    = pks,
        data_columns    = data_columns,
        engine          = engine,
        schema          = 'dbo',
        dry_run         = False
    )