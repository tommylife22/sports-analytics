
from datetime import date, timedelta
import pandas as pd
import requests
import os
from dotenv import load_dotenv
from pandas import json_normalize

from .functions import (
    upsert_via_staging,
    est_date_range_to_utc,
    fix_col_name
)

load_dotenv()

API_KEY = os.environ.get('CBB_API_KEY')
BASE_URL = "https://api.collegebasketballdata.com/games/players"

ROW_LIMIT = 1000

def getPlayerBoxscores(start_date: date, end_date: date) -> pd.DataFrame:
    
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

def getPlayerBoxscoresBatches(startDate: date, endDate: date) -> pd.DataFrame:
    
    all_frames = []
    current_start = startDate

    while current_start <= endDate:

        # Start by trying a wide window (e.g., 14 days)
        window_size = 14

        while True:
            current_end = current_start + timedelta(days=window_size)
            if current_end > endDate:
                current_end = endDate

            df = getPlayerBoxscores(current_start, current_end)
            row_count = len(df)

            print(f"Pulled {row_count:4d} rows → {current_start} → {current_end}")

            # CASE 1 → safe window (< 1000 rows)
            if row_count < ROW_LIMIT:
                all_frames.append(df)
                break

            # CASE 2 → window too large (== 1000 rows)
            else:
                window_size = max(1, window_size // 2)
                print(f"⚠ Hit API limit (1000 rows). Shrinking window → {window_size} days.")

                # If even 1 day hits the limit, you need a fallback strategy
                if window_size == 1:
                    # You may need to break down even further: per hour or per game
                    # But for NCAA/CBB this is extremely unlikely
                    print("1-day window still hit 1000 rows. Need smaller granularity.")
                    break

        # advance forward
        current_start = current_end + timedelta(days=1)

    # Combine all batches
    if all_frames:
        return pd.concat(all_frames, ignore_index=True)
    else:
        return pd.DataFrame()
    

def cleanPlayerBoxscores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Take the raw /games/players response and return a flattened
    player-game stats table with only the *player's* stats.

    - Keeps: gameId, teamId, opponentId
    - Expands: players into columns
    """
    
    base_cols = ["gameId", "teamId", "opponentId","players"]

    # 1) Keep just the columns we need and explode the list
    exploded = df[base_cols].explode("players", ignore_index=True)

    # 2) Normalize the player dict into columns
    player_stats = json_normalize(exploded["players"], sep=".")

    # 3) Clean up column names (flatten "fieldGoals.made" → "fieldGoalsMade")
    player_stats.columns = [fix_col_name(c) for c in player_stats.columns]

    # 4) Combine game/team/opponent keys with player stats
    out = pd.concat(
        [
            exploded[["gameId", "teamId", "opponentId"]].reset_index(drop=True),
            player_stats.reset_index(drop=True),
        ],
        axis=1,
    )

    return out

import pandas as pd

def coercePlayerBoxscoresDtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce dtypes for GameBoxscorePlayer based on SQL Server DDL
    and drop rows missing key IDs.
    """

    df = df.copy()

    # --- IDs (as VARCHAR in SQL, but often numeric in JSON) ---

    id_cols = ["gameId", "teamId", "opponentId", "athleteId", "athleteSourceId"]
    for col in id_cols:
        if col in df.columns:
            # Try numeric first to strip any .0, then back to string
            df[col] = pd.to_numeric(df[col], errors="coerce")
            # For sourceId it might truly be string; fall back if all NaN
            if df[col].isna().all():
                df[col] = df[col].astype("string")
            else:
                df[col] = df[col].astype("Int64").astype("string")

    # Drop any rows missing core PK components
    for col in ["gameId", "teamId", "athleteId"]:
        if col in df.columns:
            df = df[df[col].notna()]

    # --- VARCHAR columns (already covered by id_cols + name/position) ---

    varchar_cols = ["name", "position"]
    for col in varchar_cols:
        if col in df.columns:
            df[col] = df[col].astype("string")

    # --- BIT columns → boolean ---
    bit_cols = ["starter", "ejected"]
    for col in bit_cols:
        if col in df.columns:
            df[col] = df[col].astype("boolean")

    # --- INT columns → Int64 (nullable integer) ---
    int_cols = [
        "minutes", "points", "turnovers", "fouls", "assists",
        "steals", "blocks",
        "fieldGoalsMade", "fieldGoalsAttempted",
        "twoPointFieldGoalsMade", "twoPointFieldGoalsAttempted",
        "threePointFieldGoalsMade", "threePointFieldGoalsAttempted",
        "freeThrowsMade", "freeThrowsAttempted",
        "reboundsOffensive", "reboundsDefensive", "reboundsTotal",
    ]
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    # --- DECIMAL columns → float ---
    decimal_cols = [
        "gameScore", "offensiveRating", "defensiveRating",
        "netRating", "usage", "effectiveFieldGoalPct", "trueShootingPct",
        "assistsTurnoverRatio", "freeThrowRate", "offensiveReboundPct",
        "fieldGoalsPct", "twoPointFieldGoalsPct", "threePointFieldGoalsPct",
        "freeThrowsPct",
    ]
    for col in decimal_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

def loadGameBoxscorePlayer(engine, startDate: date, endDate: date):
    
    df = coercePlayerBoxscoresDtypes(cleanPlayerBoxscores(getPlayerBoxscoresBatches(startDate,endDate))).drop_duplicates()
    
    pks = ["gameId","teamId","athleteId"]
    data_columns = [c for c in df.columns if c not in pks + ["insert_date", "update_date"]]

    upsert_via_staging(
        df              = df,
        table_name      = "GameBoxscorePlayer",
        primary_keys    = pks,
        data_columns    = data_columns,
        engine          = engine,
        schema          = 'CBB',
        dry_run         = False
    )