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
BASE_URL = "https://api.collegebasketballdata.com/games/teams"

ROW_LIMIT = 3000

def getTeamBoxscores(start_date: date, end_date: date) -> pd.DataFrame:
    
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

def getTeamBoxscoresBatches(startDate: date, endDate: date) -> pd.DataFrame:
    
    all_frames = []
    current_start = startDate

    while current_start <= endDate:

        # Start by trying a wide window (e.g., 14 days)
        window_size = 14

        while True:
            current_end = current_start + timedelta(days=window_size)
            if current_end > endDate:
                current_end = endDate

            df = getTeamBoxscores(current_start, current_end)
            row_count = len(df)

            print(f"Pulled {row_count:4d} rows → {current_start} → {current_end}")

            # CASE 1 → safe window (< 3000 rows)
            if row_count < ROW_LIMIT:
                all_frames.append(df)
                break

            # CASE 2 → window too large (== 3000 rows)
            else:
                window_size = max(1, window_size // 2)
                print(f"⚠ Hit API limit (3000 rows). Shrinking window → {window_size} days.")

                # If even 1 day hits the limit, you need a fallback strategy
                if window_size == 1:
                    # You may need to break down even further: per hour or per game
                    # But for NCAA/CBB this is extremely unlikely
                    print("1-day window still hit 3000 rows. Need smaller granularity.")
                    break

        # advance forward
        current_start = current_end + timedelta(days=1)

    # Combine all batches
    if all_frames:
        return pd.concat(all_frames, ignore_index=True)
    else:
        return pd.DataFrame()

def cleanTeamBoxscores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Take the raw /games/teams response and return a flattened
    team-game stats table with only the *team's* stats.

    - Keeps: gameId, teamId, opponentId, pace
    - Expands: teamStats into columns
    """

    base_cols = ["gameId", "teamId", "opponentId", "pace"]

    # Filter out rows where teamStats is NaN or not a dict
    df = df[df["teamStats"].notna()]
    df = df[df["teamStats"].apply(lambda x: isinstance(x, dict))]

    if df.empty:
        return pd.DataFrame()

    base = df[base_cols].copy()

    # Expand teamStats dict → columns
    team_stats = json_normalize(df["teamStats"])

    # Option 1: prefix with "team." and then camelCase via fix_col_name
    team_stats.columns = [
        fix_col_name(f"{c}") for c in team_stats.columns
    ]

    # Combine
    out = pd.concat([base, team_stats], axis=1)
    
    out.drop(['pointsByPeriod'],inplace=True,axis=1)
    
    out['gameScore'] = out['gameScore'].round(2)

    return out

def coerceTeamBoxscoresDtypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1) String columns (VARCHAR)
    str_cols = [
        'gameId',
        'teamId',
        'opponentId',
    ]

    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].astype('string')

    # 2) Integer columns (INT)
    int_cols = [
        'possessions',
        'assists',
        'steals',
        'blocks',
        'pointsTotal',
        'pointsLargestLead',
        'pointsFastBreak',
        'pointsInPaint',
        'pointsOffTurnovers',
        'twoPointFieldGoalsMade',
        'twoPointFieldGoalsAttempted',
        'threePointFieldGoalsMade',
        'threePointFieldGoalsAttempted',
        'freeThrowsMade',
        'freeThrowsAttempted',
        'fieldGoalsMade',
        'fieldGoalsAttempted',
        'turnoversTotal',
        'turnoversTeamTotal',
        'reboundsOffensive',
        'reboundsDefensive',
        'reboundsTotal',
        'foulsTotal',
        'foulsTechnical',
        'foulsFlagrant',
    ]

    for col in int_cols:
        if col in df.columns:
            df[col] = (
                pd.to_numeric(df[col], errors='coerce')
                  .round(0)             # handles any float-y ints like 5.0
                  .astype('Int64')      # nullable integer
            )

    # 3) Decimal columns (DECIMAL(10,2))
    dec_cols = [
        'pace',
        'trueShooting',
        'rating',
        'gameScore',
        'twoPointFieldGoalsPct',
        'threePointFieldGoalsPct',
        'freeThrowsPct',
        'fieldGoalsPct',
        'fourFactorsEffectiveFieldGoalPct',
        'fourFactorsFreeThrowRate',
        'fourFactorsTurnoverRatio',
        'fourFactorsOffensiveReboundPct',
    ]

    for col in dec_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)

    # 4) Datetime columns (DATETIME)
    datetime_cols = ['insert_date', 'update_date']

    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            # strip timezone if present
            if pd.api.types.is_datetime64tz_dtype(df[col]):
                df[col] = df[col].dt.tz_localize(None)
                
    df = df.drop_duplicates()

    return df

def loadGameBoxscoreTeam(engine, startDate: date, endDate: date):
    
    boxscores = coerceTeamBoxscoresDtypes(cleanTeamBoxscores(getTeamBoxscoresBatches(startDate,endDate)))
    
    pks = ["gameId","teamId"]
    data_columns = [c for c in boxscores.columns if c not in pks + ["insert_date", "update_date"]]

    upsert_via_staging(
        df              = boxscores,
        table_name      = "GameBoxscoreTeam",
        primary_keys    = pks,
        data_columns    = data_columns,
        engine          = engine,
        schema          = 'dbo',
        dry_run         = False
    )