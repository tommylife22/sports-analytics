from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
import pandas as pd
import requests
import os

from .functions import (
    convert_date_columns,
    upsert_via_staging,
    est_date_range_to_utc
)

API_KEY = os.environ.get('CBB_API_KEY')
BASE_URL = "https://api.collegebasketballdata.com/games"

    
def getGameInfo(start_date: date, end_date: date) -> pd.DataFrame:
    
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

def cleanGameInfo(games: pd.DataFrame):
       
    games = convert_date_columns(games)
    games.drop(['homePeriodPoints','awayPeriodPoints'],inplace=True,axis=1)
    games.rename(columns={'id':'gameId'},inplace=True)

    # 1) Datetime
    games['startDate'] = pd.to_datetime(games['startDate'], errors='coerce')
    # if it has tz info:
    if pd.api.types.is_datetime64tz_dtype(games['startDate']):
        games['startDate'] = games['startDate'].dt.tz_localize(None)

    # 2) Integer columns (match your DDL)
    int_cols = [
        'season',
        'attendance',
        'homeSeed',
        'homePoints',
        'awaySeed',
        'awayPoints',
    ]

    for col in int_cols:
        if col in games.columns:
            games[col] = (
                pd.to_numeric(games[col], errors='coerce')
                .round(0)
                .astype('Int64')
            )

    # 3) BIT columns -> 0/1 ints
    bit_cols = [
        'startTimeTbd',
        'neutralSite',
        'conferenceGame',
        'homeWinner',
        'awayWinner',
    ]

    for col in bit_cols:
        if col in games.columns:
            games[col] = (
                games[col]
                .astype(float)
                .fillna(0)
                .astype(int)
            )

    # 4) DECIMAL(19,4) column
    if 'excitement' in games.columns:
        games['excitement'] = pd.to_numeric(games['excitement'], errors='coerce')
        
    return games

def coerceGameInfoDtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce dtypes for GameInfo to match SQL Server DDL.

    SQL Types (summary):
      - PK: sourceId (VARCHAR 25)
      - VARCHAR fields → string dtype
      - INT fields → Int64
      - BIT fields → 0/1 (Int64)
      - DATETIME2 fields → datetime64
      - DECIMAL(19,4) → float rounded to 4 decimals
    """

    df = df.copy()

    # =========================
    #   ID-like → string
    # =========================
    id_like_cols = [
        "gameId", "sourceId", "seasonLabel", "seasonType",
        "tournament", "gameType", "status",
        "homeTeamId", "homeConferenceId",
        "awayTeamId", "awayConferenceId",
        "venueId"
    ]

    for col in id_like_cols:
        if col in df.columns:
            df[col] = df[col].astype("string")

    # =========================
    #   VARCHAR text fields
    # =========================
    varchar_cols = [
        "homeTeam", "homeConference", "awayTeam", "awayConference",
        "venue", "city", "state", "gameNotes"
    ]

    for col in varchar_cols:
        if col in df.columns:
            df[col] = df[col].astype("string")

    # =========================
    #   INT numeric columns
    # =========================
    int_cols = [
        "season", "attendance",
        "homeSeed", "homePoints",
        "awaySeed", "awayPoints"
    ]

    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    # =========================
    #   BIT fields (0 or 1)
    # =========================
    bit_cols = [
        "startTimeTbd", "neutralSite",
        "conferenceGame", "homeWinner", "awayWinner"
    ]

    for col in bit_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype("boolean")           # converts truthy/falsy
                .astype("Int64")             # SQL BIT expects 0/1
            )

    # =========================
    #   DATETIME2
    # =========================
    datetime_cols = ["startDate"]

    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # =========================
    #   DECIMAL(19,4)
    # =========================
    if "excitement" in df.columns:
        df["excitement"] = (
            pd.to_numeric(df["excitement"], errors="coerce")
            .round(4)
        )

    # =========================
    #   Drop rows missing PK
    # =========================
    if "sourceId" in df.columns:
        df = df[df["sourceId"].notna()]
        
    df = df.drop(['homeConferenceId','homeConference','awayConferenceId','awayConference','homeSeed','awaySeed'],axis=1)
    df = df.drop_duplicates()

    return df

def loadGameInfo(engine, startDate: date, endDate: date):
    games = coerceGameInfoDtypes(cleanGameInfo(getGameInfo(startDate, endDate)))
        
    pks = ["gameId"]
    data_columns = [c for c in games.columns if c not in pks + ["insert_date", "update_date"]]

    upsert_via_staging(
        df              = games,
        table_name      = "GameInfo",
        primary_keys    = pks,
        data_columns    = data_columns,
        engine          = engine,
        schema          = 'dbo',
        dry_run         = False
    )