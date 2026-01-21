from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any, Dict, List

import pandas as pd
import requests

from dotenv import load_dotenv
from .functions import upsert_via_staging

load_dotenv()

DAILYLINE_URL = (
    "https://site.web.api.espn.com/apis/fitt/v3/sports/"
    "basketball/mens-college-basketball/dailyline"
)

HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    ),
}

ESPN_TZ = ZoneInfo("America/New_York")

def today_yyyymmdd() -> str:
    return datetime.now(ESPN_TZ).strftime("%Y%m%d")


def iso_to_local_date_time(iso_str: str):
    """Convert ESPN ISO time to (date_str, time_str) in US/Eastern."""
    if not iso_str:
        return None, None
    dt_utc = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    dt_local = dt_utc.astimezone(ESPN_TZ)
    date_str = dt_local.strftime("%Y%m%d")
    time_str = dt_local.strftime("%I:%M %p").lstrip("0")
    return date_str, time_str

# Fetch ALL dailyLines across all pages

def fetch_all_dailylines() -> List[Dict[str, Any]]:
    params = {
        "region": "us",
        "lang": "en",
        "contentorigin": "espn",
        "groups": 50,   # NCAA Division I
        "limit": 100,
        "page": 1,
    }

    url = DAILYLINE_URL
    games: List[Dict[str, Any]] = []

    while url:
        resp = requests.get(
            url,
            params=params if "?" not in url else None,
            headers=HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        games.extend(data.get("dailyLines", []))

        pagination = data.get("pagination") or {}
        next_url = pagination.get("next")

        if not next_url:
            break

        url = next_url
        params = None

    return games


# -------------------------------------------------------------------
# Build final table
# -------------------------------------------------------------------

def build_bpi_table_for_today() -> pd.DataFrame:
    all_games = fetch_all_dailylines()
    today_str = today_yyyymmdd()

    rows = []

    for g in all_games:
        # date & time
        game_date_iso = g.get("gameDate")
        date_str, time_str = iso_to_local_date_time(game_date_iso)
        if date_str != today_str:
            continue

        # home / away sections
        away = g.get("awayTeam") or {}
        home = g.get("homeTeam") or {}

        away_team_info = away.get("team") or {}
        home_team_info = home.get("team") or {}

        away_name = away_team_info.get("displayName")
        home_name = home_team_info.get("displayName")

        # BPI / powerIndex
        away_bpi = away.get("powerIndex")
        home_bpi = home.get("powerIndex")

        rows.append(
            {
                "date": date_str,
                "home_team": home_name,
                "away_team": away_name,
                "espn_home_win_prob": home_bpi,
                "espn_away_win_prob": away_bpi,
            }
        )

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df.rename(columns={
        'home_team':'homeTeam',
        'away_team':'awayTeam',
        'espn_home_win_prob':'espnHomeWinProbability',
        'espn_away_win_prob':'espnAwayWinProbability'
    },inplace=True)
    return df.sort_values(["homeTeam"]).reset_index(drop=True)


def loadGameESPNTeamWinProbability(engine):

    df = build_bpi_table_for_today()
    
    pks = ['date','homeTeam','awayTeam']
    data_columns = [c for c in df.columns if c not in pks + ["insert_date", "update_date"]]

    upsert_via_staging(
        df              = df,
        table_name      = "GameESPNTeamWinProbability",
        primary_keys    = pks,
        data_columns    = data_columns,
        engine          = engine,
        schema          = 'CBB',
        dry_run         = False
    )