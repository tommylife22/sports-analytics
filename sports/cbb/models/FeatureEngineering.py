import pandas as pd
from sqlalchemy import text
import sys
import os

# Add project root to path for generic imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from generic.db import retry_on_connection_error,get_engine

# -------------------------------------------------
# LOAD BASE DATA
# -------------------------------------------------
def load_home_game_data(engine, seasonStart=None, seasonEnd=None):
    sql = "SELECT * FROM CBB.vw_FullModel"
    filters, params = [], {}

    if seasonStart:
        filters.append("season >= :seasonStart")
        params["seasonStart"] = seasonStart

    if seasonEnd:
        filters.append("season <= :seasonEnd")
        params["seasonEnd"] = seasonEnd

    if filters:
        sql += " WHERE " + " AND ".join(filters)

    def _load():
        df = pd.read_sql(text(sql), engine, params=params)
        df["startDate"] = df["startDate"].dt.tz_localize("UTC")
        return df

    return retry_on_connection_error(_load)


# -------------------------------------------------
# TEAM-GAME LONG FORMAT
# -------------------------------------------------
def build_team_game_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts one-row-per-game dataframe into one-row-per-team-per-game (long).
    Schema: vw_FullModel (homeTeamId / awayTeamId).
    """

    required = {
        "gameId", "season", "startDate",
        "homeTeamId", "awayTeamId"
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    base_cols = ["gameId", "season", "startDate"]

    # identify stat columns
    home_cols = [
        c for c in df.columns
        if c.startswith("home") and c not in {"homeTeamId"}
    ]
    away_cols = [
        c for c in df.columns
        if c.startswith("away") and c not in {"awayTeamId"}
    ]

    shared_stats = sorted(
        set(c.replace("home", "") for c in home_cols)
        & set(c.replace("away", "") for c in away_cols)
    )

    # HOME rows
    home = df[
        base_cols + ["homeTeamId"] + [f"home{s}" for s in shared_stats]
    ].copy()

    home = home.rename(
        columns={f"home{s}": s for s in shared_stats}
    )
    home = home.rename(columns={"homeTeamId": "teamId"})
    home["side"] = "home"

    # AWAY rows
    away = df[
        base_cols + ["awayTeamId"] + [f"away{s}" for s in shared_stats]
    ].copy()

    away = away.rename(
        columns={f"away{s}": s for s in shared_stats}
    )
    away = away.rename(columns={"awayTeamId": "teamId"})
    away["side"] = "away"

    out = pd.concat([home, away], ignore_index=True)

    # hard guarantees
    assert "teamId" in out.columns, "teamId not created"
    assert out["teamId"].notna().all(), "Null teamId values detected"

    return out


# -------------------------------------------------
# ROLLING STATS
# -------------------------------------------------
def add_team_rolling_features_long(team_df, stats, windows):
    team_df = team_df.sort_values(["teamId", "season", "startDate"])

    for stat in stats:
        for w in windows:
            team_df[f"{stat}_l{w}"] = (
                team_df
                .groupby(["teamId", "season"])[stat]
                .transform(lambda s: s.shift(1).rolling(w, min_periods=w).mean())
            )

    return team_df


# -------------------------------------------------
# FATIGUE FEATURES
# -------------------------------------------------
def add_team_fatigue_long(df):
    df = df.copy()
    df["gameDate"] = (df["startDate"] - pd.Timedelta(hours=5)).dt.normalize()

    df = df.sort_values(["teamId", "season", "gameDate"])

    df["RestDays"] = df.groupby(["teamId", "season"])["gameDate"].diff().dt.days
    df["GamesPlayed"] = df.groupby(["teamId", "season"]).cumcount() + 1

    return df


# -------------------------------------------------
# MERGE BACK TO GAME LEVEL
# -------------------------------------------------
def merge_back_to_game_level(game_df, team_df, stats, windows):

    # ----------------------------------
    # Base game-level columns (single source of truth)
    # ----------------------------------
    base_cols = [
        "gameId",
        "season",
        "startDate",
        "homeTeamId",
        "awayTeamId",
        "homeSpread",
        "overUnder",
        "homePoints",
        "awayPoints",
    ]

    base = game_df[base_cols].copy()

    # ----------------------------------
    # Team-level engineered features ONLY
    # ----------------------------------
    team_feature_cols = [
        c for c in team_df.columns
        if (
            "_l" in c
            or c in {"RestDays", "GamesPlayed", "3in5", "4in7"}
        )
    ]

    def side_df(side):
        return (
            team_df.loc[team_df["side"] == side, ["gameId"] + team_feature_cols]
            .rename(columns=lambda c: f"{side}{c}" if c != "gameId" else c)
        )

    # ----------------------------------
    # Merge home & away features
    # ----------------------------------
    df = (
        base
        .merge(side_df("home"), on="gameId", how="left")
        .merge(side_df("away"), on="gameId", how="left")
    )

    # ----------------------------------
    # Diff features
    # ----------------------------------
    for stat in stats:
        for w in windows:
            df[f"{stat.lower()}_diff_l{w}"] = (
                df[f"home{stat}_l{w}"] - df[f"away{stat}_l{w}"]
            )
            
    # ----------------------------------
    # Sum features
    # ----------------------------------         
    for stat in stats:
        for w in windows:
            df[f"{stat.lower()}_sum_l{w}"] = (
                df[f"home{stat}_l{w}"] + df[f"away{stat}_l{w}"]
            )

    # ----------------------------------
    # Fatigue diffs
    # ----------------------------------
    df["restDiff"] = df["homeRestDays"] - df["awayRestDays"]

    return df

# -------------------------------------------------
# INTERACTION FEATURES FOR TOTALS
# -------------------------------------------------
def add_totals_interaction_features(df: pd.DataFrame, windows) -> pd.DataFrame:
    """
    Add interaction features specifically useful for over/under predictions.
    Call this after merge_back_to_game_level() if target == "wentOver"
    """
    df = df.copy()

    for w in windows:
        # Pace × Offensive efficiency (expected points)
        if f"pace_sum_l{w}" in df.columns and f"rating_sum_l{w}" in df.columns:
            df[f"pace_x_rating_l{w}"] = (
                df[f"pace_sum_l{w}"] * df[f"rating_sum_l{w}"] / 100
            )

        # Pace × True Shooting (scoring efficiency in fast games)
        if f"pace_sum_l{w}" in df.columns and f"trueshooting_sum_l{w}" in df.columns:
            df[f"pace_x_ts_l{w}"] = (
                df[f"pace_sum_l{w}"] * df[f"trueshooting_sum_l{w}"] / 100
            )

        # Offensive rebounding × Shooting (second-chance points)
        if f"orbpct_sum_l{w}" in df.columns and f"efg_sum_l{w}" in df.columns:
            df[f"second_chance_l{w}"] = (
                df[f"orbpct_sum_l{w}"] * df[f"efg_sum_l{w}"] / 100
            )

        # Paint points × Pace (inside game tempo)
        if f"paintpoints_sum_l{w}" in df.columns and f"pace_sum_l{w}" in df.columns:
            df[f"paint_tempo_l{w}"] = (
                df[f"paintpoints_sum_l{w}"] * df[f"pace_sum_l{w}"] / 100
            )

        # Free throws × Pace (foul impact on scoring)
        if f"ftr_sum_l{w}" in df.columns and f"pace_sum_l{w}" in df.columns:
            df[f"ft_tempo_l{w}"] = (
                df[f"ftr_sum_l{w}"] * df[f"pace_sum_l{w}"] / 100
            )

    # Line-based features
    if "overUnder" in df.columns:
        # Line percentile within season
        df["line_percentile"] = (
            df.groupby("season")["overUnder"]
            .rank(pct=True)
        )

    # Situational features
    if "conferenceGame" in df.columns:
        df["conferenceGame"] = df["conferenceGame"].astype(int)

    if "neutralSite" in df.columns:
        df["neutralSite"] = df["neutralSite"].astype(int)

    # Rest-based features
    if "homeRestDays" in df.columns and "awayRestDays" in df.columns:
        df["both_rested"] = (
            (df["homeRestDays"] >= 3) & (df["awayRestDays"] >= 3)
        ).astype(int)

        df["either_b2b"] = (
            (df["homeRestDays"] <= 1) | (df["awayRestDays"] <= 1)
        ).astype(int)

        df["rest_sum"] = df["homeRestDays"] + df["awayRestDays"]

    return df

# -------------------------------------------------
# TARGET CREATION (CLEANLY ISOLATED)
# -------------------------------------------------
def add_spread_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # games with final scores
    scored = df["homePoints"].notna() & df["awayPoints"].notna()

    # identify pushes (only where scored)
    push = scored & (
        (df["homePoints"] + df["homeSpread"]) == df["awayPoints"]
    )

    # drop pushes only
    df = df.loc[~push].copy()

    # initialize target as NA
    df["homeCover"] = pd.NA

    # compute cover only where scores exist
    df.loc[scored & ~push, "homeCover"] = (
        (df.loc[scored & ~push, "homePoints"] + df.loc[scored & ~push, "homeSpread"])
        > df.loc[scored & ~push, "awayPoints"]
    ).astype("Int64")

    return df

def add_over_under_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates binary target: 1 = OVER, 0 = UNDER
    Also creates predicted_total and actual_total columns
    """
    df = df.copy()
    
    # games with final scores
    scored = df["homePoints"].notna() & df["awayPoints"].notna()
    
    # actual total points
    df["actualTotal"] = df["homePoints"] + df["awayPoints"]
    
    # identify pushes (actual total equals the line)
    push = scored & (df["actualTotal"] == df["overUnder"])
    
    # drop pushes
    df = df.loc[~push].copy()
    
    # initialize target as NA
    df["wentOver"] = pd.NA
    
    # compute over/under only where scores exist
    df.loc[scored & ~push, "wentOver"] = (
        df.loc[scored & ~push, "actualTotal"] > df.loc[scored & ~push, "overUnder"]
    ).astype("Int64")
    
    return df

# -------------------------------------------------
# FULL PIPELINE
# -------------------------------------------------
def build_model_dataframe(game_df, rolling_stats, windows, target):
    team_long = build_team_game_long(game_df)
    team_long = add_team_rolling_features_long(team_long, rolling_stats, windows)
    team_long = add_team_fatigue_long(team_long)

    df = merge_back_to_game_level(game_df, team_long, rolling_stats, windows)

    if target == "homeCover":
        df = add_spread_target(df)
    elif target == "wentOver":
        df = add_totals_interaction_features(df, windows)
        df = add_over_under_target(df)

    return df