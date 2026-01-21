import pandas as pd
import numpy as np
from sqlalchemy import text
import sys
import os

# Add project root to path for generic imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from generic.db import retry_on_connection_error,get_engine


# =============================================================================
# 1. LOAD GAME-LEVEL DATA
# =============================================================================
def load_game_data(engine):
    """
    Load baseball game data from vw_MoneylineModelData view.
    Returns one row per game with home/away team stats, betting odds, and context.
    """
    sql = "SELECT * FROM vw_MoneylineModelData"

    def _load():
        df = pd.read_sql(text(sql), engine)

        if "gameDate" in df.columns:
            df["gameDate"] = pd.to_datetime(df["gameDate"])

        if "gameDateTime" in df.columns:
            df["gameDateTime"] = pd.to_datetime(df["gameDateTime"])
            if df["gameDateTime"].dt.tz is None:
                df["gameDateTime"] = df["gameDateTime"].dt.tz_localize("UTC")

        return df

    return retry_on_connection_error(_load)


# =============================================================================
# 2. TRANSFORM TO TEAM-LEVEL FOR ROLLING CALCULATIONS
# =============================================================================
def build_team_game_long(df: pd.DataFrame, team_stats: list) -> pd.DataFrame:
    """
    Converts game-level dataframe to team-level (one row per team per game).
    Sorted by date for proper rolling average calculation.

    Args:
        df: Game-level dataframe
        team_stats: List of stat names to include (e.g., ['Hits', 'OPS'])
                   These should match column suffixes (homeHits -> 'Hits')
    """
    required = {"gameId", "season", "gameDate", "homeTeamId", "awayTeamId"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    base_cols = ["gameId", "season", "gameDate"]

    # Filter to stats that exist in both home and away
    valid_stats = [
        s for s in team_stats
        if f"home{s}" in df.columns and f"away{s}" in df.columns
    ]

    if not valid_stats:
        print(f"Warning: No valid team stats found. Available columns: {[c for c in df.columns if c.startswith('home')][:10]}")

    # HOME rows
    home_cols = base_cols + ["homeTeamId"] + [f"home{s}" for s in valid_stats]
    home = df[home_cols].copy()
    home = home.rename(columns={f"home{s}": s for s in valid_stats})
    home = home.rename(columns={"homeTeamId": "teamId"})
    home["side"] = "home"

    # AWAY rows
    away_cols = base_cols + ["awayTeamId"] + [f"away{s}" for s in valid_stats]
    away = df[away_cols].copy()
    away = away.rename(columns={f"away{s}": s for s in valid_stats})
    away = away.rename(columns={"awayTeamId": "teamId"})
    away["side"] = "away"

    # Combine and sort by date
    out = pd.concat([home, away], ignore_index=True)
    out = out.sort_values(["teamId", "season", "gameDate"]).reset_index(drop=True)

    return out


def add_team_rolling_features(team_df: pd.DataFrame, stats: list, windows: list) -> pd.DataFrame:
    """
    Add rolling averages for specified stats. Uses shift(1) to avoid data leakage.

    Args:
        team_df: Team-level dataframe (sorted by date)
        stats: List of stat column names
        windows: List of window sizes (e.g., [3, 5, 10])
    """
    df = team_df.copy()

    for stat in stats:
        if stat not in df.columns:
            continue

        for w in windows:
            df[f"{stat}_roll{w}"] = (
                df.groupby(["teamId", "season"])[stat]
                .transform(lambda s: s.shift(1).rolling(w, min_periods=w).mean())
            )

    return df


def build_pitcher_long(df: pd.DataFrame, pitcher_stats: list) -> pd.DataFrame:
    """
    Converts game-level dataframe to pitcher-level for rolling calculations.

    Args:
        df: Game-level dataframe
        pitcher_stats: List of pitcher stat names (e.g., ['SP_MaxVelo', 'SP_xwOBA'])
    """
    if "homeStartingPitcherId" not in df.columns:
        return pd.DataFrame()

    # Filter to stats that exist
    valid_stats = [
        s for s in pitcher_stats
        if f"home{s}" in df.columns and f"away{s}" in df.columns
    ]

    if not valid_stats:
        return pd.DataFrame()

    base_cols = ["gameId", "season", "gameDate"]

    # HOME pitchers
    home = df[base_cols + ["homeStartingPitcherId"] + [f"home{s}" for s in valid_stats]].copy()
    home = home.rename(columns={"homeStartingPitcherId": "pitcherId"})
    home = home.rename(columns={f"home{s}": s for s in valid_stats})
    home["side"] = "home"

    # AWAY pitchers
    away = df[base_cols + ["awayStartingPitcherId"] + [f"away{s}" for s in valid_stats]].copy()
    away = away.rename(columns={"awayStartingPitcherId": "pitcherId"})
    away = away.rename(columns={f"away{s}": s for s in valid_stats})
    away["side"] = "away"

    out = pd.concat([home, away], ignore_index=True)
    out = out.sort_values(["pitcherId", "season", "gameDate"]).reset_index(drop=True)

    return out


def add_pitcher_rolling_features(pitcher_df: pd.DataFrame, stats: list, windows: list) -> pd.DataFrame:
    """
    Add rolling averages for pitcher stats. Uses shift(1) to avoid data leakage.
    """
    if pitcher_df.empty:
        return pitcher_df

    df = pitcher_df.copy()

    for stat in stats:
        if stat not in df.columns:
            continue

        for w in windows:
            df[f"{stat}_roll{w}"] = (
                df.groupby(["pitcherId", "season"])[stat]
                .transform(lambda s: s.shift(1).rolling(w, min_periods=w).mean())
            )

    return df


# =============================================================================
# 3. MERGE ROLLING FEATURES BACK + CREATE DIFFERENTIAL FEATURES
# =============================================================================
def merge_team_rolling_to_game(game_df: pd.DataFrame, team_df: pd.DataFrame,
                                stats: list, windows: list) -> pd.DataFrame:
    """
    Merge team rolling features back to game level and create differential features.
    This is where the key modeling features are created (home - away differences).
    """
    # Identify rolling columns
    roll_cols = [f"{s}_roll{w}" for s in stats for w in windows if f"{s}_roll{w}" in team_df.columns]

    if not roll_cols:
        return game_df.copy()

    # Split by side and merge
    home_features = team_df[team_df["side"] == "home"][["gameId"] + roll_cols].copy()
    home_features = home_features.rename(columns={c: f"home_{c}" for c in roll_cols})

    away_features = team_df[team_df["side"] == "away"][["gameId"] + roll_cols].copy()
    away_features = away_features.rename(columns={c: f"away_{c}" for c in roll_cols})

    df = game_df.copy()
    df = df.merge(home_features, on="gameId", how="left")
    df = df.merge(away_features, on="gameId", how="left")

    # Create differential features (home - away)
    for stat in stats:
        for w in windows:
            home_col = f"home_{stat}_roll{w}"
            away_col = f"away_{stat}_roll{w}"

            if home_col in df.columns and away_col in df.columns:
                df[f"{stat.lower()}_diff_l{w}"] = df[home_col] - df[away_col]

    # Drop the individual home/away rolling columns (keep only diffs)
    cols_to_drop = [c for c in df.columns if "_roll" in c]
    df = df.drop(columns=cols_to_drop, errors="ignore")

    return df


def merge_pitcher_rolling_to_game(game_df: pd.DataFrame, pitcher_df: pd.DataFrame,
                                   stats: list, windows: list) -> pd.DataFrame:
    """
    Merge pitcher rolling features back to game level and create differential features.
    """
    if pitcher_df.empty:
        return game_df

    roll_cols = [f"{s}_roll{w}" for s in stats for w in windows if f"{s}_roll{w}" in pitcher_df.columns]

    if not roll_cols:
        return game_df

    # Home pitchers
    home_pitcher = pitcher_df[pitcher_df["side"] == "home"][["gameId", "pitcherId"] + roll_cols].copy()
    home_pitcher = home_pitcher.rename(columns={c: f"homeSP_{c}" for c in roll_cols})

    # Away pitchers
    away_pitcher = pitcher_df[pitcher_df["side"] == "away"][["gameId", "pitcherId"] + roll_cols].copy()
    away_pitcher = away_pitcher.rename(columns={c: f"awaySP_{c}" for c in roll_cols})

    df = game_df.copy()
    df = df.merge(home_pitcher.drop(columns=["pitcherId"]), on="gameId", how="left")
    df = df.merge(away_pitcher.drop(columns=["pitcherId"]), on="gameId", how="left")

    # Create differential features for pitcher stats
    for stat in stats:
        for w in windows:
            home_col = f"homeSP_{stat}_roll{w}"
            away_col = f"awaySP_{stat}_roll{w}"

            if home_col in df.columns and away_col in df.columns:
                df[f"sp_{stat.lower()}_diff_l{w}"] = df[home_col] - df[away_col]

    # Drop individual home/away pitcher rolling columns
    cols_to_drop = [c for c in df.columns if "SP_" in c and "_roll" in c]
    df = df.drop(columns=cols_to_drop, errors="ignore")

    return df


# =============================================================================
# 4. MARKET FEATURES (Betting Data Derived Features)
# =============================================================================
def add_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features derived from betting market data.
    """
    df = df.copy()

    # Implied probabilities from moneyline
    if "homeMoneyline" in df.columns and "awayMoneyline" in df.columns:
        df["homeImpliedProb"] = _moneyline_to_probability(df["homeMoneyline"])
        df["awayImpliedProb"] = _moneyline_to_probability(df["awayMoneyline"])

        # How lopsided is the market? (0 = toss-up, 0.5 = heavy favorite)
        df["favoriteDegree"] = np.abs(df["homeImpliedProb"] - 0.5)

        # Implied probability difference
        df["impliedProbDiff"] = df["homeImpliedProb"] - df["awayImpliedProb"]

    # Over/under line features
    if "overUnder" in df.columns:
        # Line percentile within season (high/low scoring game expected)
        df["line_percentile"] = df.groupby("season")["overUnder"].rank(pct=True)

        # Deviation from season average
        df["line_vs_avg"] = df.groupby("season")["overUnder"].transform(
            lambda x: x - x.mean()
        )

    # Spread features
    if "homeSpread" in df.columns:
        # Absolute spread (how close is the game expected to be)
        df["spreadMagnitude"] = np.abs(df["homeSpread"])

    # Rest-based context features
    if "homeRestDays" in df.columns and "awayRestDays" in df.columns:
        df["restDiff"] = df["homeRestDays"] - df["awayRestDays"]
        df["both_rested"] = ((df["homeRestDays"] >= 2) & (df["awayRestDays"] >= 2)).astype(int)
        df["either_b2b"] = ((df["homeRestDays"] <= 1) | (df["awayRestDays"] <= 1)).astype(int)

    return df


def _moneyline_to_probability(moneyline: pd.Series) -> pd.Series:
    """Convert American moneyline odds to implied probability."""
    def convert(ml):
        if pd.isna(ml):
            return np.nan
        if ml > 0:
            return 100 / (ml + 100)
        else:
            return abs(ml) / (abs(ml) + 100)

    return moneyline.apply(convert)


# =============================================================================
# 5. TARGET VARIABLES
# =============================================================================
def add_all_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all possible target flags: homeWin, homeCover, wentOver.
    Does not drop any rows - targets are NA where not applicable.
    """
    df = df.copy()

    scored = df["homeRuns"].notna() & df["awayRuns"].notna()

    # --- homeWin ---
    df["homeWin"] = pd.NA
    tie = scored & (df["homeRuns"] == df["awayRuns"])
    df.loc[scored & ~tie, "homeWin"] = (
        df.loc[scored & ~tie, "homeRuns"] > df.loc[scored & ~tie, "awayRuns"]
    ).astype("Int64")

    # --- homeCover (spread) ---
    df["homeCover"] = pd.NA
    if "homeSpread" in df.columns:
        has_spread = scored & df["homeSpread"].notna()
        push = has_spread & ((df["homeRuns"] + df["homeSpread"]) == df["awayRuns"])
        df.loc[has_spread & ~push, "homeCover"] = (
            (df.loc[has_spread & ~push, "homeRuns"] + df.loc[has_spread & ~push, "homeSpread"])
            > df.loc[has_spread & ~push, "awayRuns"]
        ).astype("Int64")

    # --- wentOver ---
    df["wentOver"] = pd.NA
    if "overUnder" in df.columns:
        has_ou = scored & df["overUnder"].notna()
        total = df["homeRuns"] + df["awayRuns"]
        push = has_ou & (total == df["overUnder"])
        df.loc[has_ou & ~push, "wentOver"] = (
            total.loc[has_ou & ~push] > df.loc[has_ou & ~push, "overUnder"]
        ).astype("Int64")

    return df


# =============================================================================
# 6. BUILD FINAL MODEL DATAFRAME
# =============================================================================
def build_model_dataframe(
    game_df: pd.DataFrame,
    team_stats: list,
    windows: list,
    pitcher_stats: list = None,
) -> pd.DataFrame:
    """
    Full feature engineering pipeline.

    Args:
        game_df: Raw game-level data from database
        team_stats: Team stats to create rolling diffs for (e.g., ['Hits', 'OPS'])
        windows: Rolling window sizes (e.g., [3])
        pitcher_stats: Optional pitcher stats for rolling diffs (default: None)

    Returns:
        DataFrame with: IDs, diff features, market features, targets
    """
    pitcher_stats = pitcher_stats or []

    print(f"Starting feature engineering pipeline...")
    print(f"Initial dataset: {game_df.shape[0]} games")

    # Build team-level data and add rolling features
    print(f"Building team rolling features for {len(team_stats)} stats, windows={windows}...")
    team_long = build_team_game_long(game_df, team_stats)
    team_long = add_team_rolling_features(team_long, team_stats, windows)

    # Merge back to game level with differential features
    print("Creating differential features...")
    df = merge_team_rolling_to_game(game_df, team_long, team_stats, windows)

    # Optional: pitcher rolling features
    if pitcher_stats:
        print(f"Building pitcher rolling features for {len(pitcher_stats)} stats...")
        pitcher_long = build_pitcher_long(game_df, pitcher_stats)
        pitcher_long = add_pitcher_rolling_features(pitcher_long, pitcher_stats, windows)
        df = merge_pitcher_rolling_to_game(df, pitcher_long, pitcher_stats, windows)

    # Add market features
    print("Adding market features...")
    df = add_market_features(df)

    # Add all target variables
    print("Adding target variables...")
    df = add_all_targets(df)

    # Select final columns
    print("Selecting final columns...")
    df = _select_final_columns(df)

    print(f"Feature engineering complete!")
    print(f"Final dataset: {df.shape[0]} games, {df.shape[1]} features")

    return df


def _select_final_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the columns we want in the final model dataframe:
    - Identifiers: gameId, season, gameDate, gameDateTime, teams, venue
    - Diff features: *_diff_l* columns (rolling differentials)
    - Market features: implied probabilities, spreads, totals
    - Targets: homeWin, homeCover, wentOver
    """
    id_cols = [
        "gameId", "season", "gameDate", "gameDateTime", "gameType",
        "homeTeamId", "homeTeamName", "awayTeamId", "awayTeamName",
        "venueId", "venueName",
    ]

    market_cols = [
        "homeMoneyline", "awayMoneyline", "homeSpread", "awaySpread",
        "homeSpreadOdds", "awaySpreadOdds",
        "overUnder", "overOdds", "underOdds",
        "homeImpliedProb", "awayImpliedProb", "favoriteDegree", "impliedProbDiff",
        "line_percentile", "line_vs_avg", "spreadMagnitude",
    ]

    target_cols = ["homeWin", "homeCover", "wentOver"]

    # Diff features (dynamically identified from rolling calculations)
    diff_cols = sorted([c for c in df.columns if "_diff_l" in c])

    # Combine and filter to existing columns
    keep_cols = id_cols + diff_cols + market_cols + target_cols
    final_cols = [c for c in keep_cols if c in df.columns]

    return df[final_cols]


# =============================================================================
# EXAMPLE USAGE
# =============================================================================
if __name__ == "__main__":

    # Team batting stats (match view column naming: homeHits -> 'Hits')
    TEAM_BATTING_STATS = [
        "Runs", "Hits", "Doubles", "Triples", "HomeRuns",
        "RBI", "Walks", "Strikeouts", "StolenBases", "LOB", "HBP",
        "BA", "OBP", "SLG", "OPS",
    ]

    # Team pitching stats (match view column naming: homeEarnedRuns -> 'EarnedRuns')
    TEAM_PITCHING_STATS = [
        "EarnedRuns", "HitsAllowed", "HRAllowed",
        "WalksAllowed", "StrikeoutsPitched", "ERA",
    ]

    # Combined team stats
    TEAM_STATS = TEAM_BATTING_STATS + TEAM_PITCHING_STATS

    # No starting pitcher data in current view
    PITCHER_STATS = []

    WINDOWS = [3]

    # Load and build
    engine = get_engine('MLB')
    game_df = load_game_data(engine)
    print(f"Loaded {len(game_df)} games from database")

    model_df = build_model_dataframe(
        game_df=game_df,
        team_stats=TEAM_STATS,
        windows=WINDOWS,
    )

    print(f"\nFinal dataset shape: {model_df.shape}")
    print(f"\nColumns in final dataframe:")
    for col in model_df.columns:
        print(f"  {col}")

    print(f"\nTarget distributions:")
    for target in ["homeWin", "homeCover", "wentOver"]:
        if target in model_df.columns:
            print(f"\n{target}:")
            print(model_df[target].value_counts(dropna=False))
