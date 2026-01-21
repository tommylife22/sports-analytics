"""
Boxscore Data Cleaners
Functions for cleaning team and player boxscore data
"""
import pandas as pd


def clean_team_boxscore_data(team_boxscore_df):
    """
    Clean team boxscore data

    Args:
        team_boxscore_df (DataFrame): Raw team boxscore data

    Returns:
        DataFrame: Cleaned team boxscore data
    """
    df = team_boxscore_df.copy()

    # Ensure proper data types for numeric columns
    int_cols = [
        'is_home', 'runs', 'hits', 'doubles', 'triples', 'home_runs', 'rbi',
        'walks', 'strikeouts', 'stolen_bases', 'caught_stealing', 'left_on_base',
        'hit_by_pitch', 'earned_runs', 'hits_allowed', 'home_runs_allowed',
        'walks_allowed', 'strikeouts_pitched', 'pitches_thrown', 'strikes'
    ]

    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Ensure proper data types for float columns
    float_cols = ['avg', 'obp', 'slg', 'ops', 'era']

    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Sort by game_id and team_id
    df = df.sort_values(['game_id', 'team_id']).reset_index(drop=True)

    return df


def clean_player_boxscore_data(player_boxscore_df):
    """
    Clean player boxscore data

    Args:
        player_boxscore_df (DataFrame): Raw player boxscore data

    Returns:
        DataFrame: Cleaned player boxscore data
    """
    df = player_boxscore_df.copy()

    # Ensure proper data types for numeric columns (batting and pitching combined)
    int_cols = [
        'is_home', 'at_bats', 'runs', 'hits', 'doubles', 'triples', 'home_runs',
        'rbi', 'walks', 'strikeouts', 'stolen_bases', 'caught_stealing',
        'hit_by_pitch', 'hits_allowed', 'runs_allowed', 'earned_runs',
        'walks_allowed', 'strikeouts_pitched', 'home_runs_allowed',
        'pitches_thrown', 'strikes', 'win', 'loss', 'save', 'blown_save', 'hold'
    ]

    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Ensure proper data types for float columns
    float_cols = ['avg', 'obp', 'slg', 'ops', 'innings_pitched', 'era']

    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Check for and handle duplicate primary keys (game_id, player_id)
    pk_columns = ['game_id', 'player_id']
    duplicates = df[df.duplicated(subset=pk_columns, keep=False)]

    if len(duplicates) > 0:
        num_duplicate_pairs = len(duplicates[pk_columns].drop_duplicates())
        print(f"  ⚠ Warning: Found {len(duplicates)} duplicate records ({num_duplicate_pairs} unique game/player pairs)")

        # Keep the first occurrence of each duplicate
        # This handles edge cases like players appearing in both teams' data
        df = df.drop_duplicates(subset=pk_columns, keep='first')
        print(f"  ✓ Removed duplicates, kept first occurrence: {len(df)} records remaining")

    # Sort by game_id and player_id
    df = df.sort_values(['game_id', 'player_id']).reset_index(drop=True)

    return df
