"""
Game Data Cleaners
Cleaning and validation for game information and boxscores
"""
import pandas as pd
from .base_cleaners import validate_dataframe, remove_nulls_in_required_columns


def clean_game_data(games_df):
    """
    Clean and validate game data

    Args:
        games_df (pd.DataFrame): Raw game data from extractor

    Returns:
        pd.DataFrame: Cleaned game data
    """
    print("\n--- Cleaning Game Data ---")
    original_len = len(games_df)
    
    # Select relevant columns
    cols_to_keep = ['id', 'sourceId', 'season', 'seasonType', 'startDate',
                    'homeTeamId', 'awayTeamId', 'homePoints', 'awayPoints',
                    'status', 'neutralSite', 'conferenceGame']
    
    games_df = games_df[[col for col in cols_to_keep if col in games_df.columns]].copy()
    
    # Rename columns to match database schema
    games_df.rename(columns={
        'id': 'game_id',
        'sourceId': 'source_id',
        'seasonType': 'season_type',
        'startDate': 'game_date',
        'homeTeamId': 'home_team_id',
        'awayTeamId': 'away_team_id',
        'homePoints': 'home_points',
        'awayPoints': 'away_points',
        'neutralSite': 'neutral_site',
        'conferenceGame': 'conference_game'
    }, inplace=True)
    
    # Remove rows with null required fields
    required = ['game_id', 'season', 'home_team_id', 'away_team_id']
    games_df = remove_nulls_in_required_columns(games_df, required, "Games")
    
    # Convert numeric columns
    numeric_cols = ['game_id', 'source_id', 'season', 'home_team_id', 'away_team_id', 
                    'home_points', 'away_points']
    for col in numeric_cols:
        if col in games_df.columns:
            games_df[col] = pd.to_numeric(games_df[col], errors='coerce').astype('Int64')
    
    # Convert date
    games_df['game_date'] = pd.to_datetime(games_df['game_date'], errors='coerce')
    
    # Convert boolean columns
    bool_cols = ['neutral_site', 'conference_game']
    for col in bool_cols:
        if col in games_df.columns:
            games_df[col] = games_df[col].astype(bool)
    
    # Fill missing status
    games_df['status'].fillna('Unknown', inplace=True)
    games_df['season_type'].fillna('Regular', inplace=True)
    
    print(f"  Original rows: {original_len}")
    print(f"  Cleaned rows: {len(games_df)}")
    print(f"  Columns: {list(games_df.columns)}")
    
    return games_df


def clean_team_boxscore_data(boxscores_df):
    """
    Clean and validate team boxscore data

    Args:
        boxscores_df (pd.DataFrame): Raw team boxscore data from extractor

    Returns:
        pd.DataFrame: Cleaned team boxscore data
    """
    print("\n--- Cleaning Team Boxscore Data ---")
    original_len = len(boxscores_df)
    
    # This is a placeholder - actual cleaning depends on API response structure
    # Select commonly available boxscore columns
    if len(boxscores_df) == 0:
        print("  No data to clean")
        return boxscores_df
    
    print(f"  Original rows: {original_len}")
    print(f"  Cleaned rows: {len(boxscores_df)}")
    
    return boxscores_df


def clean_player_boxscore_data(boxscores_df):
    """
    Clean and validate player boxscore data

    Args:
        boxscores_df (pd.DataFrame): Raw player boxscore data from extractor

    Returns:
        pd.DataFrame: Cleaned player boxscore data
    """
    print("\n--- Cleaning Player Boxscore Data ---")
    original_len = len(boxscores_df)
    
    # This is a placeholder - actual cleaning depends on API response structure
    # Select commonly available boxscore columns
    if len(boxscores_df) == 0:
        print("  No data to clean")
        return boxscores_df
    
    print(f"  Original rows: {original_len}")
    print(f"  Cleaned rows: {len(boxscores_df)}")
    
    return boxscores_df
