"""
Game Data Cleaners
Cleaning and validation for games, boxscores, and related data
"""
import pandas as pd
from .base_cleaners import validate_dataframe, remove_nulls_in_required_columns


def clean_duplicate_games(games_df):
    """
    Remove duplicate game_ids by:
    1. Dropping all rows where status='Postponed'
    2. For remaining duplicates, keep the row with the earlier game_datetime

    Args:
        games_df (pd.DataFrame): DataFrame with games data

    Returns:
        pd.DataFrame: Cleaned DataFrame with unique game_ids
    """
    print(f"  Total games before cleaning: {len(games_df)}")
    print(f"  Duplicate game_ids: {games_df['game_id'].duplicated().sum()}")

    # Step 1: Remove postponed games
    games_df = games_df[games_df['status'] != 'Postponed'].copy()
    print(f"  After removing postponed: {len(games_df)}")

    # Step 2: For remaining duplicates, keep earliest game_datetime
    games_df = games_df.sort_values('game_datetime')
    games_df = games_df.drop_duplicates(subset=['game_id'], keep='first')

    print(f"  After deduplication: {len(games_df)}")
    print(f"  Remaining duplicates: {games_df['game_id'].duplicated().sum()}")

    return games_df


def clean_teams_data(teams_df):
    """
    Clean and validate teams data

    Args:
        teams_df (pd.DataFrame): Raw teams data

    Returns:
        pd.DataFrame: Cleaned teams data
    """
    required = ['team_id', 'season', 'team_name', 'team_abbr']
    validate_dataframe(teams_df, required, "Teams DataFrame")

    # Remove rows with null required columns
    teams_df = remove_nulls_in_required_columns(teams_df, required, "Teams")

    return teams_df


def clean_players_data(players_df):
    """
    Clean and validate players data

    Args:
        players_df (pd.DataFrame): Raw players data

    Returns:
        pd.DataFrame: Cleaned players data
    """
    required = ['player_id', 'team_id', 'season', 'full_name']
    validate_dataframe(players_df, required, "Players DataFrame")

    # Remove rows with null required columns
    players_df = remove_nulls_in_required_columns(players_df, required, "Players")

    return players_df


def clean_games_data(games_df):
    """
    Clean and validate games data (includes deduplication)

    Args:
        games_df (pd.DataFrame): Raw games data

    Returns:
        pd.DataFrame: Cleaned games data
    """
    required = ['game_id', 'season', 'game_datetime', 'home_id', 'away_id']
    validate_dataframe(games_df, required, "Games DataFrame")

    # Remove rows with null required columns
    games_df = remove_nulls_in_required_columns(games_df, required, "Games")

    # Remove duplicates (postponed games, etc.)
    games_df = clean_duplicate_games(games_df)

    return games_df
