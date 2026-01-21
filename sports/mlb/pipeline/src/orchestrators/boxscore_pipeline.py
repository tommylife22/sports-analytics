"""
Boxscore Pipeline Orchestrator
Coordinates extraction, cleaning, and loading for boxscore data
"""
import pandas as pd
from ..extractors import get_all_boxscores_by_season
from ..cleaners import clean_team_boxscore_data, clean_player_boxscore_data
from ..loaders import load_all_boxscores_to_database


def build_boxscore_database(season, games_df=None, load_to_db=False, engine=None, dry_run=False):
    """
    Build boxscore database for a season

    Args:
        season (int): Season year
        games_df (DataFrame, optional): Pre-loaded games dataframe to iterate through
        load_to_db (bool): If True, load data to database
        engine: SQLAlchemy engine (if None and load_to_db=True, creates new one)
        dry_run (bool): If True, don't actually load to database

    Returns:
        tuple: (team_boxscore_df, player_boxscore_df)
    """
    print(f"\n{'#'*60}")
    print(f"# BUILDING BOXSCORE DATA FOR {season} SEASON")
    print(f"{'#'*60}\n")

    # 1. Extract Boxscores
    print("="*50)
    print("EXTRACTING BOXSCORES")
    print("="*50)
    team_boxscores, player_boxscores = get_all_boxscores_by_season(season, games_df)
    team_boxscore_df = pd.DataFrame(team_boxscores)
    player_boxscore_df = pd.DataFrame(player_boxscores)
    print(f"  ✓ Extracted {len(team_boxscore_df)} team boxscores")
    print(f"  ✓ Extracted {len(player_boxscore_df)} player boxscores")

    # 2. Clean Data
    print("\n" + "="*50)
    print("CLEANING DATA")
    print("="*50)
    team_boxscore_df = clean_team_boxscore_data(team_boxscore_df)
    player_boxscore_df = clean_player_boxscore_data(player_boxscore_df)
    print("  ✓ All data cleaned")

    # 3. Summary
    print("\n" + "="*50)
    print("EXTRACTION SUMMARY")
    print("="*50)
    print(f"  Team Boxscores: {len(team_boxscore_df)}")
    print(f"  Player Boxscores: {len(player_boxscore_df)}")
    print("="*50)

    # 4. Load to database if requested
    if load_to_db:
        load_all_boxscores_to_database(team_boxscore_df, player_boxscore_df, engine=engine, dry_run=dry_run)

    return team_boxscore_df, player_boxscore_df


def build_multi_season_boxscore_database(seasons, games_dict=None, load_to_db=False, engine=None, dry_run=False):
    """
    Build boxscore database for multiple seasons

    Args:
        seasons (list): List of season years
        games_dict (dict, optional): Dictionary of {season: games_df} to use for extraction
        load_to_db (bool): If True, load data to database
        engine: SQLAlchemy engine (if None and load_to_db=True, creates new one)
        dry_run (bool): If True, don't actually load to database

    Returns:
        tuple: (team_boxscore_df, player_boxscore_df) combined across all seasons
    """
    all_team_boxscores = []
    all_player_boxscores = []

    for season in seasons:
        # Get games dataframe for this season if provided
        games_df = games_dict.get(season) if games_dict else None

        team_boxscores, player_boxscores = build_boxscore_database(
            season=season,
            games_df=games_df,
            load_to_db=False,  # Don't load individually
            engine=engine,
            dry_run=dry_run
        )

        all_team_boxscores.append(team_boxscores)
        all_player_boxscores.append(player_boxscores)

    # Combine all seasons
    print("\n" + "="*50)
    print("COMBINING ALL SEASONS")
    print("="*50)

    team_boxscore_combined = pd.concat(all_team_boxscores, ignore_index=True)
    player_boxscore_combined = pd.concat(all_player_boxscores, ignore_index=True)

    print(f"  Seasons processed: {len(seasons)}")
    print(f"  Total team boxscores: {len(team_boxscore_combined)}")
    print(f"  Total player boxscores: {len(player_boxscore_combined)}")
    print("="*50)

    # Load combined data if requested
    if load_to_db:
        load_all_boxscores_to_database(team_boxscore_combined, player_boxscore_combined, engine=engine, dry_run=dry_run)

    return team_boxscore_combined, player_boxscore_combined
