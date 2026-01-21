"""
Season Pipeline Orchestrator
Coordinates extraction, cleaning, and loading for full season data
"""
import pandas as pd
from ..extractors import (
    get_teams_by_season,
    get_all_players_by_season,
    get_games_by_season
)
from ..cleaners import (
    clean_teams_data,
    clean_players_data,
    clean_games_data
)
from ..loaders import load_all_info_to_database


def build_season_database(season, load_to_db=False, engine=None, dry_run=False):
    """
    Build complete database for a season

    Args:
        season (int): Season year
        load_to_db (bool): If True, load data to database
        engine: SQLAlchemy engine (if None and load_to_db=True, creates new one)
        dry_run (bool): If True, don't actually load to database

    Returns:
        tuple: (teams_df, players_df, games_df)
    """
    print(f"\n{'#'*60}")
    print(f"# BUILDING DATABASE FOR {season} SEASON")
    print(f"{'#'*60}\n")

    # 1. Extract Teams
    print("="*50)
    print("EXTRACTING TEAMS")
    print("="*50)
    teams = get_teams_by_season(season)
    teams_df = pd.DataFrame(teams)
    print(f"  ✓ Extracted {len(teams_df)} teams")

    # 2. Extract Players
    print("\n" + "="*50)
    print("EXTRACTING PLAYERS")
    print("="*50)
    players = get_all_players_by_season(season)
    players_df = pd.DataFrame(players)
    print(f"  ✓ Extracted {len(players_df)} players")

    # 3. Extract Games
    print("\n" + "="*50)
    print("EXTRACTING GAMES")
    print("="*50)
    games = get_games_by_season(season)
    games_df = pd.DataFrame(games)
    print(f"  ✓ Extracted {len(games_df)} games")

    # 4. Clean Data
    print("\n" + "="*50)
    print("CLEANING DATA")
    print("="*50)
    teams_df = clean_teams_data(teams_df)
    players_df = clean_players_data(players_df)
    games_df = clean_games_data(games_df)
    print("  ✓ All data cleaned")

    # 5. Summary
    print("\n" + "="*50)
    print("EXTRACTION SUMMARY")
    print("="*50)
    print(f"  Teams: {len(teams_df)}")
    print(f"  Players: {len(players_df)}")
    print(f"  Games: {len(games_df)}")
    print("="*50)

    # 6. Load to database if requested
    if load_to_db:
        load_all_info_to_database(teams_df, players_df, games_df, engine, dry_run)

    return teams_df, players_df, games_df


def build_multi_season_database(seasons, load_to_db=False, engine=None, dry_run=False):
    """
    Build database for multiple seasons

    Args:
        seasons (list): List of season years
        load_to_db (bool): If True, load data to database
        engine: SQLAlchemy engine (if None and load_to_db=True, creates new one)
        dry_run (bool): If True, don't actually load to database

    Returns:
        tuple: (teams_df, players_df, games_df) combined across all seasons
    """
    all_teams = []
    all_players = []
    all_games = []

    for season in seasons:
        teams, players, games = build_season_database(
            season=season,
            load_to_db=False,  # Don't load individually
            engine=engine,
            dry_run=dry_run
        )

        all_teams.append(teams)
        all_players.append(players)
        all_games.append(games)

    # Combine all seasons
    print("\n" + "="*50)
    print("COMBINING ALL SEASONS")
    print("="*50)

    teams_combined = pd.concat(all_teams, ignore_index=True)
    players_combined = pd.concat(all_players, ignore_index=True)
    games_combined = pd.concat(all_games, ignore_index=True)

    print(f"  Seasons processed: {len(seasons)}")
    print(f"  Total teams: {len(teams_combined)}")
    print(f"  Total players: {len(players_combined)}")
    print(f"  Total games: {len(games_combined)}")
    print("="*50)

    # Load combined data if requested
    if load_to_db:
        load_all_info_to_database(teams_combined, players_combined, games_combined, engine, dry_run)

    return teams_combined, players_combined, games_combined
