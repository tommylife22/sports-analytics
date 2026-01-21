"""
Boxscore Data Loaders
Functions for loading team and player boxscore data to database
"""
from .base_loader import load_table_to_database
from ..utils.constants import DEFAULT_SCHEMA


def load_team_boxscore_to_database(team_boxscore_df, engine=None, schema=DEFAULT_SCHEMA, dry_run=False):
    """
    Load team boxscore data to database

    Args:
        team_boxscore_df (DataFrame): Team boxscore data
        engine: SQLAlchemy engine (if None, creates new one)
        schema (str): Database schema name
        dry_run (bool): If True, don't actually load to database

    Returns:
        dict: Load results
    """
    return load_table_to_database(
        df=team_boxscore_df,
        table_name='TeamBoxscore',
        engine=engine,
        schema=schema,
        dry_run=dry_run
    )


def load_player_boxscore_to_database(player_boxscore_df, engine=None, schema=DEFAULT_SCHEMA, dry_run=False):
    """
    Load player boxscore data to database

    Args:
        player_boxscore_df (DataFrame): Player boxscore data
        engine: SQLAlchemy engine (if None, creates new one)
        schema (str): Database schema name
        dry_run (bool): If True, don't actually load to database

    Returns:
        dict: Load results
    """
    return load_table_to_database(
        df=player_boxscore_df,
        table_name='PlayerBoxscore',
        engine=engine,
        schema=schema,
        dry_run=dry_run
    )


def load_all_boxscores_to_database(team_boxscore_df, player_boxscore_df, engine=None, schema=DEFAULT_SCHEMA, dry_run=False):
    """
    Load all boxscore data (team and player) to database

    Args:
        team_boxscore_df (DataFrame): Team boxscore data
        player_boxscore_df (DataFrame): Player boxscore data
        engine: SQLAlchemy engine (if None, creates new one)
        schema (str): Database schema name
        dry_run (bool): If True, don't actually load to database

    Returns:
        dict: Combined load results
    """
    print("\n" + "="*50)
    print("LOADING BOXSCORE DATA TO DATABASE")
    print("="*50)

    # Load team boxscores
    print("\nLoading TeamBoxscore...")
    team_result = load_team_boxscore_to_database(
        team_boxscore_df=team_boxscore_df,
        engine=engine,
        schema=schema,
        dry_run=dry_run
    )

    # Load player boxscores
    print("\nLoading PlayerBoxscore...")
    player_result = load_player_boxscore_to_database(
        player_boxscore_df=player_boxscore_df,
        engine=engine,
        schema=schema,
        dry_run=dry_run
    )

    # Combined summary
    print("\n" + "="*50)
    print("BOXSCORE LOAD SUMMARY")
    print("="*50)
    print(f"  Team boxscores: {team_result.get('rows_inserted', 0)} inserted, {team_result.get('rows_updated', 0)} updated")
    print(f"  Player boxscores: {player_result.get('rows_inserted', 0)} inserted, {player_result.get('rows_updated', 0)} updated")
    print("="*50)

    return {
        'team': team_result,
        'player': player_result
    }
