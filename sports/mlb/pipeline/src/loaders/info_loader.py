"""
Info Tables Loader
Handles loading team, player, and game info tables
"""
import sys
import os

# Add project root to path for generic imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from .base_loader import load_table_to_database
from generic.db import get_engine


def load_teams_to_database(teams_df, engine=None, dry_run=False):
    """
    Load teams data to TeamInfo table

    Args:
        teams_df (pd.DataFrame): Teams data
        engine: SQLAlchemy engine (if None, creates new one)
        dry_run (bool): If True, don't actually load data

    Returns:
        dict: Result from upsert operation
    """
    return load_table_to_database(
        df=teams_df,
        table_name='TeamInfo',
        engine=engine,
        dry_run=dry_run
    )


def load_players_to_database(players_df, engine=None, dry_run=False):
    """
    Load players data to PlayerInfo table

    Args:
        players_df (pd.DataFrame): Players data
        engine: SQLAlchemy engine (if None, creates new one)
        dry_run (bool): If True, don't actually load data

    Returns:
        dict: Result from upsert operation
    """
    return load_table_to_database(
        df=players_df,
        table_name='PlayerInfo',
        engine=engine,
        dry_run=dry_run
    )


def load_games_to_database(games_df, engine=None, dry_run=False):
    """
    Load games data to GameInfo table

    Args:
        games_df (pd.DataFrame): Games data
        engine: SQLAlchemy engine (if None, creates new one)
        dry_run (bool): If True, don't actually load data

    Returns:
        dict: Result from upsert operation
    """
    return load_table_to_database(
        df=games_df,
        table_name='GameInfo',
        engine=engine,
        dry_run=dry_run
    )


def load_all_info_to_database(teams_df, players_df, games_df, engine=None, dry_run=False):
    """
    Load all info tables (teams, players, games) to database

    Args:
        teams_df (pd.DataFrame): Teams data
        players_df (pd.DataFrame): Players data
        games_df (pd.DataFrame): Games data
        engine: SQLAlchemy engine (if None, creates new one)
        dry_run (bool): If True, don't actually load data

    Returns:
        dict: Results from all upsert operations
    """
    if engine is None:
        engine = get_engine('MLB')

    print("\n" + "="*50)
    print("LOADING INFO TABLES TO DATABASE")
    print("="*50)

    results = {}

    # Load in order of dependencies: Teams first, then Games and Players
    results['teams'] = load_teams_to_database(teams_df, engine, dry_run)
    results['games'] = load_games_to_database(games_df, engine, dry_run)
    results['players'] = load_players_to_database(players_df, engine, dry_run)

    print("="*50)
    print("✓ ALL INFO TABLES LOADED SUCCESSFULLY")
    print("="*50)

    return results
