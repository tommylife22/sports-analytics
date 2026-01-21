"""
Loaders Package
Database loading functions
"""
from .base_loader import load_table_to_database

from .info_loader import (
    load_teams_to_database,
    load_players_to_database,
    load_games_to_database,
    load_all_info_to_database,
)

from .boxscore_loader import (
    load_team_boxscore_to_database,
    load_player_boxscore_to_database,
    load_all_boxscores_to_database,
)

__all__ = [
    'load_table_to_database',
    'load_teams_to_database',
    'load_players_to_database',
    'load_games_to_database',
    'load_all_info_to_database',
    'load_team_boxscore_to_database',
    'load_player_boxscore_to_database',
    'load_all_boxscores_to_database',
]
