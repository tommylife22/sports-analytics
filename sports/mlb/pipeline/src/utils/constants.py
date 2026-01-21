"""
Constants and Configuration
Central location for all table configurations and constants
"""

# Table configurations with primary keys and metadata
TABLE_CONFIGS = {
    'TeamInfo': {
        'primary_keys': ['team_id', 'season'],
        'description': 'MLB team information by season',
        'dependencies': [],
    },
    'GameInfo': {
        'primary_keys': ['game_id'],
        'description': 'Game results and details',
        'dependencies': ['TeamInfo'],
    },
    'PlayerInfo': {
        'primary_keys': ['player_id', 'team_id', 'season'],
        'description': 'Player roster information',
        'dependencies': ['TeamInfo'],
    },
    'TeamBoxscore': {
        'primary_keys': ['game_id', 'team_id'],
        'description': 'Team-level boxscore statistics',
        'dependencies': ['GameInfo', 'TeamInfo'],
    },
    'PlayerBoxscore': {
        'primary_keys': ['game_id', 'player_id'],
        'description': 'Player-level boxscore statistics',
        'dependencies': ['GameInfo', 'PlayerInfo'],
    },
    'BettingOdds': {
        'primary_keys': ['game_id', 'sportsbook', 'bet_type'],
        'description': 'Betting odds from various sportsbooks',
        'dependencies': [],
    },
}

# MLB Stats API constants
MLB_SPORT_ID = 1
ROSTER_TYPES = {
    'ACTIVE': 'active',
    'FORTY_MAN': '40Man',
    'FULL_SEASON': 'fullSeason',
    'ALL_TIME': 'allTime',
}

# Database schema
DEFAULT_SCHEMA = 'dbo'

# Pipeline settings
DEFAULT_ROSTER_TYPE = ROSTER_TYPES['FORTY_MAN']
STAGING_PREFIX = 'Staging_'

# Date formats
DATE_FORMAT = '%Y-%m-%d'
DATETIME_FORMAT_UTC = '%Y-%m-%dT%H:%M:%SZ'
DATETIME_FORMAT = '%Y-%m-%dT%H:%M:%S'