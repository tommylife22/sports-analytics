"""
MLB Boxscore Extractor
Handles extraction of team and player boxscore data from MLB Stats API
"""
import statsapi
from tqdm import tqdm
from ..utils.helpers import to_string_id


def get_boxscore(game_id):
    """
    Get both team and player boxscore data for a game in a single API call

    Args:
        game_id (str or int): Game ID

    Returns:
        tuple: (team_boxscores_list, player_boxscores_list)
    """
    game_id = to_string_id(game_id)

    # Single API call for all boxscore data
    boxscore = statsapi.boxscore_data(game_id)

    # Extract team boxscores
    team_boxscores = _extract_team_boxscores(game_id, boxscore)

    # Extract player boxscores
    player_boxscores = _extract_player_boxscores(game_id, boxscore)

    return team_boxscores, player_boxscores


def _extract_team_boxscores(game_id, boxscore):
    """
    Extract team boxscore data from API response

    Args:
        game_id (str): Game ID
        boxscore (dict): Boxscore data from API

    Returns:
        list: List of team boxscore dictionaries
    """
    team_boxscores = []

    # Process both teams (away and home)
    for location in ['away', 'home']:
        team_stats = boxscore[location]

        # Batting stats
        batting = team_stats.get('teamStats', {}).get('batting', {})

        # Pitching stats
        pitching = team_stats.get('teamStats', {}).get('pitching', {})

        team_boxscore = {
            'game_id': game_id,
            'team_id': to_string_id(team_stats['team']['id']),
            'is_home': 1 if location == 'home' else 0,

            # Batting stats
            'runs': batting.get('runs'),
            'hits': batting.get('hits'),
            'doubles': batting.get('doubles'),
            'triples': batting.get('triples'),
            'home_runs': batting.get('homeRuns'),
            'rbi': batting.get('rbi'),
            'walks': batting.get('baseOnBalls'),
            'strikeouts': batting.get('strikeOuts'),
            'stolen_bases': batting.get('stolenBases'),
            'caught_stealing': batting.get('caughtStealing'),
            'left_on_base': batting.get('leftOnBase'),
            'hit_by_pitch': batting.get('hitByPitch'),
            'avg': batting.get('avg'),
            'obp': batting.get('obp'),
            'slg': batting.get('slg'),
            'ops': batting.get('ops'),

            # Pitching stats
            'earned_runs': pitching.get('earnedRuns'),
            'hits_allowed': pitching.get('hits'),
            'home_runs_allowed': pitching.get('homeRuns'),
            'walks_allowed': pitching.get('baseOnBalls'),
            'strikeouts_pitched': pitching.get('strikeOuts'),
            'pitches_thrown': pitching.get('numberOfPitches'),
            'strikes': pitching.get('strikes'),
            'era': pitching.get('era'),
        }

        team_boxscores.append(team_boxscore)

    return team_boxscores


def _extract_player_boxscores(game_id, boxscore):
    """
    Extract player boxscore data from API response

    Args:
        game_id (str): Game ID
        boxscore (dict): Boxscore data from API

    Returns:
        list: List of player boxscore dictionaries
    """
    player_boxscores = []

    # Process both teams (away and home)
    for location in ['away', 'home']:
        team_stats = boxscore[location]
        team_id = to_string_id(team_stats['team']['id'])
        is_home = 1 if location == 'home' else 0

        # Get all players for this team
        players = team_stats.get('players', {})
        if not isinstance(players, dict):
            # Skip games with incomplete data
            continue

        # Process each player
        for player_key, player_data in players.items():
            # Extract player ID from key (e.g., 'ID669257' -> '669257')
            if player_key.startswith('ID'):
                player_id = player_key[2:]  # Remove 'ID' prefix
            else:
                player_id = player_key

            player_id = to_string_id(player_id)

            # Get position info
            position_info = player_data.get('position', {})
            position = position_info.get('abbreviation') if isinstance(position_info, dict) else None
            batting_order = player_data.get('battingOrder')

            # Get stats
            stats = player_data.get('stats', {})
            batting_stats = stats.get('batting', {})
            pitching_stats = stats.get('pitching', {})

            # Only include players who actually played (have batting or pitching stats)
            if not batting_stats and not pitching_stats:
                continue

            player_boxscore = {
                'game_id': game_id,
                'player_id': player_id,
                'team_id': team_id,
                'is_home': is_home,
                'position': position,
                'batting_order': batting_order,

                # Batting stats (will be None if empty dict)
                'at_bats': batting_stats.get('atBats'),
                'runs': batting_stats.get('runs'),
                'hits': batting_stats.get('hits'),
                'doubles': batting_stats.get('doubles'),
                'triples': batting_stats.get('triples'),
                'home_runs': batting_stats.get('homeRuns'),
                'rbi': batting_stats.get('rbi'),
                'walks': batting_stats.get('baseOnBalls'),
                'strikeouts': batting_stats.get('strikeOuts'),
                'stolen_bases': batting_stats.get('stolenBases'),
                'caught_stealing': batting_stats.get('caughtStealing'),
                'hit_by_pitch': batting_stats.get('hitByPitch'),
                'avg': batting_stats.get('avg'),
                'obp': batting_stats.get('obp'),
                'slg': batting_stats.get('slg'),
                'ops': batting_stats.get('ops'),

                # Pitching stats (will be None if empty dict)
                'innings_pitched': pitching_stats.get('inningsPitched'),
                'hits_allowed': pitching_stats.get('hits'),
                'runs_allowed': pitching_stats.get('runs'),
                'earned_runs': pitching_stats.get('earnedRuns'),
                'walks_allowed': pitching_stats.get('baseOnBalls'),
                'strikeouts_pitched': pitching_stats.get('strikeOuts'),
                'home_runs_allowed': pitching_stats.get('homeRuns'),
                'pitches_thrown': pitching_stats.get('numberOfPitches'),
                'strikes': pitching_stats.get('strikes'),
                'era': pitching_stats.get('era'),
                'win': pitching_stats.get('wins'),
                'loss': pitching_stats.get('losses'),
                'save': pitching_stats.get('saves'),
                'blown_save': pitching_stats.get('blownSaves'),
                'hold': pitching_stats.get('holds'),
            }

            player_boxscores.append(player_boxscore)

    return player_boxscores


# Legacy functions for backward compatibility (deprecated - use get_boxscore instead)
def get_team_boxscore(game_id):
    """
    Get team-level boxscore statistics for a game

    DEPRECATED: Use get_boxscore() instead for better performance (single API call)

    Args:
        game_id (str or int): Game ID

    Returns:
        list: List of team boxscore dictionaries (one for home, one for away)
    """
    team_boxscores, _ = get_boxscore(game_id)
    return team_boxscores


def get_player_boxscore(game_id):
    """
    Get player-level boxscore statistics for a game

    DEPRECATED: Use get_boxscore() instead for better performance (single API call)

    Args:
        game_id (str or int): Game ID

    Returns:
        list: List of player boxscore dictionaries
    """
    _, player_boxscores = get_boxscore(game_id)
    return player_boxscores


def get_all_boxscores_by_season(season, games_df=None):
    """
    Get all team and player boxscores for a season

    Args:
        season (int): Season year
        games_df (DataFrame, optional): Pre-loaded games dataframe to iterate through

    Returns:
        tuple: (team_boxscores_list, player_boxscores_list)
    """
    # If no games dataframe provided, need to get games first
    if games_df is None:
        from .mlb_stats import get_games_by_season
        import pandas as pd
        games = get_games_by_season(season)
        games_df = pd.DataFrame(games)

    all_team_boxscores = []
    all_player_boxscores = []

    # Filter to only completed games (all variations of Final/Completed)
    completed_statuses = ['Final', 'Final: Tied', 'Completed Early',
                         'Completed Early: Rain', 'Final: Tie, decision by tiebreaker']
    completed_games = games_df[games_df['status'].isin(completed_statuses)]

    # Progress bar for boxscore extraction
    for _, game in tqdm(completed_games.iterrows(),
                        total=len(completed_games),
                        desc=f"  Fetching {season} boxscores",
                        unit="game"):
        game_id = game['game_id']

        try:
            # Single API call gets both team and player boxscores
            team_boxscores, player_boxscores = get_boxscore(game_id)
            all_team_boxscores.extend(team_boxscores)
            all_player_boxscores.extend(player_boxscores)

        except Exception as e:
            tqdm.write(f"  ⚠ Error fetching boxscore for game {game_id}: {e}")
            continue

    print(f"  ✓ Fetched {len(all_team_boxscores)} team boxscores")
    print(f"  ✓ Fetched {len(all_player_boxscores)} player boxscores")

    return all_team_boxscores, all_player_boxscores
