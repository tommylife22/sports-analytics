"""
MLB Stats API Extractor
Handles data extraction from MLB Stats API for teams, players, and games
"""
import statsapi
import time
from requests.exceptions import HTTPError
from ..utils.helpers import parse_datetime, to_string_id
from ..utils.constants import MLB_SPORT_ID, DEFAULT_ROSTER_TYPE


def get_teams_by_season(season):
    """
    Get all MLB teams for a given season

    Args:
        season (int): Season year

    Returns:
        list: List of team dictionaries with string IDs
    """
    teams = statsapi.get('teams', {'sportId': MLB_SPORT_ID, 'season': season})

    teams_list = []
    for team in teams['teams']:
        team_info = {
            'team_id': to_string_id(team['id']),
            'team_name': team['name'],
            'team_abbr': team['abbreviation'],
            'franchise_name': team.get('franchiseName'),
            'club_name': team.get('clubName'),
            'division': team.get('division', {}).get('name'),
            'league': team.get('league', {}).get('name'),
            'venue_name': team.get('venue', {}).get('name'),
            'first_year': team.get('firstYearOfPlay'),
            'season': str(season)
        }
        teams_list.append(team_info)

    return teams_list


def get_roster_by_season(team_id, season, roster_type=DEFAULT_ROSTER_TYPE):
    """
    Get team roster for a given season

    Args:
        team_id (str or int): Team ID
        season (int): Season year
        roster_type (str): Roster type - 'active', '40Man', 'fullSeason', etc.

    Returns:
        list: List of player dictionaries with string IDs
    """
    team_id = to_string_id(team_id)

    roster_data = statsapi.get('team_roster', {
        'teamId': team_id,
        'season': season,
        'rosterType': roster_type
    })

    players_list = []
    for player in roster_data['roster']:
        player_info = {
            'player_id': to_string_id(player['person']['id']),
            'full_name': player['person']['fullName'],
            'jersey_number': player.get('jerseyNumber'),
            'position_code': player['position']['code'],
            'position_name': player['position']['name'],
            'position_type': player['position']['type'],
            'status': player.get('status', {}).get('description'),
            'team_id': team_id,
            'season': str(season),
            'roster_type': roster_type
        }
        players_list.append(player_info)

    return players_list


def get_all_players_by_season(season, roster_type=DEFAULT_ROSTER_TYPE):
    """
    Get all players from all teams for a given season

    Args:
        season (int): Season year
        roster_type (str): Roster type - 'active', '40Man', etc.

    Returns:
        list: List of all player dictionaries
    """
    teams = statsapi.get('teams', {'sportId': MLB_SPORT_ID, 'season': season})

    all_players = []
    for team in teams['teams']:
        team_id = to_string_id(team['id'])
        team_name = team['name']

        print(f"  Fetching roster for {team_name}...")

        try:
            players = get_roster_by_season(team_id, season, roster_type)
            all_players.extend(players)
        except Exception as e:
            print(f"  ⚠ Error fetching {team_name}: {e}")

    return all_players


def get_games_by_season(season, team_id=None, max_retries=3, retry_delay=5):
    """
    Get all games for a season with retry logic

    Args:
        season (int): Season year
        team_id (str or int, optional): Filter by specific team
        max_retries (int): Maximum number of retry attempts
        retry_delay (int): Seconds to wait between retries

    Returns:
        list: List of game dictionaries with string IDs and datetime objects
    """
    # Get season info
    season_info = statsapi.get('season', {'seasonId': season, 'sportId': MLB_SPORT_ID})

    for s in season_info['seasons']:
        if s['seasonId'] == str(season):
            start_date = s['regularSeasonStartDate']
            # Use postSeasonEndDate to include playoffs, fallback to regularSeasonEndDate
            end_date = s.get('postSeasonEndDate', s['regularSeasonEndDate'])
            break

    print(f"  Fetching games from {start_date} to {end_date} (includes playoffs)")

    # Retry logic for API calls
    for attempt in range(max_retries):
        try:
            # Get schedule
            if team_id:
                team_id = to_string_id(team_id)
                games = statsapi.schedule(
                    start_date=start_date,
                    end_date=end_date,
                    team=team_id
                )
            else:
                games = statsapi.schedule(
                    start_date=start_date,
                    end_date=end_date
                )

            # If successful, break out of retry loop
            break

        except HTTPError as e:
            if attempt < max_retries - 1:
                print(f"  ⚠ API error (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"  Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"  ✗ Failed after {max_retries} attempts")
                raise

    games_list = []
    for game in games:
        game_info = {
            'game_id': to_string_id(game['game_id']),
            'game_datetime': parse_datetime(game.get('game_datetime')),
            'away_id': to_string_id(game['away_id']),
            'away_name': game['away_name'],
            'home_id': to_string_id(game['home_id']),
            'home_name': game['home_name'],
            'away_score': game.get('away_score'),
            'home_score': game.get('home_score'),
            'status': game['status'],
            'game_type': game.get('game_type', 'R'),  # R=Regular, P=Playoff, S=Spring, etc.
            'venue_id': to_string_id(game.get('venue_id')),
            'venue_name': game.get('venue_name'),
            'season': str(season)
        }
        games_list.append(game_info)

    print(f"  ✓ Fetched {len(games_list)} games")
    return games_list
