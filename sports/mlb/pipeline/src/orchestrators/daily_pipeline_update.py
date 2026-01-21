"""
Daily Pipeline Update
Complete incremental update for all MLB data (games, boxscores, odds)
Designed for scheduled runs (Azure Functions, cron, etc.)
"""
import sys
import os
from datetime import datetime, timedelta

# Add project root to path for generic imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
from ..extractors.mlb_stats import get_games_by_season
from ..orchestrators.boxscore_pipeline import build_boxscore_database
from ..extractors.odds_scraper import scrape_mlb_odds_range
from ..loaders.odds_loader import load_scraped_odds_to_database
from ..loaders.info_loader import load_games_to_database
from generic.db import get_engine


def get_current_season():
    """
    Get current MLB season year

    Returns:
        int: Current season year
    """
    now = datetime.now()
    # MLB season year matches calendar year it starts in
    # If before March, use previous year
    if now.month < 3:
        return now.year - 1
    return now.year


def get_update_date_range(lookback_days=5):
    """
    Get date range for incremental update

    Args:
        lookback_days (int): Number of days to look back from today

    Returns:
        tuple: (start_date, end_date) as strings in YYYY-MM-DD format
    """
    today = datetime.now().date()
    start_date = today - timedelta(days=lookback_days)
    end_date = today

    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')


def get_recent_games(season, lookback_days=5):
    """
    Get games from recent days for a season

    Args:
        season (int): Season year
        lookback_days (int): Number of days to look back

    Returns:
        DataFrame: Recent games
    """
    from ..cleaners.game_cleaners import clean_games_data

    print(f"\n  Fetching games for {season} season...")

    # Get all games for season
    all_games = get_games_by_season(season)
    games_df = pd.DataFrame(all_games)

    if len(games_df) == 0:
        print("  ⚠ No games found")
        return pd.DataFrame()

    # Clean games
    games_df = clean_games_data(games_df)

    # Filter to recent dates
    start_date, end_date = get_update_date_range(lookback_days)
    games_df['game_date'] = pd.to_datetime(games_df['game_date']).dt.date

    recent_games = games_df[
        (games_df['game_date'] >= pd.to_datetime(start_date).date()) &
        (games_df['game_date'] <= pd.to_datetime(end_date).date())
    ]

    print(f"  ✓ Found {len(recent_games)} recent games ({start_date} to {end_date})")

    return recent_games


def daily_pipeline_update(lookback_days=5, load_to_db=True, dry_run=False):
    """
    Run complete daily pipeline update

    Updates:
    1. Recent games (GameInfo)
    2. Boxscores (TeamBoxscore, PlayerBoxscore)
    3. Betting odds (BettingOdds)

    Args:
        lookback_days (int): Number of days to look back (default: 5)
        load_to_db (bool): Whether to load data to database
        dry_run (bool): If True, preview without actually loading

    Returns:
        dict: Update results including counts and status
    """
    print("\n" + "="*70)
    print("DAILY MLB PIPELINE UPDATE")
    print("="*70)

    start_date, end_date = get_update_date_range(lookback_days)
    season = get_current_season()

    print(f"Season:            {season}")
    print(f"Update date range: {start_date} to {end_date}")
    print(f"Lookback days:     {lookback_days}")
    print(f"Load to DB:        {load_to_db}")
    print(f"Dry run:           {dry_run}")
    print("="*70)

    results = {
        'success': True,
        'season': season,
        'start_date': start_date,
        'end_date': end_date,
        'games': None,
        'boxscores': None,
        'odds': None,
        'errors': []
    }

    engine = get_engine('MLB') if load_to_db else None

    # =========================================================================
    # STEP 1: UPDATE GAMES
    # =========================================================================
    try:
        print("\n" + "="*70)
        print("STEP 1: UPDATING RECENT GAMES")
        print("="*70)

        recent_games = get_recent_games(season, lookback_days)

        if len(recent_games) > 0 and load_to_db:
            game_result = load_games_to_database(
                games_df=recent_games,
                engine=engine,
                dry_run=dry_run
            )
            results['games'] = {
                'count': len(recent_games),
                'inserted': game_result.get('rows_inserted', 0),
                'updated': game_result.get('rows_updated', 0)
            }
            print(f"\n  ✓ Games: {len(recent_games)} records")
            print(f"    Inserted: {game_result.get('rows_inserted', 0)}")
            print(f"    Updated:  {game_result.get('rows_updated', 0)}")
        else:
            results['games'] = {
                'count': len(recent_games),
                'inserted': 0,
                'updated': 0
            }
            print(f"\n  ℹ Games: {len(recent_games)} records (not loaded)")

    except Exception as e:
        print(f"\n  ⚠ Error updating games: {e}")
        results['errors'].append(f"Games: {e}")
        results['success'] = False

    # =========================================================================
    # STEP 2: UPDATE BOXSCORES
    # =========================================================================
    try:
        print("\n" + "="*70)
        print("STEP 2: UPDATING RECENT BOXSCORES")
        print("="*70)

        if len(recent_games) > 0:
            # Get list of recent game IDs
            recent_game_ids = recent_games['game_id'].tolist()
            print(f"\n  Processing boxscores for {len(recent_game_ids)} games...")

            # Use existing boxscore pipeline but filtered to recent games
            from ..extractors.boxscore import get_boxscore
            from ..cleaners.boxscore_cleaners import clean_team_boxscore_data, clean_player_boxscore_data
            from ..loaders.boxscore_loader import load_all_boxscores_to_database
            from tqdm import tqdm

            all_team_boxscores = []
            all_player_boxscores = []

            # Only fetch boxscores for completed games
            completed_games = recent_games[recent_games['status'].str.contains('Final|Completed', na=False)]

            for _, game in tqdm(completed_games.iterrows(),
                               total=len(completed_games),
                               desc="  Fetching boxscores",
                               unit="game"):
                try:
                    team_boxscores, player_boxscores = get_boxscore(game['game_id'])
                    all_team_boxscores.extend(team_boxscores)
                    all_player_boxscores.extend(player_boxscores)
                except Exception as e:
                    tqdm.write(f"  ⚠ Error fetching boxscore for game {game['game_id']}: {e}")
                    continue

            # Convert to DataFrames and clean
            team_boxscore_df = pd.DataFrame(all_team_boxscores)
            player_boxscore_df = pd.DataFrame(all_player_boxscores)

            if len(team_boxscore_df) > 0:
                team_boxscore_df = clean_team_boxscore_data(team_boxscore_df)
            if len(player_boxscore_df) > 0:
                player_boxscore_df = clean_player_boxscore_data(player_boxscore_df)

            print(f"\n  ✓ Fetched {len(team_boxscore_df)} team boxscores")
            print(f"  ✓ Fetched {len(player_boxscore_df)} player boxscores")

            # Load to database
            if load_to_db and (len(team_boxscore_df) > 0 or len(player_boxscore_df) > 0):
                boxscore_result = load_all_boxscores_to_database(
                    team_boxscore_df=team_boxscore_df,
                    player_boxscore_df=player_boxscore_df,
                    engine=engine,
                    dry_run=dry_run
                )
                results['boxscores'] = {
                    'team_count': len(team_boxscore_df),
                    'player_count': len(player_boxscore_df),
                    'team_inserted': boxscore_result['team'].get('rows_inserted', 0),
                    'team_updated': boxscore_result['team'].get('rows_updated', 0),
                    'player_inserted': boxscore_result['player'].get('rows_inserted', 0),
                    'player_updated': boxscore_result['player'].get('rows_updated', 0)
                }
            else:
                results['boxscores'] = {
                    'team_count': len(team_boxscore_df),
                    'player_count': len(player_boxscore_df),
                    'team_inserted': 0,
                    'team_updated': 0,
                    'player_inserted': 0,
                    'player_updated': 0
                }
        else:
            print("\n  ℹ No recent games to process")
            results['boxscores'] = {
                'team_count': 0,
                'player_count': 0
            }

    except Exception as e:
        print(f"\n  ⚠ Error updating boxscores: {e}")
        import traceback
        traceback.print_exc()
        results['errors'].append(f"Boxscores: {e}")
        results['success'] = False

    # =========================================================================
    # STEP 3: UPDATE ODDS
    # =========================================================================
    try:
        print("\n" + "="*70)
        print("STEP 3: UPDATING RECENT ODDS")
        print("="*70)

        odds_results = scrape_mlb_odds_range(start_date, end_date)

        moneyline_count = len(odds_results.get('moneyline', []))
        spread_count = len(odds_results.get('spread', []))
        total_count = len(odds_results.get('total', []))

        if load_to_db and (moneyline_count > 0 or spread_count > 0 or total_count > 0):
            odds_load_result = load_scraped_odds_to_database(
                odds_results=odds_results,
                engine=engine,
                dry_run=dry_run
            )
            results['odds'] = {
                'moneyline_count': moneyline_count,
                'spread_count': spread_count,
                'total_count': total_count,
                'inserted': odds_load_result.get('rows_inserted', 0),
                'updated': odds_load_result.get('rows_updated', 0)
            }
        else:
            results['odds'] = {
                'moneyline_count': moneyline_count,
                'spread_count': spread_count,
                'total_count': total_count,
                'inserted': 0,
                'updated': 0
            }
            print(f"\n  ℹ Odds: {moneyline_count + spread_count + total_count} records (not loaded)")

    except Exception as e:
        print(f"\n  ⚠ Error updating odds: {e}")
        import traceback
        traceback.print_exc()
        results['errors'].append(f"Odds: {e}")
        results['success'] = False

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("DAILY UPDATE COMPLETE")
    print("="*70)
    print(f"Season:       {season}")
    print(f"Date range:   {start_date} to {end_date}")
    print(f"\nGames:")
    if results['games']:
        print(f"  Total:    {results['games']['count']}")
        print(f"  Inserted: {results['games']['inserted']}")
        print(f"  Updated:  {results['games']['updated']}")

    print(f"\nBoxscores:")
    if results['boxscores']:
        print(f"  Team records:    {results['boxscores']['team_count']}")
        print(f"    Inserted: {results['boxscores'].get('team_inserted', 0)}")
        print(f"    Updated:  {results['boxscores'].get('team_updated', 0)}")
        print(f"  Player records:  {results['boxscores']['player_count']}")
        print(f"    Inserted: {results['boxscores'].get('player_inserted', 0)}")
        print(f"    Updated:  {results['boxscores'].get('player_updated', 0)}")

    print(f"\nOdds:")
    if results['odds']:
        print(f"  Total records: {results['odds']['moneyline_count'] + results['odds']['spread_count'] + results['odds']['total_count']}")
        print(f"  Inserted: {results['odds']['inserted']}")
        print(f"  Updated:  {results['odds']['updated']}")

    if results['errors']:
        print(f"\n⚠ Errors encountered: {len(results['errors'])}")
        for error in results['errors']:
            print(f"  - {error}")
        results['success'] = False
    else:
        print(f"\n✓ All updates completed successfully")

    print("="*70)

    return results


def daily_update_with_retry(lookback_days=5, max_retries=3, retry_delay=60):
    """
    Run daily update with retry logic (useful for Azure Functions)

    Args:
        lookback_days (int): Number of days to look back
        max_retries (int): Maximum number of retry attempts
        retry_delay (int): Seconds to wait between retries

    Returns:
        dict: Update results
    """
    import time

    for attempt in range(max_retries):
        print(f"\n{'='*70}")
        print(f"ATTEMPT {attempt + 1} OF {max_retries}")
        print(f"{'='*70}")

        result = daily_pipeline_update(lookback_days=lookback_days)

        if result['success']:
            return result

        if attempt < max_retries - 1:
            print(f"\n⚠ Update failed. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)

    return result


def get_update_summary_message(result):
    """
    Generate a human-readable summary message for the update

    Args:
        result (dict): Result from daily_pipeline_update

    Returns:
        str: Summary message
    """
    if result['success']:
        msg = f"✓ Daily pipeline update successful\n"
        msg += f"  Season: {result['season']}\n"
        msg += f"  Date range: {result['start_date']} to {result['end_date']}\n\n"

        if result['games']:
            msg += f"  Games: {result['games']['count']} "
            msg += f"({result['games']['inserted']} inserted, {result['games']['updated']} updated)\n"

        if result['boxscores']:
            msg += f"  Team boxscores: {result['boxscores']['team_count']} "
            msg += f"({result['boxscores'].get('team_inserted', 0)} inserted, {result['boxscores'].get('team_updated', 0)} updated)\n"
            msg += f"  Player boxscores: {result['boxscores']['player_count']} "
            msg += f"({result['boxscores'].get('player_inserted', 0)} inserted, {result['boxscores'].get('player_updated', 0)} updated)\n"

        if result['odds']:
            total_odds = result['odds']['moneyline_count'] + result['odds']['spread_count'] + result['odds']['total_count']
            msg += f"  Odds: {total_odds} "
            msg += f"({result['odds']['inserted']} inserted, {result['odds']['updated']} updated)\n"

        return msg
    else:
        msg = f"✗ Daily pipeline update failed\n"
        msg += f"  Errors: {len(result.get('errors', []))}\n"
        for error in result.get('errors', []):
            msg += f"    - {error}\n"
        return msg
