"""
Betting Odds Pipeline Orchestrator
Coordinates scraping and loading of MLB betting odds
"""
import sys
import os
from datetime import datetime, timedelta

# Add project root to path for generic imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
from ..extractors.odds_scraper import scrape_mlb_odds, scrape_mlb_odds_range
from ..loaders.odds_loader import load_scraped_odds_to_database
from generic.db import get_engine


def build_odds_database(start_date, end_date, load_to_db=True, dry_run=False):
    """
    Build betting odds database for a date range

    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        load_to_db (bool): Whether to load data to database
        dry_run (bool): If True, preview without actually loading

    Returns:
        dict: Dictionary with DataFrames for each bet type
    """
    print("\n" + "="*70)
    print("MLB BETTING ODDS PIPELINE")
    print("="*70)
    print(f"Date range: {start_date} to {end_date}")
    print(f"Load to DB: {load_to_db}")
    print(f"Dry run:    {dry_run}")
    print("="*70)

    # Scrape odds
    print("\n" + "="*70)
    print("STEP 1: SCRAPING ODDS")
    print("="*70)

    odds_results = scrape_mlb_odds_range(start_date, end_date)

    # Load to database
    if load_to_db:
        print("\n" + "="*70)
        print("STEP 2: LOADING TO DATABASE")
        print("="*70)

        engine = get_engine('MLB')
        result = load_scraped_odds_to_database(
            odds_results=odds_results,
            engine=engine,
            dry_run=dry_run
        )
    else:
        print("\n⚠ Skipping database load (load_to_db=False)")
        result = None

    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)

    return odds_results, result


def build_odds_database_by_season(season, load_to_db=True, dry_run=False):
    """
    Build betting odds database for an entire season

    Args:
        season (int): Season year
        load_to_db (bool): Whether to load data to database
        dry_run (bool): If True, preview without actually loading

    Returns:
        dict: Dictionary with DataFrames for each bet type
    """
    # Define season date ranges (approximate - adjust as needed)
    # MLB regular season typically runs from late March/early April through September/early October
    season_dates = {
        2024: ('2024-03-20', '2024-10-01'),
        2023: ('2023-03-30', '2023-10-01'),
        2022: ('2022-04-07', '2022-10-05'),
        2021: ('2021-04-01', '2021-10-03'),
        2020: ('2020-07-23', '2020-09-27'),  # Shortened COVID season
        2019: ('2019-03-20', '2019-09-29'),
        2018: ('2018-03-29', '2018-10-01'),
    }

    if season not in season_dates:
        # Default to April 1 - October 1 for unlisted seasons
        start_date = f"{season}-04-01"
        end_date = f"{season}-10-01"
        print(f"⚠ Using default dates for {season} season: {start_date} to {end_date}")
    else:
        start_date, end_date = season_dates[season]

    print(f"\nScraping {season} season odds from {start_date} to {end_date}")

    return build_odds_database(start_date, end_date, load_to_db, dry_run)


def backfill_historical_odds(start_year, end_year, load_to_db=True, dry_run=False):
    """
    Backfill historical odds data for multiple seasons

    Args:
        start_year (int): First season to scrape
        end_year (int): Last season to scrape (inclusive)
        load_to_db (bool): Whether to load data to database
        dry_run (bool): If True, preview without actually loading

    Returns:
        dict: Summary of results by season
    """
    print("\n" + "="*70)
    print("HISTORICAL ODDS BACKFILL")
    print("="*70)
    print(f"Seasons: {start_year} to {end_year}")
    print(f"Load to DB: {load_to_db}")
    print(f"Dry run:    {dry_run}")
    print("="*70)

    all_results = {}

    for season in range(start_year, end_year + 1):
        print(f"\n{'='*70}")
        print(f"PROCESSING SEASON {season}")
        print(f"{'='*70}")

        try:
            odds_results, load_result = build_odds_database_by_season(
                season=season,
                load_to_db=load_to_db,
                dry_run=dry_run
            )

            all_results[season] = {
                'success': True,
                'moneyline_records': len(odds_results.get('moneyline', [])),
                'spread_records': len(odds_results.get('spread', [])),
                'total_records': len(odds_results.get('total', [])),
                'load_result': load_result
            }

        except Exception as e:
            print(f"\n⚠ Error processing season {season}: {e}")
            all_results[season] = {
                'success': False,
                'error': str(e)
            }
            continue

    # Final summary
    print("\n" + "="*70)
    print("BACKFILL COMPLETE - SUMMARY")
    print("="*70)

    for season, result in all_results.items():
        print(f"\n{season}:")
        if result['success']:
            print(f"  ✓ Moneyline: {result['moneyline_records']} records")
            print(f"  ✓ Spread:    {result['spread_records']} records")
            print(f"  ✓ Total:     {result['total_records']} records")
            if result['load_result']:
                print(f"  Inserted:    {result['load_result'].get('rows_inserted', 0)}")
                print(f"  Updated:     {result['load_result'].get('rows_updated', 0)}")
        else:
            print(f"  ✗ Failed: {result['error']}")

    print("\n" + "="*70)

    return all_results


def find_available_date_range():
    """
    Helper function to find what date range has available odds data
    Tests a few dates to determine availability

    Returns:
        tuple: (earliest_available_date, latest_available_date) or (None, None)
    """
    print("Testing data availability...")

    test_dates = [
        '2024-06-15',  # Recent
        '2023-06-15',  # Last year
        '2022-06-15',  # Two years ago
        '2021-04-04',  # Three years ago
        '2020-07-24',  # Four years ago (COVID season)
    ]

    available_dates = []

    for date in test_dates:
        try:
            print(f"  Testing {date}...", end=" ")
            results = scrape_mlb_odds(date, bet_types=['moneyline'])
            if len(results['moneyline']) > 0:
                available_dates.append(date)
                print("✓ Data available")
            else:
                print("✗ No data")
        except Exception as e:
            print(f"✗ Error: {e}")

    if available_dates:
        print(f"\n✓ Found available data from {min(available_dates)} to {max(available_dates)}")
        return min(available_dates), max(available_dates)
    else:
        print("\n⚠ No available data found in test dates")
        return None, None
