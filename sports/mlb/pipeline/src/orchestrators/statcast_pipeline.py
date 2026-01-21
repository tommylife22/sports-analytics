"""
Statcast Data Pipeline Orchestrator
Coordinates extraction, cleaning, and loading of Statcast data
"""
import sys
import os

# Add project root to path for generic imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ..extractors.statcast import get_statcast_data, get_statcast_bulk, get_statcast_season
from ..cleaners.statcast_cleaners import clean_statcast_pitches
from ..loaders.statcast_loader import load_statcast_pitches, check_existing_data
from generic.db import get_engine


def run_statcast_daily_pipeline(date, dry_run=False, skip_if_exists=True):
    """
    Run Statcast pipeline for a single day

    Args:
        date (str): Date in 'YYYY-MM-DD' format
        dry_run (bool): If True, don't load to database
        skip_if_exists (bool): Skip if data already exists

    Returns:
        dict: Pipeline results
    """
    print(f"\n{'='*60}")
    print(f"STATCAST DAILY PIPELINE: {date}")
    print(f"{'='*60}\n")

    engine = get_engine('MLB')

    # Check if data already exists
    if skip_if_exists:
        print("Checking for existing data...")
        existing = check_existing_data(date, date, engine)

        if existing['exists']:
            print(f"  ✓ Data already exists for {date}")
            print(f"    Pitches: {existing['pitch_count']:,}")
            print(f"    Games: {existing['game_count']}")
            print("  Skipping (use skip_if_exists=False to force reload)\n")
            return {
                'status': 'skipped',
                'reason': 'data_exists',
                'existing_pitches': existing['pitch_count']
            }

    # Extract
    print("1. EXTRACTING data from Baseball Savant...")
    raw_data = get_statcast_data(start_date=date, end_date=date, verbose=True)

    if len(raw_data) == 0:
        print("  ⚠ No data found for this date\n")
        return {
            'status': 'completed',
            'pitches_loaded': 0,
            'reason': 'no_games'
        }

    # Clean
    print("\n2. CLEANING data...")
    cleaned_data = clean_statcast_pitches(raw_data)
    print(f"  ✓ Cleaned {len(cleaned_data):,} pitches")

    # Load
    print("\n3. LOADING to database...")
    if dry_run:
        print("  DRY RUN - Not loading to database")
        result = {'inserted': 0, 'updated': 0}
    else:
        result = load_statcast_pitches(cleaned_data, engine=engine)

    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE")
    print(f"  Date: {date}")
    print(f"  Pitches: {len(cleaned_data):,}")
    print(f"  Inserted: {result.get('inserted', 0):,}")
    print(f"  Updated: {result.get('updated', 0):,}")
    print(f"{'='*60}\n")

    return {
        'status': 'completed',
        'date': date,
        'pitches_extracted': len(raw_data),
        'pitches_cleaned': len(cleaned_data),
        'pitches_inserted': result.get('inserted', 0),
        'pitches_updated': result.get('updated', 0)
    }


def run_statcast_daterange_pipeline(start_date, end_date, chunk_days=7, dry_run=False, skip_if_exists=True):
    """
    Run Statcast pipeline for a date range

    Args:
        start_date (str): Start date 'YYYY-MM-DD'
        end_date (str): End date 'YYYY-MM-DD'
        chunk_days (int): Days per chunk when fetching from API
        dry_run (bool): If True, don't load to database
        skip_if_exists (bool): Skip if data already exists

    Returns:
        dict: Pipeline results
    """
    print(f"\n{'='*60}")
    print(f"STATCAST DATE RANGE PIPELINE")
    print(f"  {start_date} to {end_date}")
    print(f"{'='*60}\n")

    engine = get_engine('MLB')

    # Check if data already exists
    if skip_if_exists:
        print("Checking for existing data...")
        existing = check_existing_data(start_date, end_date, engine)

        if existing['exists']:
            print(f"  ✓ Data already exists for date range")
            print(f"    Date range: {existing['min_date']} to {existing['max_date']}")
            print(f"    Pitches: {existing['pitch_count']:,}")
            print(f"    Games: {existing['game_count']}")
            print("  Skipping (use skip_if_exists=False to force reload)\n")
            return {
                'status': 'skipped',
                'reason': 'data_exists',
                'existing_pitches': existing['pitch_count']
            }

    # Extract
    print("1. EXTRACTING data from Baseball Savant...")
    raw_data = get_statcast_bulk(
        start_date=start_date,
        end_date=end_date,
        chunk_days=chunk_days,
        verbose=True
    )

    if len(raw_data) == 0:
        print("  ⚠ No data found for this date range\n")
        return {
            'status': 'completed',
            'pitches_loaded': 0,
            'reason': 'no_data'
        }

    # Clean
    print("\n2. CLEANING data...")
    cleaned_data = clean_statcast_pitches(raw_data)
    print(f"  ✓ Cleaned {len(cleaned_data):,} pitches")

    # Load
    print("\n3. LOADING to database...")
    if dry_run:
        print("  DRY RUN - Not loading to database")
        result = {'inserted': 0, 'updated': 0}
    else:
        result = load_statcast_pitches(cleaned_data, engine=engine, batch_size=10000)

    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE")
    print(f"  Date range: {start_date} to {end_date}")
    print(f"  Pitches: {len(cleaned_data):,}")
    print(f"  Games: {cleaned_data['game_pk'].nunique()}")
    print(f"  Inserted: {result.get('inserted', 0):,}")
    print(f"  Updated: {result.get('updated', 0):,}")
    print(f"{'='*60}\n")

    return {
        'status': 'completed',
        'start_date': start_date,
        'end_date': end_date,
        'pitches_extracted': len(raw_data),
        'pitches_cleaned': len(cleaned_data),
        'pitches_inserted': result.get('inserted', 0),
        'pitches_updated': result.get('updated', 0),
        'games': cleaned_data['game_pk'].nunique()
    }


def run_statcast_season_pipeline(season, dry_run=False, skip_if_exists=False):
    """
    Run Statcast pipeline for an entire season

    Args:
        season (int): Season year (e.g., 2024)
        dry_run (bool): If True, don't load to database
        skip_if_exists (bool): Skip if data already exists

    Returns:
        dict: Pipeline results

    Example:
        # Load 2023 season
        result = run_statcast_season_pipeline(2023)
    """
    print(f"\n{'='*60}")
    print(f"STATCAST SEASON PIPELINE: {season}")
    print(f"{'='*60}\n")

    engine = get_engine('MLB')

    # Extract
    print("1. EXTRACTING season data from Baseball Savant...")
    raw_data = get_statcast_season(season=season, verbose=True)

    if len(raw_data) == 0:
        print(f"  ⚠ No data found for {season} season\n")
        return {
            'status': 'completed',
            'pitches_loaded': 0,
            'reason': 'no_data'
        }

    # Clean
    print("\n2. CLEANING data...")
    cleaned_data = clean_statcast_pitches(raw_data)
    print(f"  ✓ Cleaned {len(cleaned_data):,} pitches")

    # Load
    print("\n3. LOADING to database...")
    if dry_run:
        print("  DRY RUN - Not loading to database")
        result = {'inserted': 0, 'updated': 0}
    else:
        result = load_statcast_pitches(cleaned_data, engine=engine, batch_size=10000)

    print(f"\n{'='*60}")
    print(f"SEASON PIPELINE COMPLETE: {season}")
    print(f"  Pitches: {len(cleaned_data):,}")
    print(f"  Games: {cleaned_data['game_pk'].nunique()}")
    print(f"  Inserted: {result.get('inserted', 0):,}")
    print(f"  Updated: {result.get('updated', 0):,}")
    print(f"{'='*60}\n")

    return {
        'status': 'completed',
        'season': season,
        'pitches_extracted': len(raw_data),
        'pitches_cleaned': len(cleaned_data),
        'pitches_inserted': result.get('inserted', 0),
        'pitches_updated': result.get('updated', 0),
        'games': cleaned_data['game_pk'].nunique()
    }
