"""
Run Daily Update
Incremental update for all MLB data (games, boxscores, odds)
"""
import argparse
from src.orchestrators.daily_pipeline_update import (
    daily_pipeline_update,
    daily_update_with_retry,
    get_update_summary_message
)


def main():
    parser = argparse.ArgumentParser(
        description='Run Daily MLB Pipeline Update',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run daily update (last 5 days by default)
  python run_daily_update.py

  # Update last 7 days
  python run_daily_update.py --lookback-days 7

  # Update last 3 days
  python run_daily_update.py --lookback-days 3

  # Preview without loading to database
  python run_daily_update.py --dry-run

  # Update with retry logic (good for unreliable connections)
  python run_daily_update.py --with-retry

What gets updated:
  - GameInfo (recent games)
  - TeamBoxscore (team stats for recent games)
  - PlayerBoxscore (player stats for recent games)
  - BettingOdds (odds for recent dates)

Why 5 days lookback?
  - Catches late-posted odds
  - Updates odds that may have changed
  - Catches any missed games
  - Updates game statuses (postponed, completed, etc.)
        """
    )

    # Update arguments
    parser.add_argument(
        '--lookback-days',
        type=int,
        default=5,
        help='Number of days to look back (default: 5)'
    )

    # Retry arguments
    parser.add_argument(
        '--with-retry',
        action='store_true',
        help='Enable retry logic (3 attempts with 60s delay)'
    )

    # Database arguments
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview without loading to database'
    )
    parser.add_argument(
        '--no-load',
        action='store_true',
        help='Extract only, do not load to database'
    )

    args = parser.parse_args()

    # Determine load_to_db flag
    load_to_db = not args.no_load

    # Run update
    if args.with_retry:
        result = daily_update_with_retry(
            lookback_days=args.lookback_days
        )
    else:
        result = daily_pipeline_update(
            lookback_days=args.lookback_days,
            load_to_db=load_to_db,
            dry_run=args.dry_run
        )

    # Print summary
    print("\n" + "="*70)
    print("UPDATE SUMMARY")
    print("="*70)
    print(get_update_summary_message(result))
    print("="*70)

    # Exit code based on success
    if result['success']:
        print("\n✓ Daily update completed successfully!")
        exit(0)
    else:
        print("\n✗ Daily update failed!")
        exit(1)


if __name__ == "__main__":
    main()
