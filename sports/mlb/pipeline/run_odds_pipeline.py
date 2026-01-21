"""
Run Odds Pipeline
Extract, clean, and load MLB betting odds data to database
"""
import argparse
from src.orchestrators.odds_pipeline import (
    build_odds_database,
    build_odds_database_by_season,
    backfill_historical_odds,
    find_available_date_range
)


def main():
    parser = argparse.ArgumentParser(
        description='Run MLB Betting Odds Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Find what historical data is available
  python run_odds_pipeline.py --check-availability

  # Scrape and load a single season
  python run_odds_pipeline.py --season 2024

  # Scrape and load a date range
  python run_odds_pipeline.py --start-date 2024-06-01 --end-date 2024-06-30

  # Backfill all available historical data (2020-2024)
  python run_odds_pipeline.py --backfill --start-year 2020 --end-year 2024

  # Preview without loading to database
  python run_odds_pipeline.py --season 2024 --dry-run

  # Extract only, don't load to database
  python run_odds_pipeline.py --season 2024 --no-load
        """
    )

    # Mode selection
    parser.add_argument('--check-availability', action='store_true',
                        help='Check what historical data is available')
    parser.add_argument('--backfill', action='store_true',
                        help='Backfill multiple seasons')

    # Date arguments
    parser.add_argument('--season', type=int, help='Single season to process')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--start-year', type=int, help='Start year for backfill')
    parser.add_argument('--end-year', type=int, help='End year for backfill')

    # Database arguments
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview without loading to database')
    parser.add_argument('--no-load', action='store_true',
                        help='Extract only, do not load to database')

    args = parser.parse_args()

    # Check availability mode
    if args.check_availability:
        print("\n" + "="*70)
        print("CHECKING DATA AVAILABILITY")
        print("="*70)
        earliest, latest = find_available_date_range()
        if earliest and latest:
            print(f"\n✓ Recommended backfill:")
            print(f"  python run_odds_pipeline.py --backfill --start-year {earliest[:4]} --end-year {latest[:4]}")
        return

    # Determine load_to_db flag
    load_to_db = not args.no_load

    # Backfill mode
    if args.backfill:
        if not args.start_year or not args.end_year:
            parser.error("--backfill requires --start-year and --end-year")

        results = backfill_historical_odds(
            start_year=args.start_year,
            end_year=args.end_year,
            load_to_db=load_to_db,
            dry_run=args.dry_run
        )

    # Single season mode
    elif args.season:
        odds_results, load_result = build_odds_database_by_season(
            season=args.season,
            load_to_db=load_to_db,
            dry_run=args.dry_run
        )

    # Date range mode
    elif args.start_date and args.end_date:
        odds_results, load_result = build_odds_database(
            start_date=args.start_date,
            end_date=args.end_date,
            load_to_db=load_to_db,
            dry_run=args.dry_run
        )

    # No valid arguments
    else:
        parser.print_help()
        return

    print("\n✓ Pipeline completed successfully!")


if __name__ == "__main__":
    main()