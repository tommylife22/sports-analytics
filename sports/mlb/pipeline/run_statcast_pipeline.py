"""
Statcast Pipeline Runner
Execute Statcast data pipeline for different scenarios

Usage:
    # Load single day
    python run_statcast_pipeline.py --date 2024-07-15

    # Load date range
    python run_statcast_pipeline.py --start 2024-07-01 --end 2024-07-31

    # Load full season
    python run_statcast_pipeline.py --season 2024

    # Load multiple seasons (backfill)
    python run_statcast_pipeline.py --seasons 2020 2021 2022 2023 2024

    # Dry run (don't load to DB)
    python run_statcast_pipeline.py --season 2024 --dry-run
"""
import argparse
from datetime import datetime, timedelta
from src.orchestrators.statcast_pipeline import (
    run_statcast_daily_pipeline,
    run_statcast_daterange_pipeline,
    run_statcast_season_pipeline
)


def main():
    parser = argparse.ArgumentParser(description='Run Statcast Data Pipeline')

    # Date/season options (mutually exclusive)
    date_group = parser.add_mutually_exclusive_group(required=True)
    date_group.add_argument('--date', help='Single date (YYYY-MM-DD)')
    date_group.add_argument('--start', help='Start date for range (YYYY-MM-DD)')
    date_group.add_argument('--season', type=int, help='Single season year (e.g., 2024)')
    date_group.add_argument('--seasons', type=int, nargs='+', help='Multiple season years (e.g., 2020 2021 2022)')
    date_group.add_argument('--yesterday', action='store_true', help='Load yesterday\'s data')

    # End date for range
    parser.add_argument('--end', help='End date for range (YYYY-MM-DD). Required if --start is used')

    # Options
    parser.add_argument('--dry-run', action='store_true', help='Don\'t load to database (test only)')
    parser.add_argument('--force', action='store_true', help='Force reload even if data exists')
    parser.add_argument('--chunk-days', type=int, default=7, help='Days per chunk for API calls (default: 7)')

    args = parser.parse_args()

    # Validate arguments
    if args.start and not args.end:
        parser.error('--end is required when using --start')

    skip_if_exists = not args.force

    # Run appropriate pipeline
    if args.date:
        # Single day
        print(f"\n🚀 Running Statcast pipeline for {args.date}")
        result = run_statcast_daily_pipeline(
            date=args.date,
            dry_run=args.dry_run,
            skip_if_exists=skip_if_exists
        )
        print_summary([result])

    elif args.yesterday:
        # Yesterday's data
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        print(f"\n🚀 Running Statcast pipeline for yesterday ({yesterday})")
        result = run_statcast_daily_pipeline(
            date=yesterday,
            dry_run=args.dry_run,
            skip_if_exists=skip_if_exists
        )
        print_summary([result])

    elif args.start and args.end:
        # Date range
        print(f"\n🚀 Running Statcast pipeline for {args.start} to {args.end}")
        result = run_statcast_daterange_pipeline(
            start_date=args.start,
            end_date=args.end,
            chunk_days=args.chunk_days,
            dry_run=args.dry_run,
            skip_if_exists=skip_if_exists
        )
        print_summary([result])

    elif args.season:
        # Single season
        print(f"\n🚀 Running Statcast pipeline for {args.season} season")
        result = run_statcast_season_pipeline(
            season=args.season,
            dry_run=args.dry_run,
            skip_if_exists=skip_if_exists
        )
        print_summary([result])

    elif args.seasons:
        # Multiple seasons
        print(f"\n🚀 Running Statcast pipeline for seasons: {', '.join(map(str, args.seasons))}")
        results = []
        for season in sorted(args.seasons):
            result = run_statcast_season_pipeline(
                season=season,
                dry_run=args.dry_run,
                skip_if_exists=skip_if_exists
            )
            results.append(result)
        print_summary(results)


def print_summary(results):
    """Print summary of pipeline results"""
    print("\n" + "="*60)
    print("📊 PIPELINE SUMMARY")
    print("="*60)

    total_pitches = sum(r.get('pitches_inserted', 0) + r.get('pitches_updated', 0) for r in results)
    total_games = sum(r.get('games', 0) for r in results)
    skipped = sum(1 for r in results if r.get('status') == 'skipped')

    print(f"  Pipelines run: {len(results)}")
    print(f"  Skipped: {skipped}")
    print(f"  Total pitches loaded: {total_pitches:,}")
    if total_games > 0:
        print(f"  Total games: {total_games}")

    print("="*60 + "\n")


if __name__ == '__main__':
    main()
