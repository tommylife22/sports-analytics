"""
Run Boxscore Pipeline
Extract, clean, and load MLB boxscore data to database
"""
import argparse
from src import build_boxscore_database, build_multi_season_boxscore_database


def main():
    parser = argparse.ArgumentParser(description='Run MLB Boxscore Pipeline')

    # Season arguments
    parser.add_argument('--season', type=int, help='Single season to process')
    parser.add_argument('--seasons', type=int, nargs='+', help='Multiple seasons to process')

    # Database arguments
    parser.add_argument('--dry-run', action='store_true', help='Preview without loading to database')
    parser.add_argument('--no-load', action='store_true', help='Extract and clean only, do not load to database')

    args = parser.parse_args()

    # Determine load_to_db flag
    load_to_db = not args.no_load

    # Run pipeline
    if args.season:
        # Single season
        team_boxscores, player_boxscores = build_boxscore_database(
            season=args.season,
            load_to_db=load_to_db,
            dry_run=args.dry_run
        )
    elif args.seasons:
        # Multiple seasons
        team_boxscores, player_boxscores = build_multi_season_boxscore_database(
            seasons=args.seasons,
            load_to_db=load_to_db,
            dry_run=args.dry_run
        )
    else:
        parser.print_help()
        return

    print("\n✓ Pipeline completed successfully!")


if __name__ == "__main__":
    main()