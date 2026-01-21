"""
MLB Pipeline Runner
Main entry point for running the MLB data pipeline
"""
from src.orchestrators import (
    build_season_database,
    build_multi_season_database,
    build_boxscore_database,
    build_multi_season_boxscore_database
)


def run_single_season(season, load_to_db=True, dry_run=False, include_boxscores=False):
    """
    Run pipeline for a single season

    Args:
        season (int): Season year
        load_to_db (bool): If True, load data to database
        dry_run (bool): If True, don't actually load to database
        include_boxscores (bool): If True, also extract and load boxscore data

    Returns:
        dict: Dictionary with 'info' and optionally 'boxscore' dataframes
    """
    # Build info tables (teams, players, games)
    teams_df, players_df, games_df = build_season_database(
        season=season,
        load_to_db=load_to_db,
        dry_run=dry_run
    )

    result = {
        'info': {
            'teams': teams_df,
            'players': players_df,
            'games': games_df
        }
    }

    # Optionally build boxscore tables
    if include_boxscores:
        team_boxscores_df, player_boxscores_df = build_boxscore_database(
            season=season,
            games_df=games_df,  # Reuse games we already fetched
            load_to_db=load_to_db,
            dry_run=dry_run
        )
        result['boxscore'] = {
            'team': team_boxscores_df,
            'player': player_boxscores_df
        }

    return result


def run_multi_season(seasons, load_to_db=True, dry_run=False, include_boxscores=False):
    """
    Run pipeline for multiple seasons

    Args:
        seasons (list): List of season years
        load_to_db (bool): If True, load data to database
        dry_run (bool): If True, don't actually load to database
        include_boxscores (bool): If True, also extract and load boxscore data

    Returns:
        dict: Dictionary with 'info' and optionally 'boxscore' dataframes
    """
    # Build info tables (teams, players, games)
    teams_df, players_df, games_df = build_multi_season_database(
        seasons=seasons,
        load_to_db=load_to_db,
        dry_run=dry_run
    )

    result = {
        'info': {
            'teams': teams_df,
            'players': players_df,
            'games': games_df
        }
    }

    # Optionally build boxscore tables
    if include_boxscores:
        # Create games_dict for efficient lookup by season
        games_dict = {}
        for season in seasons:
            games_dict[season] = games_df[games_df['season'] == str(season)]

        team_boxscores_df, player_boxscores_df = build_multi_season_boxscore_database(
            seasons=seasons,
            games_dict=games_dict,  # Reuse games we already fetched
            load_to_db=load_to_db,
            dry_run=dry_run
        )
        result['boxscore'] = {
            'team': team_boxscores_df,
            'player': player_boxscores_df
        }

    return result


if __name__ == "__main__":
    """
    Example usage - modify as needed
    """
    import argparse

    parser = argparse.ArgumentParser(description='Run MLB data pipeline')
    parser.add_argument('--season', type=int, help='Single season to process (e.g., 2025)')
    parser.add_argument('--seasons', type=int, nargs='+', help='Multiple seasons to process (e.g., 2023 2024 2025)')
    parser.add_argument('--dry-run', action='store_true', help='Dry run - do not load to database')
    parser.add_argument('--no-load', action='store_true', help='Extract and clean only, do not load to database')
    parser.add_argument('--include-boxscores', action='store_true', help='Also extract and load boxscore data')

    args = parser.parse_args()

    # Determine what to run
    if args.season:
        print(f"Running pipeline for {args.season} season...")
        result = run_single_season(
            season=args.season,
            load_to_db=not args.no_load,
            dry_run=args.dry_run,
            include_boxscores=args.include_boxscores
        )

    elif args.seasons:
        print(f"Running pipeline for seasons: {args.seasons}")
        result = run_multi_season(
            seasons=args.seasons,
            load_to_db=not args.no_load,
            dry_run=args.dry_run,
            include_boxscores=args.include_boxscores
        )

    else:
        # Default: run for 2025 season
        print("No arguments provided. Running default: 2025 season")
        print("Usage examples:")
        print("  python run_pipeline.py --season 2025")
        print("  python run_pipeline.py --seasons 2023 2024 2025")
        print("  python run_pipeline.py --season 2025 --dry-run")
        print("  python run_pipeline.py --season 2025 --no-load")
        print("  python run_pipeline.py --season 2025 --include-boxscores")
        print("\nRunning default...")

        result = run_single_season(
            season=2025,
            load_to_db=True,
            dry_run=False,
            include_boxscores=False
        )

    print(f"\n✓ Pipeline completed successfully!")
    print(f"  Teams: {len(result['info']['teams'])}")
    print(f"  Players: {len(result['info']['players'])}")
    print(f"  Games: {len(result['info']['games'])}")
    if 'boxscore' in result:
        print(f"  Team Boxscores: {len(result['boxscore']['team'])}")
        print(f"  Player Boxscores: {len(result['boxscore']['player'])}")
