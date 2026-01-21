"""
Betting Odds Data Loaders
Functions for loading betting odds data to database
"""
import pandas as pd
from .base_loader import load_table_to_database
from ..utils.constants import DEFAULT_SCHEMA


def prepare_odds_for_database(moneyline_df, spread_df, total_df):
    """
    Combine and prepare odds data for database loading

    Args:
        moneyline_df (DataFrame): Moneyline odds data
        spread_df (DataFrame): Spread odds data
        total_df (DataFrame): Total odds data

    Returns:
        DataFrame: Combined and formatted odds data
    """
    all_odds = []

    # Process moneyline
    if len(moneyline_df) > 0:
        ml_formatted = moneyline_df.copy()
        ml_formatted['away_spread'] = None
        ml_formatted['away_spread_odds'] = None
        ml_formatted['home_spread'] = None
        ml_formatted['home_spread_odds'] = None
        ml_formatted['total'] = None
        ml_formatted['over_odds'] = None
        ml_formatted['under_odds'] = None
        all_odds.append(ml_formatted)

    # Process spread
    if len(spread_df) > 0:
        spread_formatted = spread_df.copy()
        # Rename to match database schema
        spread_formatted['away_line'] = None
        spread_formatted['home_line'] = None
        spread_formatted['away_spread_odds'] = spread_formatted['away_odds']
        spread_formatted['home_spread_odds'] = spread_formatted['home_odds']
        spread_formatted.drop(['away_odds', 'home_odds'], axis=1, inplace=True)
        spread_formatted['total'] = None
        spread_formatted['over_odds'] = None
        spread_formatted['under_odds'] = None
        all_odds.append(spread_formatted)

    # Process total
    if len(total_df) > 0:
        total_formatted = total_df.copy()
        total_formatted['away_line'] = None
        total_formatted['home_line'] = None
        total_formatted['away_spread'] = None
        total_formatted['away_spread_odds'] = None
        total_formatted['home_spread'] = None
        total_formatted['home_spread_odds'] = None
        all_odds.append(total_formatted)

    # Combine all odds
    if all_odds:
        combined_df = pd.concat(all_odds, ignore_index=True)

        # Ensure correct column order matching database schema
        columns_order = [
            'date', 'game_id', 'game_time', 'away_team', 'home_team',
            'sportsbook', 'bet_type',
            'away_line', 'home_line',
            'away_spread', 'away_spread_odds', 'home_spread', 'home_spread_odds',
            'total', 'over_odds', 'under_odds'
        ]

        combined_df = combined_df[columns_order]

        return combined_df

    return pd.DataFrame()


def load_odds_to_database(odds_df, engine=None, schema=DEFAULT_SCHEMA, dry_run=False):
    """
    Load betting odds data to database

    Args:
        odds_df (DataFrame): Combined odds data
        engine: SQLAlchemy engine (if None, creates new one)
        schema (str): Database schema name
        dry_run (bool): If True, don't actually load to database

    Returns:
        dict: Load results
    """
    if len(odds_df) == 0:
        print("  ⚠ No odds data to load")
        return {'rows_inserted': 0, 'rows_updated': 0}

    # Make a copy to avoid modifying the original
    odds_df = odds_df.copy()

    # Convert game_id to string to match schema
    odds_df['game_id'] = odds_df['game_id'].astype(str)

    # Convert date to datetime (always convert, don't rely on dtype check)
    odds_df['date'] = pd.to_datetime(odds_df['date'])

    # Convert game_time to datetime, removing timezone info for SQL Server compatibility
    # Parse with utc=True to handle timezone strings, then remove tz
    odds_df['game_time'] = pd.to_datetime(odds_df['game_time'], utc=True).dt.tz_localize(None)

    return load_table_to_database(
        df=odds_df,
        table_name='BettingOdds',
        engine=engine,
        schema=schema,
        dry_run=dry_run
    )


def load_scraped_odds_to_database(odds_results, engine=None, schema=DEFAULT_SCHEMA, dry_run=False):
    """
    Load scraped odds results (dict of DataFrames) to database

    Args:
        odds_results (dict): Dictionary with keys 'moneyline', 'spread', 'total'
        engine: SQLAlchemy engine (if None, creates new one)
        schema (str): Database schema name
        dry_run (bool): If True, don't actually load to database

    Returns:
        dict: Load results
    """
    print("\n" + "="*50)
    print("LOADING BETTING ODDS TO DATABASE")
    print("="*50)

    # Get DataFrames
    moneyline_df = odds_results.get('moneyline', pd.DataFrame())
    spread_df = odds_results.get('spread', pd.DataFrame())
    total_df = odds_results.get('total', pd.DataFrame())

    print(f"\n  Moneyline records: {len(moneyline_df)}")
    print(f"  Spread records:    {len(spread_df)}")
    print(f"  Total records:     {len(total_df)}")

    # Prepare combined data
    print("\n  Preparing data for database...")
    combined_df = prepare_odds_for_database(moneyline_df, spread_df, total_df)

    print(f"  Combined records:  {len(combined_df)}")

    # Load to database
    result = load_odds_to_database(
        odds_df=combined_df,
        engine=engine,
        schema=schema,
        dry_run=dry_run
    )

    # Summary
    print("\n" + "="*50)
    print("BETTING ODDS LOAD SUMMARY")
    print("="*50)
    print(f"  Records inserted: {result.get('rows_inserted', 0)}")
    print(f"  Records updated:  {result.get('rows_updated', 0)}")
    print("="*50)

    return result
