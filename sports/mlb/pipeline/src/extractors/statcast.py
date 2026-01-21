"""
Statcast Data Extractor
Handles data extraction from pybaseball for Statcast pitch-level data
"""
from pybaseball import statcast
import pandas as pd
from datetime import datetime, timedelta
import time


def get_statcast_data(start_date, end_date=None, team=None, verbose=True, max_retries=3, retry_delay=10):
    """
    Get Statcast pitch-level data for a date range

    Args:
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str, optional): End date in 'YYYY-MM-DD' format. If None, uses start_date
        team (str, optional): Team abbreviation (e.g., 'NYY', 'BOS')
        verbose (bool): Show progress messages
        max_retries (int): Maximum retry attempts on failure
        retry_delay (int): Seconds to wait between retries

    Returns:
        pd.DataFrame: Statcast data with pitch-level details

    Note:
        - Baseball Savant limits queries to 30,000 rows
        - Data availability: 2008+ (launch angle/speed from 2015+)
        - Recommended to query in small date ranges (1-7 days)
    """
    if end_date is None:
        end_date = start_date

    if verbose:
        print(f"  Fetching Statcast data: {start_date} to {end_date}")
        if team:
            print(f"  Team filter: {team}")

    for attempt in range(max_retries):
        try:
            # Call pybaseball statcast function
            df = statcast(start_dt=start_date, end_dt=end_date, team=team, verbose=verbose)

            if df is None or len(df) == 0:
                if verbose:
                    print(f"  ⚠ No data returned for {start_date} to {end_date}")
                return pd.DataFrame()

            if verbose:
                print(f"  ✓ Fetched {len(df):,} pitches")

            return df

        except Exception as e:
            if attempt < max_retries - 1:
                if verbose:
                    print(f"  ⚠ Error (attempt {attempt + 1}/{max_retries}): {e}")
                    print(f"  Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                if verbose:
                    print(f"  ✗ Failed after {max_retries} attempts: {e}")
                raise


def get_statcast_bulk(start_date, end_date, chunk_days=7, team=None, verbose=True):
    """
    Get Statcast data for a large date range by chunking into smaller requests

    Args:
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        chunk_days (int): Number of days per chunk (default 7 to stay under 30k row limit)
        team (str, optional): Team abbreviation filter
        verbose (bool): Show progress

    Returns:
        pd.DataFrame: Combined Statcast data for entire date range

    Example:
        # Get full 2024 season
        df = get_statcast_bulk('2024-03-28', '2024-09-29')
    """
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')

    all_data = []
    current = start

    if verbose:
        total_days = (end - start).days + 1
        print(f"\n📊 Fetching Statcast data: {start_date} to {end_date} ({total_days} days)")
        print(f"  Chunking into {chunk_days}-day segments...")

    chunk_num = 1
    while current <= end:
        chunk_end = min(current + timedelta(days=chunk_days - 1), end)

        chunk_start_str = current.strftime('%Y-%m-%d')
        chunk_end_str = chunk_end.strftime('%Y-%m-%d')

        if verbose:
            print(f"\n  Chunk {chunk_num}: {chunk_start_str} to {chunk_end_str}")

        try:
            df = get_statcast_data(
                start_date=chunk_start_str,
                end_date=chunk_end_str,
                team=team,
                verbose=verbose
            )

            if len(df) > 0:
                all_data.append(df)

            chunk_num += 1
            current = chunk_end + timedelta(days=1)

            # Be nice to Baseball Savant API
            if current <= end:
                time.sleep(2)

        except Exception as e:
            print(f"  ✗ Error processing chunk {chunk_start_str} to {chunk_end_str}: {e}")
            # Continue with next chunk
            current = chunk_end + timedelta(days=1)
            continue

    if len(all_data) == 0:
        if verbose:
            print("\n  ⚠ No data retrieved for date range")
        return pd.DataFrame()

    # Combine all chunks
    combined_df = pd.concat(all_data, ignore_index=True)

    if verbose:
        print(f"\n✓ Successfully fetched {len(combined_df):,} total pitches")
        print(f"  Date range: {combined_df['game_date'].min()} to {combined_df['game_date'].max()}")
        print(f"  Games: {combined_df['game_pk'].nunique()}")
        print(f"  Pitchers: {combined_df['pitcher'].nunique()}")
        print(f"  Batters: {combined_df['batter'].nunique()}")

    return combined_df


def get_statcast_season(season, team=None, verbose=True):
    """
    Get Statcast data for an entire MLB season

    Args:
        season (int): Season year (e.g., 2024)
        team (str, optional): Team abbreviation filter
        verbose (bool): Show progress

    Returns:
        pd.DataFrame: Season Statcast data

    Note:
        Automatically determines season dates
        Regular season typically runs March/April through September/October
    """
    # Typical season dates by year
    season_dates = {
        2024: ('2024-03-28', '2024-09-29'),
        2023: ('2023-03-30', '2023-10-01'),
        2022: ('2022-04-07', '2022-10-05'),
        2021: ('2021-04-01', '2021-10-03'),
        2020: ('2020-07-23', '2020-09-27'),  # COVID shortened season
        2019: ('2019-03-28', '2019-09-29'),
        2018: ('2018-03-29', '2018-10-01'),
        2017: ('2017-04-02', '2017-10-01'),
        2016: ('2016-04-03', '2016-10-02'),
        2015: ('2015-04-05', '2015-10-04'),
    }

    if season in season_dates:
        start_date, end_date = season_dates[season]
    else:
        # Fallback: estimate season dates
        start_date = f"{season}-03-28"
        end_date = f"{season}-10-01"
        if verbose:
            print(f"  ⚠ Using estimated dates for {season} season: {start_date} to {end_date}")

    if verbose:
        print(f"\n📅 Fetching {season} season Statcast data")

    return get_statcast_bulk(
        start_date=start_date,
        end_date=end_date,
        chunk_days=7,
        team=team,
        verbose=verbose
    )
