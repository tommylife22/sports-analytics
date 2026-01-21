"""
Statcast Data Cleaners
Transform raw Statcast data into database-ready format
"""
import pandas as pd
import numpy as np


def clean_statcast_pitches(df):
    """
    Clean and prepare Statcast pitch-level data for database insertion

    Args:
        df (pd.DataFrame): Raw Statcast data from pybaseball

    Returns:
        pd.DataFrame: Cleaned data matching StatcastPitches table schema
    """
    if df is None or len(df) == 0:
        return pd.DataFrame()

    # Create a copy to avoid modifying original
    cleaned = df.copy()

    # ====================
    # Column Selection & Renaming
    # ====================
    column_mapping = {
        # Identifiers
        'game_pk': 'game_pk',
        'game_date': 'game_date',
        'game_year': 'game_year',
        'at_bat_number': 'at_bat_number',
        'pitch_number': 'pitch_number',

        # Players (MLBAM IDs)
        'pitcher': 'pitcher',
        'batter': 'batter',
        'player_name': 'pitcher_name',  # This is pitcher name in pybaseball

        # Game State
        'inning': 'inning',
        'inning_topbot': 'inning_topbot',
        'outs_when_up': 'outs_when_up',
        'balls': 'balls',
        'strikes': 'strikes',
        'on_1b': 'on_1b',
        'on_2b': 'on_2b',
        'on_3b': 'on_3b',
        'stand': 'stand',
        'p_throws': 'p_throws',

        # Pitch Characteristics
        'pitch_type': 'pitch_type',
        'pitch_name': 'pitch_name',
        'release_speed': 'release_speed',
        'release_pos_x': 'release_pos_x',
        'release_pos_y': 'release_pos_y',
        'release_pos_z': 'release_pos_z',
        'release_spin_rate': 'release_spin_rate',
        'release_extension': 'release_extension',
        'spin_axis': 'spin_axis',

        # Pitch Movement
        'pfx_x': 'pfx_x',
        'pfx_z': 'pfx_z',
        'plate_x': 'plate_x',
        'plate_z': 'plate_z',

        # Velocity Components
        'vx0': 'vx0',
        'vy0': 'vy0',
        'vz0': 'vz0',

        # Acceleration Components
        'ax': 'ax',
        'ay': 'ay',
        'az': 'az',

        # Strike Zone
        'zone': 'zone',
        'sz_top': 'sz_top',
        'sz_bot': 'sz_bot',

        # Batted Ball Data
        'launch_speed': 'launch_speed',
        'launch_angle': 'launch_angle',
        'hit_distance_sc': 'hit_distance_sc',
        'bb_type': 'bb_type',
        'hc_x': 'hc_x',
        'hc_y': 'hc_y',

        # Barrel/Expected Stats
        'barrel': 'barrel',
        'estimated_ba_using_speedangle': 'estimated_ba_using_speedangle',
        'estimated_woba_using_speedangle': 'estimated_woba_using_speedangle',
        'woba_value': 'woba_value',
        'woba_denom': 'woba_denom',
        'babip_value': 'babip_value',
        'iso_value': 'iso_value',

        # Outcome
        'type': 'type',
        'description': 'description',
        'events': 'events',
        'des': 'des',

        # Home Run Details
        'home_run': 'home_run',
        'hit_location': 'hit_location',

        # Additional Context
        'if_fielding_alignment': 'if_fielding_alignment',
        'of_fielding_alignment': 'of_fielding_alignment',
    }

    # Select and rename columns that exist
    available_columns = {k: v for k, v in column_mapping.items() if k in cleaned.columns}
    cleaned = cleaned[list(available_columns.keys())].rename(columns=available_columns)

    # ====================
    # Data Type Conversions
    # ====================

    # Convert game_pk to string (matches GameInfo.game_id VARCHAR(10))
    if 'game_pk' in cleaned.columns:
        cleaned['game_pk'] = cleaned['game_pk'].astype(str)

    # Convert date to proper format
    if 'game_date' in cleaned.columns:
        cleaned['game_date'] = pd.to_datetime(cleaned['game_date']).dt.date

    # Integer conversions with NaN handling
    int_columns = [
        'game_year', 'pitcher', 'batter', 'inning', 'outs_when_up', 'balls', 'strikes',
        'on_1b', 'on_2b', 'on_3b', 'release_spin_rate', 'spin_axis', 'zone',
        'hit_distance_sc', 'woba_denom', 'at_bat_number', 'pitch_number'
    ]
    for col in int_columns:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors='coerce')
            # Keep as float if contains NaN, otherwise convert to int
            if not cleaned[col].isna().any():
                cleaned[col] = cleaned[col].astype('Int64')

    # Float/Decimal conversions
    float_columns = [
        'release_speed', 'release_pos_x', 'release_pos_y', 'release_pos_z',
        'release_extension', 'pfx_x', 'pfx_z', 'plate_x', 'plate_z',
        'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az', 'sz_top', 'sz_bot',
        'launch_speed', 'launch_angle', 'hc_x', 'hc_y',
        'estimated_ba_using_speedangle', 'estimated_woba_using_speedangle',
        'woba_value', 'babip_value', 'iso_value'
    ]
    for col in float_columns:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors='coerce')

    # Boolean conversions
    if 'barrel' in cleaned.columns:
        cleaned['barrel'] = cleaned['barrel'].astype(bool).astype(int)  # Convert to 0/1 for BIT type

    if 'home_run' in cleaned.columns:
        # Create home_run column if not present (1 if events == 'home_run', else 0)
        if 'events' in cleaned.columns:
            cleaned['home_run'] = (cleaned['events'] == 'home_run').astype(int)

    # String columns - truncate to max lengths
    string_columns = {
        'pitcher_name': 100,
        'batter_name': 100,
        'inning_topbot': 10,
        'stand': 1,
        'p_throws': 1,
        'pitch_type': 10,
        'pitch_name': 50,
        'bb_type': 20,
        'type': 2,
        'description': 100,
        'events': 50,
        'des': 500,
        'hit_location': 10,
        'if_fielding_alignment': 20,
        'of_fielding_alignment': 20
    }
    for col, max_len in string_columns.items():
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].astype(str).str[:max_len]
            cleaned[col] = cleaned[col].replace('nan', None)  # Replace string 'nan' with None

    # ====================
    # Add batter_name if not present
    # ====================
    # pybaseball doesn't always include batter names, may need to join with PlayerInfo later
    if 'batter_name' not in cleaned.columns:
        cleaned['batter_name'] = None

    # ====================
    # Remove duplicates
    # ====================
    # Some pitches may be duplicated in the data
    # Use game_pk, at_bat_number, pitch_number as composite key
    if all(col in cleaned.columns for col in ['game_pk', 'at_bat_number', 'pitch_number']):
        cleaned = cleaned.drop_duplicates(
            subset=['game_pk', 'at_bat_number', 'pitch_number'],
            keep='last'
        )

    # ====================
    # Data Quality Filtering
    # ====================
    # Remove rows with missing critical fields
    cleaned = cleaned.dropna(subset=['game_pk', 'game_date', 'pitcher', 'batter'])

    # ====================
    # Sort by game and pitch sequence
    # ====================
    if all(col in cleaned.columns for col in ['game_date', 'game_pk', 'at_bat_number', 'pitch_number']):
        cleaned = cleaned.sort_values(
            ['game_date', 'game_pk', 'at_bat_number', 'pitch_number']
        ).reset_index(drop=True)

    return cleaned


def add_derived_features(df):
    """
    Add calculated features from raw Statcast data (optional for ML)

    Args:
        df (pd.DataFrame): Cleaned Statcast data

    Returns:
        pd.DataFrame: Data with additional derived features

    Note:
        This is optional - use if you want to pre-compute common features
        Otherwise, calculate features on-the-fly from raw pitch data
    """
    df = df.copy()

    # Calculate swing indicator
    if 'description' in df.columns:
        swing_descriptions = ['swinging_strike', 'foul', 'hit_into_play', 'swinging_strike_blocked',
                              'foul_tip', 'missed_bunt', 'foul_bunt']
        df['swing'] = df['description'].isin(swing_descriptions).astype(int)

    # Calculate whiff (swing and miss)
    if 'description' in df.columns:
        whiff_descriptions = ['swinging_strike', 'swinging_strike_blocked', 'missed_bunt']
        df['whiff'] = df['description'].isin(whiff_descriptions).astype(int)

    # Calculate in-zone indicator
    if 'zone' in df.columns:
        df['in_zone'] = (df['zone'] <= 9).astype(int)  # Zones 1-9 are strike zone

    # Calculate hard hit (95+ mph exit velo)
    if 'launch_speed' in df.columns:
        df['hard_hit'] = (df['launch_speed'] >= 95).astype(int)

    return df
