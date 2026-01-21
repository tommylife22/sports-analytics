"""
Player Data Cleaners
Cleaning and validation for player information and rosters
"""
import pandas as pd
from pandas import json_normalize
from .base_cleaners import validate_dataframe, remove_nulls_in_required_columns


def clean_player_roster_data(rosters_df):
    """
    Clean and validate player roster data - flattens nested players column

    Args:
        rosters_df (pd.DataFrame): Raw roster data from extractor with nested players column

    Returns:
        pd.DataFrame: Cleaned player data with one row per player
    """
    print("\n--- Cleaning Player Roster Data ---")
    original_len = len(rosters_df)
    
    # Explode players column so each player is a separate row
    rosters_df = rosters_df.copy()
    rosters_df = rosters_df.explode("players", ignore_index=True)
    
    # Flatten player dictionaries into columns
    players_flat = json_normalize(rosters_df["players"], sep="_")
    
    # Combine team info with flattened player data
    output_df = pd.concat([
        rosters_df[['teamId', 'team', 'conference', 'season']].reset_index(drop=True),
        players_flat.reset_index(drop=True)
    ], axis=1)
    
    # Rename columns to match database schema
    output_df.rename(columns={
        'teamId': 'team_id',
        'team': 'team_name',
        'id': 'player_id',
        'sourceId': 'source_id',
        'firstName': 'first_name',
        'lastName': 'last_name',
        'jersey': 'jersey_number',
        'height': 'height_inches',
        'weight': 'weight_lbs',
        'position': 'position',
        'startSeason': 'start_season',
        'endSeason': 'end_season'
    }, inplace=True)
    
    # Select relevant columns
    cols_to_keep = ['team_id', 'team_name', 'season', 'conference', 
                    'player_id', 'source_id', 'name', 'first_name', 'last_name',
                    'jersey_number', 'position', 'height_inches', 'weight_lbs',
                    'start_season', 'end_season']
    
    output_df = output_df[[col for col in cols_to_keep if col in output_df.columns]]
    
    # Remove rows with null player_id
    required = ['player_id', 'team_id', 'season']
    output_df = remove_nulls_in_required_columns(output_df, required, "Players")
    
    # Convert numeric columns
    numeric_cols = ['team_id', 'player_id', 'source_id', 'jersey_number', 
                    'height_inches', 'weight_lbs', 'start_season', 'end_season']
    for col in numeric_cols:
        if col in output_df.columns:
            output_df[col] = pd.to_numeric(output_df[col], errors='coerce').astype('Int64')
    
    # Fill missing jersey numbers
    output_df['jersey_number'].fillna(-1, inplace=True)
    
    # Fill missing season info
    output_df['start_season'].fillna(output_df['season'], inplace=True)
    output_df['end_season'].fillna(output_df['season'], inplace=True)
    
    print(f"  Original team rosters: {original_len}")
    print(f"  Total players: {len(output_df)}")
    print(f"  Columns: {list(output_df.columns)}")
    
    return output_df
