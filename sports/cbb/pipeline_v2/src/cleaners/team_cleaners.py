"""
Team Data Cleaners
Cleaning and validation for team information
"""
import pandas as pd
from .base_cleaners import validate_dataframe, remove_nulls_in_required_columns


def clean_team_data(teams_df):
    """
    Clean and validate team data

    Args:
        teams_df (pd.DataFrame): Raw team data from extractor

    Returns:
        pd.DataFrame: Cleaned team data
    """
    print("\n--- Cleaning Team Data ---")
    original_len = len(teams_df)
    
    # Select relevant columns
    cols_to_keep = ['id', 'sourceId', 'school', 'mascot', 'abbreviation', 
                    'displayName', 'conferenceId', 'conference']
    
    teams_df = teams_df[[col for col in cols_to_keep if col in teams_df.columns]].copy()
    
    # Rename columns to match database schema
    teams_df.rename(columns={
        'id': 'team_id',
        'sourceId': 'source_id',
        'school': 'school_name',
        'abbreviation': 'team_abbr',
        'displayName': 'display_name',
        'conference': 'conference_name',
        'conferenceId': 'conference_id'
    }, inplace=True)
    
    # Remove rows with null required fields
    required = ['team_id', 'school_name', 'team_abbr']
    teams_df = remove_nulls_in_required_columns(teams_df, required, "Teams")
    
    # Convert IDs to int
    for col in ['team_id', 'source_id', 'conference_id']:
        if col in teams_df.columns:
            teams_df[col] = pd.to_numeric(teams_df[col], errors='coerce').astype('Int64')
    
    # Fill missing conference info
    teams_df['conference_name'].fillna('Unknown', inplace=True)
    teams_df['conference_id'].fillna(-1, inplace=True)
    
    print(f"  Original rows: {original_len}")
    print(f"  Cleaned rows: {len(teams_df)}")
    print(f"  Columns: {list(teams_df.columns)}")
    
    return teams_df
