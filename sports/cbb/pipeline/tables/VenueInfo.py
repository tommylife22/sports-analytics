import pandas as pd
import requests
import os
from dotenv import load_dotenv

from .functions import upsert_via_staging

load_dotenv()

API_KEY = os.environ.get('CBB_API_KEY')
BASE_URL = "https://api.collegebasketballdata.com/venues"

def getVenueInfo() -> pd.DataFrame:

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Accept": "application/json",
    }

    resp = requests.get(BASE_URL, headers=headers, timeout=30)
    resp.raise_for_status()
    
    return pd.DataFrame(resp.json())

def coerceVenueInfoDtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce dtypes for VenueInfo to match SQL Server DDL.

    SQL Types:
      - venueId: VARCHAR(25)
      - sourceId: VARCHAR(255)
      - name, city, state, country: VARCHAR
      - PK = venueId → must not be null
    """

    df = df.copy()
    
    df.rename(columns={'id':'venueId'},inplace=True)

    # ---- ID-like fields → numeric → remove decimals → string ----
    id_like_cols = ["venueId", "sourceId"]

    for col in id_like_cols:
        if col in df.columns:
            # Convert numeric-looking values to numeric so .0 goes away
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].astype("Int64")        # for nullable ints
            df[col] = df[col].astype("string")       # final string type

    # ---- VARCHAR fields ----
    varchar_cols = ["name", "city", "state", "country"]

    for col in varchar_cols:
        if col in df.columns:
            df[col] = df[col].astype("string")

    # ---- Drop rows with missing venueId (PK constraint) ----
    if "venueId" in df.columns:
        df = df[df["venueId"].notna()]

    return df


def loadVenueInfo(engine):
    
    df = coerceVenueInfoDtypes(getVenueInfo())
    
    pks = ['venueId']
    data_columns = [c for c in df.columns if c not in pks + ["insert_date", "update_date"]]

    upsert_via_staging(
        df              = df,
        table_name      = "VenueInfo",
        primary_keys    = pks,
        data_columns    = data_columns,
        engine          = engine,
        schema          = 'dbo',
        dry_run         = False
    )