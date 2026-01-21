import pandas as pd
import requests
import os
from dotenv import load_dotenv

from .functions import upsert_via_staging

load_dotenv()

API_KEY = os.environ.get('CBB_API_KEY')
BASE_URL = "https://api.collegebasketballdata.com/conferences"

def getConferenceInfo() -> pd.DataFrame:

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Accept": "application/json",
    }

    resp = requests.get(BASE_URL, headers=headers, timeout=30)
    resp.raise_for_status()
    
    return pd.DataFrame(resp.json())

def coerceConferenceInfoDtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce dtypes for ConferenceInfo to match SQL Server DDL.

    SQL Types:
      - conferenceId: VARCHAR(25)
      - sourceId: VARCHAR(50)
      - name, abbreviation, shortName: VARCHAR
      - PK = conferenceId → must not be null
    """

    df = df.copy()
    
    df.rename(columns={'id':'conferenceId'},inplace=True)

    # ---- ID-like fields → numeric → strip decimals → string ----
    id_like_cols = ["conferenceId", "sourceId"]

    for col in id_like_cols:
        if col in df.columns:
            # Convert numeric-looking values -> remove decimal
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].astype("Int64")      # support NA
            df[col] = df[col].astype("string")     # final type

    # ---- Text fields ----
    varchar_cols = ["name", "abbreviation", "shortName"]

    for col in varchar_cols:
        if col in df.columns:
            df[col] = df[col].astype("string")

    # ---- PK enforcement ----
    if "conferenceId" in df.columns:
        df = df[df["conferenceId"].notna()]

    return df

def loadConferenceInfo(engine):
    
    df = coerceConferenceInfoDtypes(getConferenceInfo())
    
    pks = ['conferenceId']
    data_columns = [c for c in df.columns if c not in pks + ["insert_date", "update_date"]]

    upsert_via_staging(
        df              = df,
        table_name      = "ConferenceInfo",
        primary_keys    = pks,
        data_columns    = data_columns,
        engine          = engine,
        schema          = 'CBB',
        dry_run         = False
    )
    
#loadConferenceInfo(engine)