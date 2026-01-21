"""
Daily Home Spread Cover Model Prediction Script

This script loads a pre-trained XGBoost model and generates spread cover
predictions for today's college basketball games.

Usage:
    python RunHomeSpreadCoverModel.py
"""

import sys
import os

# Add project root to path for generic imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from generic.db import get_engine,upsert_via_staging
from FeatureEngineering import load_home_game_data, build_model_dataframe
from DisplayTodaysPicks import display_todays_picks
from generic.pipeline_code.PickleModels import load_model

import pandas as pd
from tabulate import tabulate


def display_picks_table(picks_df: pd.DataFrame) -> None:
    """Display picks in a formatted table."""
    display_df = picks_df.copy()

    # Format columns for display
    display_df["startDate"] = pd.to_datetime(display_df["startDate"]).dt.strftime("%m/%d %I:%M %p")
    display_df["predProb"] = display_df["predProb"].apply(lambda x: f"{x:.1%}")
    display_df["edge"] = display_df["edge"].apply(lambda x: f"{x:.2%}")
    display_df["homeSpread"] = display_df["homeSpread"].apply(lambda x: f"{x:+.1f}")

    # Select and rename columns for display
    display_df = display_df[[
        "pickSide", "homeTeam", "awayTeam", "homeSpread", "predProb", "edge", "startDate", "venue"
    ]]
    display_df.columns = ["Pick", "Home", "Away", "Spread", "Prob", "Edge", "Time", "Venue"]

    print(tabulate(display_df, headers="keys", tablefmt="pretty", showindex=False))


# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "sports/cbb/models/saved_models/SpreadModel.pkl"

TARGET_COLUMN = "homeCover"

ROLLING_STATS = [
    "Pace", "OrbPct", "PointsOffTO", "Rating"
]

FEATURES = [
    "restDiff",
    "orbpct_diff_l3",
    "pace_diff_l3",
    "rating_diff_l3",
    "pointsoffto_diff_l3",
    "homeSpread",
]

ROLL_WINDOWS = [3]

# Predetermined thresholds from backtest optimization
LOWER_THRESHOLD = 0.423
UPPER_THRESHOLD = 0.530


def main():
    print("=" * 60)
    print("Home Spread Cover Model - Daily Predictions")
    print("=" * 60)

    # -----------------------------
    # LOAD MODEL
    # -----------------------------
    print("\nLoading pre-trained model...")
    model = load_model(MODEL_PATH)

    # -----------------------------
    # LOAD + BUILD DATA
    # -----------------------------
    print("Connecting to database...")
    engine = get_engine('CBB')

    print("Loading game data...")
    raw_games = load_home_game_data(engine)

    print("Building model dataframe with rolling features...")
    df_model = build_model_dataframe(
        game_df=raw_games,
        rolling_stats=ROLLING_STATS,
        windows=ROLL_WINDOWS,
        target=TARGET_COLUMN
    )

    # Get future games only
    now_utc = pd.Timestamp.utcnow()
    df_future = df_model[df_model["startDate"] >= now_utc]

    if df_future.empty:
        print("\nNo upcoming games found.")
        return

    print(f"\nFound {len(df_future)} upcoming games")

    # -----------------------------
    # GENERATE PREDICTIONS
    # -----------------------------
    print("Generating predictions...")
    X_pred = df_future[FEATURES]
    preds = model.predict(X_pred)

    df_out = pd.concat(
        [df_future.reset_index(drop=True), preds.reset_index(drop=True)],
        axis=1
    )

    # -----------------------------
    # DISPLAY TODAY'S PICKS
    # -----------------------------
    print(f"\nApplying thresholds: Lower={LOWER_THRESHOLD:.1%}, Upper={UPPER_THRESHOLD:.1%}")

    picks_df = display_todays_picks(
        df_out,
        LOWER_THRESHOLD,
        UPPER_THRESHOLD,
        engine
    )

    if picks_df is None or picks_df.empty:
        print("\nNo picks meet threshold criteria today.")
        return

    print(f"\n{len(picks_df)} picks generated:\n")
    display_picks_table(picks_df)

    # -----------------------------
    # SAVE TO DATABASE
    # -----------------------------
    print("\nSaving picks to database...")

    pks = ["gameId"]
    data_columns = [c for c in picks_df.columns if c not in pks + ["insert_date", "update_date"]]

    upsert_via_staging(
        df=picks_df,
        table_name="SpreadModelPicks",
        primary_keys=pks,
        data_columns=data_columns,
        engine=engine,
        schema='CBB',
        dry_run=False
    )

    print("\nDone!")


if __name__ == "__main__":
    main()