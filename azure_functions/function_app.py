"""
Azure Function App - Sports Analytics CBB Functions

Two timer-triggered functions:
1. cbb_pipeline_timer - Runs daily data pipeline at 6:00 AM UTC
2. cbb_spread_model_timer - Runs spread model predictions at 6:30 AM UTC
"""
import azure.functions as func
import logging
import sys
import os
from datetime import date, timedelta

# =============================================================================
# PATH SETUP - Must happen before project imports
# =============================================================================
FUNCTION_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(FUNCTION_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# =============================================================================
# PROJECT IMPORTS
# =============================================================================
from generic.db import get_engine, upsert_via_staging
from sports.cbb.pipeline.tables.TeamInfo import loadTeamInfo
from sports.cbb.pipeline.tables.PlayerInfo import loadPlayerInfo
from sports.cbb.pipeline.tables.ConferenceInfo import loadConferenceInfo
from sports.cbb.pipeline.tables.VenueInfo import loadVenueInfo
from sports.cbb.pipeline.tables.GameInfo import loadGameInfo
from sports.cbb.pipeline.tables.GameBoxscoreTeam import loadGameBoxscoreTeam
from sports.cbb.pipeline.tables.GameBoxscorePlayer import loadGameBoxscorePlayer
from sports.cbb.pipeline.tables.GameLines import loadGameLines

# =============================================================================
# INITIALIZE FUNCTION APP
# =============================================================================
app = func.FunctionApp()


# =============================================================================
# PIPELINE CONFIGURATION
# =============================================================================
PIPELINE_CONFIG = {
    "season": 2025,
    "days_back": 7,
    "days_ahead": 1,
}

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
MODEL_CONFIG = {
    "model_path": os.path.join(PROJECT_ROOT, "sports/cbb/models/saved_models/SpreadModel.pkl"),
    "target_column": "homeCover",
    "rolling_stats": ["Pace", "OrbPct", "PointsOffTO", "Rating"],
    "features": [
        "restDiff",
        "orbpct_diff_l3",
        "pace_diff_l3",
        "rating_diff_l3",
        "pointsoffto_diff_l3",
        "homeSpread",
    ],
    "roll_windows": [3],
    "lower_threshold": 0.423,
    "upper_threshold": 0.530,
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def run_pipeline_step(name: str, func_to_run, *args) -> dict:
    """Execute a pipeline step with error handling and timing."""
    import time
    start = time.time()
    try:
        func_to_run(*args)
        elapsed = time.time() - start
        logging.info(f"  {name}: OK ({elapsed:.1f}s)")
        return {"step": name, "status": "success", "elapsed": round(elapsed, 2)}
    except Exception as e:
        elapsed = time.time() - start
        logging.error(f"  {name}: FAILED - {str(e)}")
        return {"step": name, "status": "failed", "error": str(e), "elapsed": round(elapsed, 2)}


def run_cbb_pipeline() -> dict:
    """
    Run the CBB data pipeline.

    Returns dict with execution results.
    """
    import time

    season = PIPELINE_CONFIG["season"]
    days_back = PIPELINE_CONFIG["days_back"]
    days_ahead = PIPELINE_CONFIG["days_ahead"]

    start_date = date.today() - timedelta(days=days_back)
    end_date = date.today() + timedelta(days=days_ahead)

    logging.info("=" * 50)
    logging.info("CBB PIPELINE - DAILY UPDATE")
    logging.info("=" * 50)
    logging.info(f"  Season: {season}")
    logging.info(f"  Date range: {start_date} to {end_date}")
    logging.info("=" * 50)

    total_start = time.time()
    results = []

    try:
        engine = get_engine('CBB')

        # Info tables
        results.append(run_pipeline_step("TeamInfo", loadTeamInfo, engine, season))
        results.append(run_pipeline_step("PlayerInfo", loadPlayerInfo, engine, season))
        results.append(run_pipeline_step("ConferenceInfo", loadConferenceInfo, engine))
        results.append(run_pipeline_step("VenueInfo", loadVenueInfo, engine))

        # Game tables
        results.append(run_pipeline_step("GameInfo", loadGameInfo, engine, start_date, end_date))
        results.append(run_pipeline_step("GameBoxscoreTeam", loadGameBoxscoreTeam, engine, start_date, end_date))
        results.append(run_pipeline_step("GameBoxscorePlayer", loadGameBoxscorePlayer, engine, start_date, end_date))
        results.append(run_pipeline_step("GameLines", loadGameLines, engine, start_date, end_date))

    except Exception as e:
        logging.error(f"Pipeline failed with error: {str(e)}")
        return {"status": "failed", "error": str(e), "steps": results}

    total_elapsed = time.time() - total_start
    failed_steps = [r for r in results if r["status"] == "failed"]

    logging.info("=" * 50)
    logging.info("PIPELINE SUMMARY")
    logging.info("=" * 50)
    for r in results:
        status = "OK" if r["status"] == "success" else "FAILED"
        logging.info(f"  {r['step']:<25} {status:<8} {r['elapsed']:>6.1f}s")
    logging.info("-" * 50)
    logging.info(f"  {'TOTAL':<25} {'':<8} {total_elapsed:>6.1f}s")
    logging.info("=" * 50)

    return {
        "status": "success" if not failed_steps else "partial_failure",
        "steps": results,
        "failed_count": len(failed_steps),
        "total_elapsed": round(total_elapsed, 2),
    }


def run_spread_model() -> dict:
    """
    Run the spread model predictions.

    Returns dict with execution results.
    """
    import time
    import pandas as pd

    # Import model-specific modules
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "sports/cbb/models"))
    from FeatureEngineering import load_home_game_data, build_model_dataframe
    from DisplayTodaysPicks import display_todays_picks
    from generic.pipeline_code.PickleModels import load_model

    start_time = time.time()

    logging.info("=" * 50)
    logging.info("CBB SPREAD MODEL - DAILY PREDICTIONS")
    logging.info("=" * 50)

    try:
        # Load model
        logging.info("Loading pre-trained model...")
        model = load_model(MODEL_CONFIG["model_path"])

        if model is None:
            logging.error(f"Could not load model from {MODEL_CONFIG['model_path']}")
            return {"status": "failed", "error": "Model not found"}

        # Connect and load data
        logging.info("Connecting to database...")
        engine = get_engine('CBB')

        logging.info("Loading game data from CBB.vw_FullModel...")
        raw_games = load_home_game_data(engine)

        logging.info("Building model dataframe with rolling features...")
        df_model = build_model_dataframe(
            game_df=raw_games,
            rolling_stats=MODEL_CONFIG["rolling_stats"],
            windows=MODEL_CONFIG["roll_windows"],
            target=MODEL_CONFIG["target_column"]
        )

        # Filter to future games
        now_utc = pd.Timestamp.utcnow()
        df_future = df_model[df_model["startDate"] >= now_utc]

        if df_future.empty:
            logging.info("No upcoming games found.")
            return {"status": "success", "message": "No upcoming games", "picks_count": 0}

        logging.info(f"Found {len(df_future)} upcoming games")

        # Generate predictions
        logging.info("Generating predictions...")
        X_pred = df_future[MODEL_CONFIG["features"]]
        preds = model.predict(X_pred)

        df_out = pd.concat(
            [df_future.reset_index(drop=True), preds.reset_index(drop=True)],
            axis=1
        )

        # Get picks meeting threshold
        logging.info(f"Applying thresholds: Lower={MODEL_CONFIG['lower_threshold']:.1%}, Upper={MODEL_CONFIG['upper_threshold']:.1%}")

        picks_df = display_todays_picks(
            df_out,
            MODEL_CONFIG["lower_threshold"],
            MODEL_CONFIG["upper_threshold"],
            engine
        )

        if picks_df is None or picks_df.empty:
            logging.info("No picks meet threshold criteria today.")
            return {
                "status": "success",
                "message": "No picks meet thresholds",
                "picks_count": 0,
                "games_analyzed": len(df_future)
            }

        # Save to database
        logging.info(f"Saving {len(picks_df)} picks to database...")
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

        elapsed = time.time() - start_time

        logging.info("=" * 50)
        logging.info(f"COMPLETED: {len(picks_df)} picks generated in {elapsed:.1f}s")
        logging.info("=" * 50)

        return {
            "status": "success",
            "picks_count": len(picks_df),
            "games_analyzed": len(df_future),
            "elapsed": round(elapsed, 2),
        }

    except Exception as e:
        logging.exception("Spread model execution failed")
        return {"status": "failed", "error": str(e)}


# =============================================================================
# FUNCTION 1: CBB PIPELINE TIMER
# =============================================================================

@app.timer_trigger(
    schedule="0 0 6 * * *",  # 6:00 AM UTC daily
    arg_name="timer",
    run_on_startup=False
)
def cbb_pipeline_timer(timer: func.TimerRequest) -> None:
    """
    Timer-triggered CBB Pipeline.
    Runs daily at 6:00 AM UTC.
    """
    logging.info("CBB Pipeline Timer triggered")

    if timer.past_due:
        logging.warning("Timer is past due!")

    result = run_cbb_pipeline()

    if result["status"] == "success":
        logging.info(f"Pipeline completed successfully in {result['total_elapsed']}s")
    elif result["status"] == "partial_failure":
        logging.warning(f"Pipeline completed with {result['failed_count']} failures")
    else:
        logging.error(f"Pipeline failed: {result.get('error', 'Unknown error')}")


# =============================================================================
# FUNCTION 2: CBB SPREAD MODEL TIMER
# =============================================================================

@app.timer_trigger(
    schedule="0 30 6 * * *",  # 6:30 AM UTC daily (30 min after pipeline)
    arg_name="timer",
    run_on_startup=False
)
def cbb_spread_model_timer(timer: func.TimerRequest) -> None:
    """
    Timer-triggered CBB Spread Model.
    Runs daily at 6:30 AM UTC (after pipeline completes).
    """
    logging.info("CBB Spread Model Timer triggered")

    if timer.past_due:
        logging.warning("Timer is past due!")

    result = run_spread_model()

    if result["status"] == "success":
        logging.info(f"Model completed: {result.get('picks_count', 0)} picks generated")
    else:
        logging.error(f"Model failed: {result.get('error', 'Unknown error')}")
