#!/usr/bin/env python3
"""
CBB Pipeline - Daily Data Update
Loads team info, player info, games, boxscores, and betting lines
"""
import sys
import os
import argparse
import time
from datetime import date, timedelta

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from generic.db import get_engine
from sports.cbb.pipeline.tables.TeamInfo import loadTeamInfo
from sports.cbb.pipeline.tables.PlayerInfo import loadPlayerInfo
from sports.cbb.pipeline.tables.ConferenceInfo import loadConferenceInfo
from sports.cbb.pipeline.tables.VenueInfo import loadVenueInfo
from sports.cbb.pipeline.tables.GameInfo import loadGameInfo
from sports.cbb.pipeline.tables.GameBoxscoreTeam import loadGameBoxscoreTeam
from sports.cbb.pipeline.tables.GameBoxscorePlayer import loadGameBoxscorePlayer
from sports.cbb.pipeline.tables.GameLines import loadGameLines


def run_step(name, func, *args):
    """Run a pipeline step with timing and status output"""
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    start = time.time()
    try:
        func(*args)
        elapsed = time.time() - start
        print(f"  Done ({elapsed:.1f}s)")
        return True, elapsed
    except Exception as e:
        elapsed = time.time() - start
        print(f"  FAILED: {e}")
        return False, elapsed


def main():
    parser = argparse.ArgumentParser(description='Run CBB daily data pipeline')
    parser.add_argument('--season', type=int, default=2025, help='Season year (default: 2025)')
    parser.add_argument('--days-back', type=int, default=7, help='Days to look back (default: 7)')
    parser.add_argument('--days-ahead', type=int, default=1, help='Days to look ahead (default: 1)')
    parser.add_argument('--skip-info', action='store_true', help='Skip team/player/conference/venue info')
    parser.add_argument('--skip-games', action='store_true', help='Skip game data')
    parser.add_argument('--skip-boxscores', action='store_true', help='Skip boxscore data')
    parser.add_argument('--skip-lines', action='store_true', help='Skip betting lines')
    args = parser.parse_args()

    # Date range
    start_date = date.today() - timedelta(days=args.days_back)
    end_date = date.today() + timedelta(days=args.days_ahead)

    # Header
    print("\n" + "#" * 60)
    print("#  CBB PIPELINE - DAILY UPDATE")
    print("#" * 60)
    print(f"  Season:     {args.season}")
    print(f"  Date range: {start_date} to {end_date}")
    print("#" * 60)

    engine = get_engine('CBB')
    results = []
    total_start = time.time()

    # --- INFO TABLES ---
    if not args.skip_info:
        results.append(("TeamInfo", *run_step("Loading TeamInfo", loadTeamInfo, engine, args.season)))
        results.append(("PlayerInfo", *run_step("Loading PlayerInfo", loadPlayerInfo, engine, args.season)))
        results.append(("ConferenceInfo", *run_step("Loading ConferenceInfo", loadConferenceInfo, engine)))
        results.append(("VenueInfo", *run_step("Loading VenueInfo", loadVenueInfo, engine)))

    # --- GAME TABLES ---
    if not args.skip_games:
        results.append(("GameInfo", *run_step("Loading GameInfo", loadGameInfo, engine, start_date, end_date)))

    if not args.skip_boxscores:
        results.append(("GameBoxscoreTeam", *run_step("Loading GameBoxscoreTeam", loadGameBoxscoreTeam, engine, start_date, end_date)))
        results.append(("GameBoxscorePlayer", *run_step("Loading GameBoxscorePlayer", loadGameBoxscorePlayer, engine, start_date, end_date)))

    if not args.skip_lines:
        results.append(("GameLines", *run_step("Loading GameLines", loadGameLines, engine, start_date, end_date)))

    # --- SUMMARY ---
    total_elapsed = time.time() - total_start
    print("\n" + "=" * 60)
    print("  PIPELINE SUMMARY")
    print("=" * 60)
    for name, success, elapsed in results:
        status = "OK" if success else "FAILED"
        print(f"  {name:<25} {status:<8} {elapsed:>6.1f}s")
    print("-" * 60)
    print(f"  {'TOTAL':<25} {'':<8} {total_elapsed:>6.1f}s")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()