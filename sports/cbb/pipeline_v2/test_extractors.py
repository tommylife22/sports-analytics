"""
Test script for CBB Pipeline Extractors
"""
import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sports.cbb.pipeline_v2.src.extractors.team_extractor import TeamExtractor
from sports.cbb.pipeline_v2.src.extractors.game_extractor import GameExtractor
from sports.cbb.pipeline_v2.src.extractors.player_extractor import PlayerExtractor
from datetime import date, timedelta


def test_team_extractor():
    """Test TeamExtractor"""
    print("\n=== Testing TeamExtractor ===")
    try:
        extractor = TeamExtractor()
        season = 2026
        
        # Extract team data
        print(f"Extracting teams for season {season}...")
        df = extractor.extract_team_data(season)
        
        print(f"✓ Successfully extracted {len(df)} teams")
        print(f"  Columns: {list(df.columns)}")
        print(f"\n  Sample team data:")
        print(df.head(2).to_string())
        
        return True
    except Exception as e:
        print(f"✗ TeamExtractor failed: {e}")
        return False


def test_game_extractor():
    """Test GameExtractor"""
    print("\n=== Testing GameExtractor ===")
    try:
        extractor = GameExtractor()
        end_date = date.today()
        start_date = end_date - timedelta(days=7)
        
        # Extract game data
        print(f"Extracting games from {start_date} to {end_date}...")
        df = extractor.extract_game_data(start_date, end_date)
        
        print(f"✓ Successfully extracted {len(df)} games")
        print(f"  Columns: {list(df.columns)}")
        if len(df) > 0:
            print(f"\n  Sample game data:")
            print(df.head(2).to_string())
        else:
            print("  (No games in this date range)")
        
        return True
    except Exception as e:
        print(f"✗ GameExtractor failed: {e}")
        return False


def test_player_extractor():
    """Test PlayerExtractor"""
    print("\n=== Testing PlayerExtractor ===")
    try:
        extractor = PlayerExtractor()
        season = 2026
        
        # Extract roster data
        print(f"Extracting rosters for season {season}...")
        df = extractor.extract_roster_data(season)
        
        print(f"✓ Successfully extracted roster data with {len(df)} rows")
        print(f"  Columns: {list(df.columns)}")
        print(f"\n  Sample roster data:")
        print(df.head(2).to_string())
        
        return True
    except Exception as e:
        print(f"✗ PlayerExtractor failed: {e}")
        return False


if __name__ == "__main__":
    print("Testing CBB Pipeline Extractors...")
    
    results = {
        "TeamExtractor": test_team_extractor(),
        "GameExtractor": test_game_extractor(),
        "PlayerExtractor": test_player_extractor(),
    }
    
    print("\n" + "="*50)
    print("SUMMARY:")
    for extractor, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"  {extractor}: {status}")
    
    all_passed = all(results.values())
    print("="*50)
    print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    
    sys.exit(0 if all_passed else 1)
