"""
Test script for CBB Pipeline Cleaners
"""
import sys
import os
from datetime import date, timedelta

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sports.cbb.pipeline_v2.src.extractors.team_extractor import TeamExtractor
from sports.cbb.pipeline_v2.src.extractors.game_extractor import GameExtractor
from sports.cbb.pipeline_v2.src.extractors.player_extractor import PlayerExtractor
from sports.cbb.pipeline_v2.src.cleaners.team_cleaners import clean_team_data
from sports.cbb.pipeline_v2.src.cleaners.game_cleaners import clean_game_data
from sports.cbb.pipeline_v2.src.cleaners.player_cleaners import clean_player_roster_data


def test_team_cleaner():
    """Test TeamCleaner"""
    print("\n=== Testing Team Cleaner ===")
    try:
        # Extract
        extractor = TeamExtractor()
        raw_df = extractor.extract_team_data(2026)
        print(f"Extracted {len(raw_df)} teams")
        
        # Clean
        cleaned_df = clean_team_data(raw_df)
        
        print(f"✓ Successfully cleaned teams")
        print(f"  Final columns: {list(cleaned_df.columns)}")
        print(f"\n  Sample cleaned data:")
        print(cleaned_df.head(2).to_string())
        
        return True
    except Exception as e:
        print(f"✗ Team Cleaner failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_game_cleaner():
    """Test GameCleaner"""
    print("\n=== Testing Game Cleaner ===")
    try:
        # Extract
        extractor = GameExtractor()
        end_date = date.today()
        start_date = end_date - timedelta(days=7)
        
        raw_df = extractor.extract_game_data(start_date, end_date)
        print(f"Extracted {len(raw_df)} games")
        
        # Clean
        cleaned_df = clean_game_data(raw_df)
        
        print(f"✓ Successfully cleaned games")
        print(f"  Final columns: {list(cleaned_df.columns)}")
        if len(cleaned_df) > 0:
            print(f"\n  Sample cleaned data:")
            print(cleaned_df.head(2).to_string())
        
        return True
    except Exception as e:
        print(f"✗ Game Cleaner failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_player_cleaner():
    """Test PlayerCleaner"""
    print("\n=== Testing Player Cleaner ===")
    try:
        # Extract
        extractor = PlayerExtractor()
        raw_df = extractor.extract_roster_data(2026)
        print(f"Extracted {len(raw_df)} rosters")
        
        # Clean
        cleaned_df = clean_player_roster_data(raw_df)
        
        print(f"✓ Successfully cleaned players")
        print(f"  Final columns: {list(cleaned_df.columns)}")
        print(f"\n  Sample cleaned data:")
        print(cleaned_df.head(2).to_string())
        
        return True
    except Exception as e:
        print(f"✗ Player Cleaner failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Testing CBB Pipeline Cleaners...")
    
    results = {
        "TeamCleaner": test_team_cleaner(),
        "GameCleaner": test_game_cleaner(),
        "PlayerCleaner": test_player_cleaner(),
    }
    
    print("\n" + "="*50)
    print("SUMMARY:")
    for cleaner, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"  {cleaner}: {status}")
    
    all_passed = all(results.values())
    print("="*50)
    print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    
    sys.exit(0 if all_passed else 1)
