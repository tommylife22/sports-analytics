"""
Game Data Extractor
Handles extraction of game information and boxscore data from CBB API
"""
import pandas as pd
from datetime import datetime
from .base_extractor import BaseExtractor


class GameExtractor(BaseExtractor):
    """Extractor for game and boxscore information"""
    
    def extract_games(self, start_date, end_date):
        """
        Extract games for a date range
        
        Args:
            start_date (date or str): Start date (YYYY-MM-DD)
            end_date (date or str): End date (YYYY-MM-DD)
            
        Returns:
            list: List of game data dictionaries
        """
        params = {
            "startDateRange": str(start_date),
            "endDateRange": str(end_date)
        }
        return self._make_request("/games", params=params)
    
    def extract_game_data(self, start_date, end_date):
        """
        Extract and return game data as DataFrame
        
        Args:
            start_date (date or str): Start date
            end_date (date or str): End date
            
        Returns:
            pd.DataFrame: Game information
        """
        games = self.extract_games(start_date, end_date)
        return pd.DataFrame(games)
    
    def extract_team_boxscores(self, start_date, end_date):
        """
        Extract team boxscore data for games
        
        Args:
            start_date (date or str): Start date
            end_date (date or str): End date
            
        Returns:
            list: List of team boxscore dictionaries
        """
        params = {
            "startDateRange": str(start_date),
            "endDateRange": str(end_date)
        }
        return self._make_request("/boxscores", params=params)
    
    def extract_player_boxscores(self, start_date, end_date):
        """
        Extract player boxscore data for games
        
        Args:
            start_date (date or str): Start date
            end_date (date or str): End date
            
        Returns:
            list: List of player boxscore dictionaries
        """
        params = {
            "startDateRange": str(start_date),
            "endDateRange": str(end_date)
        }
        return self._make_request("/player-boxscores", params=params)
