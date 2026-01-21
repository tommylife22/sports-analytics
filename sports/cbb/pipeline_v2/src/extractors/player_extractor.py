"""
Player Data Extractor
Handles extraction of player information from CBB API
"""
import pandas as pd
from .base_extractor import BaseExtractor


class PlayerExtractor(BaseExtractor):
    """Extractor for player information"""
    
    def extract_roster(self, season):
        """
        Extract team rosters for a given season
        
        Args:
            season (int): Season year
            
        Returns:
            list: List of team roster data dictionaries
        """
        params = {"season": season}
        return self._make_request("/teams/roster", params=params)
    
    def extract_roster_data(self, season):
        """
        Extract and return roster data as DataFrame
        
        Args:
            season (int): Season year
            
        Returns:
            pd.DataFrame: Player roster information
        """
        rosters = self.extract_roster(season)
        return pd.DataFrame(rosters)
