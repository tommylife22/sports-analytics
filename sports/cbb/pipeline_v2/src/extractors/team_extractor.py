"""
Team Data Extractor
Handles extraction of team information from CBB API
"""
import pandas as pd
from .base_extractor import BaseExtractor


class TeamExtractor(BaseExtractor):
    """Extractor for team information"""
    
    def extract_teams(self, season):
        """
        Extract team information for a given season
        
        Args:
            season (int): Season year
            
        Returns:
            list: List of team data dictionaries
        """
        params = {"season": season}
        return self._make_request("/teams", params=params)
    
    def extract_team_data(self, season):
        """
        Extract and return team data as DataFrame
        
        Args:
            season (int): Season year
            
        Returns:
            pd.DataFrame: Team information
        """
        teams = self.extract_teams(season)
        return pd.DataFrame(teams)
