"""
Base Extractor Class
Provides common functionality for all CBB data extractors
"""
import os
import requests
from dotenv import load_dotenv

load_dotenv()

class BaseExtractor:
    """Base class for all CBB extractors"""
    
    def __init__(self):
        self.api_key = os.environ.get('CBB_API_KEY')
        self.base_url = "https://api.collegebasketballdata.com"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }
    
    def _make_request(self, endpoint, params=None, timeout=30):
        """
        Make API request with standard headers and error handling
        
        Args:
            endpoint (str): API endpoint
            params (dict): Query parameters
            timeout (int): Request timeout in seconds
            
        Returns:
            dict or list: JSON response from API
            
        Raises:
            requests.HTTPError: If request fails
        """
        url = f"{self.base_url}{endpoint}"
        resp = requests.get(url, params=params, headers=self.headers, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
