"""
MLB Betting Odds Scraper
Scrapes MLB betting odds from SportsbookReview for moneylines, spreads, and totals
"""
import requests
import json
import re
from datetime import datetime, timedelta
import pandas as pd
from bs4 import BeautifulSoup
import time


class MLBOddsScraper:
    """
    Scraper for MLB betting odds from SportsbookReview
    """

    BASE_URL = "https://www.sportsbookreview.com/betting-odds/mlb-baseball"

    # URL patterns for different bet types
    BET_TYPES = {
        'moneyline': '/betting-odds/mlb-baseball/?date={}',
        'spread': '/betting-odds/mlb-baseball/pointspread/full-game/?date={}',
        'total': '/betting-odds/mlb-baseball/totals/full-game/?date={}'
    }

    def __init__(self, delay=1.0):
        """
        Initialize scraper

        Args:
            delay (float): Delay between requests in seconds (be respectful)
        """
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def _fetch_page(self, url):
        """
        Fetch a page with error handling

        Args:
            url (str): URL to fetch

        Returns:
            str: Page HTML content
        """
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            time.sleep(self.delay)  # Be respectful
            return response.text
        except requests.RequestException as e:
            print(f"  ⚠ Error fetching {url}: {e}")
            return None

    def _extract_json_data(self, html):
        """
        Extract embedded JSON data from Next.js page

        Args:
            html (str): Page HTML

        Returns:
            dict: Extracted JSON data
        """
        if not html:
            return None

        soup = BeautifulSoup(html, 'html.parser')

        # Find the Next.js data script tag
        script_tag = soup.find('script', id='__NEXT_DATA__')

        if not script_tag:
            print("  ⚠ Could not find __NEXT_DATA__ script tag")
            return None

        try:
            data = json.loads(script_tag.string)
            return data
        except json.JSONDecodeError as e:
            print(f"  ⚠ Error parsing JSON: {e}")
            return None

    def _parse_odds_data(self, json_data, bet_type, date):
        """
        Parse odds data from JSON

        Args:
            json_data (dict): Extracted JSON data
            bet_type (str): Type of bet (moneyline, spread, total)
            date (str): Date of games

        Returns:
            list: List of game odds dictionaries
        """
        if not json_data:
            return []

        try:
            # Navigate to the odds data
            page_props = json_data.get('props', {}).get('pageProps', {})
            odds_tables = page_props.get('oddsTables', [])

            if not odds_tables:
                print(f"  ⚠ No odds tables found for {date}")
                return []

            # Get the MLB league data (usually first table)
            mlb_table = odds_tables[0]

            # Check for gameRows in oddsTableModel (newer structure)
            odds_table_model = mlb_table.get('oddsTableModel', {})
            game_rows = odds_table_model.get('gameRows', [])

            # Fallback to old structure if needed
            if not game_rows:
                game_rows = mlb_table.get('gameRows', [])

            odds_list = []

            for game in game_rows:
                if not game:
                    continue

                game_view = game.get('gameView', {})
                if not game_view:
                    continue

                odds_views = game.get('oddsViews', [])

                # Extract game info
                game_id = game_view.get('gameId')
                away_team_data = game_view.get('awayTeam')
                home_team_data = game_view.get('homeTeam')

                away_team = away_team_data.get('name') if away_team_data else None
                home_team = home_team_data.get('name') if home_team_data else None
                game_time = game_view.get('startDate')

                if not away_team or not home_team:
                    continue

                # Process odds from each sportsbook
                for odds_view in odds_views:
                    if not odds_view:
                        continue  # Skip if odds_view is None

                    current_line = odds_view.get('currentLine')
                    if not current_line:
                        continue  # Skip if no current line data

                    sportsbook = odds_view.get('sportsbook', 'unknown')

                    # Extract odds based on bet type
                    if bet_type == 'moneyline':
                        away_odds = current_line.get('awayOdds')
                        home_odds = current_line.get('homeOdds')

                        odds_list.append({
                            'date': date,
                            'game_id': game_id,
                            'game_time': game_time,
                            'away_team': away_team,
                            'home_team': home_team,
                            'sportsbook': sportsbook,
                            'bet_type': 'moneyline',
                            'away_line': away_odds,
                            'home_line': home_odds,
                        })

                    elif bet_type == 'spread':
                        away_spread = current_line.get('awaySpread')
                        home_spread = current_line.get('homeSpread')
                        away_odds = current_line.get('awayOdds')
                        home_odds = current_line.get('homeOdds')

                        odds_list.append({
                            'date': date,
                            'game_id': game_id,
                            'game_time': game_time,
                            'away_team': away_team,
                            'home_team': home_team,
                            'sportsbook': sportsbook,
                            'bet_type': 'spread',
                            'away_spread': away_spread,
                            'away_odds': away_odds,
                            'home_spread': home_spread,
                            'home_odds': home_odds,
                        })

                    elif bet_type == 'total':
                        total = current_line.get('total')
                        over_odds = current_line.get('overOdds')
                        under_odds = current_line.get('underOdds')

                        odds_list.append({
                            'date': date,
                            'game_id': game_id,
                            'game_time': game_time,
                            'away_team': away_team,
                            'home_team': home_team,
                            'sportsbook': sportsbook,
                            'bet_type': 'total',
                            'total': total,
                            'over_odds': over_odds,
                            'under_odds': under_odds,
                        })

            return odds_list

        except Exception as e:
            print(f"  ⚠ Error parsing odds data: {e}")
            return []

    def scrape_odds_for_date(self, date, bet_type='moneyline'):
        """
        Scrape odds for a specific date and bet type

        Args:
            date (str): Date in YYYY-MM-DD format
            bet_type (str): Type of bet (moneyline, spread, total)

        Returns:
            list: List of odds dictionaries
        """
        if bet_type not in self.BET_TYPES:
            raise ValueError(f"Invalid bet_type. Must be one of: {list(self.BET_TYPES.keys())}")

        # Build URL
        url = f"https://www.sportsbookreview.com{self.BET_TYPES[bet_type].format(date)}"

        print(f"  Fetching {bet_type} odds for {date}...")

        # Fetch page
        html = self._fetch_page(url)
        if not html:
            return []

        # Extract JSON data
        json_data = self._extract_json_data(html)

        # Parse odds
        odds_list = self._parse_odds_data(json_data, bet_type, date)

        print(f"  ✓ Found {len(odds_list)} odds records")
        return odds_list

    def scrape_all_bet_types_for_date(self, date):
        """
        Scrape all bet types (moneyline, spread, total) for a specific date

        Args:
            date (str): Date in YYYY-MM-DD format

        Returns:
            dict: Dictionary with keys 'moneyline', 'spread', 'total' containing DataFrames
        """
        print(f"\n{'='*60}")
        print(f"SCRAPING ODDS FOR {date}")
        print(f"{'='*60}")

        results = {}

        for bet_type in ['moneyline', 'spread', 'total']:
            odds_list = self.scrape_odds_for_date(date, bet_type)
            results[bet_type] = pd.DataFrame(odds_list)

        # Summary
        print(f"\n{'='*60}")
        print("SCRAPING SUMMARY")
        print(f"{'='*60}")
        print(f"Moneyline records: {len(results['moneyline'])}")
        print(f"Spread records:    {len(results['spread'])}")
        print(f"Total records:     {len(results['total'])}")
        print(f"{'='*60}")

        return results

    def scrape_date_range(self, start_date, end_date, bet_types=None):
        """
        Scrape odds for a date range

        Args:
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            bet_types (list): List of bet types to scrape (default: all)

        Returns:
            dict: Dictionary with bet types as keys and DataFrames as values
        """
        if bet_types is None:
            bet_types = ['moneyline', 'spread', 'total']

        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')

        all_results = {bet_type: [] for bet_type in bet_types}

        current_date = start
        while current_date <= end:
            date_str = current_date.strftime('%Y-%m-%d')

            for bet_type in bet_types:
                odds_list = self.scrape_odds_for_date(date_str, bet_type)
                all_results[bet_type].extend(odds_list)

            current_date += timedelta(days=1)

        # Convert to DataFrames
        results_df = {
            bet_type: pd.DataFrame(odds_list)
            for bet_type, odds_list in all_results.items()
        }

        # Summary
        print(f"\n{'='*60}")
        print(f"SCRAPING COMPLETE: {start_date} to {end_date}")
        print(f"{'='*60}")
        for bet_type, df in results_df.items():
            print(f"{bet_type.capitalize()}: {len(df)} records")
        print(f"{'='*60}")

        return results_df


def scrape_mlb_odds(date, bet_types=None):
    """
    Convenience function to scrape MLB odds for a single date

    Args:
        date (str): Date in YYYY-MM-DD format
        bet_types (list): List of bet types to scrape (default: all)

    Returns:
        dict: Dictionary with bet types as keys and DataFrames as values
    """
    if bet_types is None:
        bet_types = ['moneyline', 'spread', 'total']

    scraper = MLBOddsScraper()

    if len(bet_types) == 1:
        odds_list = scraper.scrape_odds_for_date(date, bet_types[0])
        return {bet_types[0]: pd.DataFrame(odds_list)}
    else:
        return scraper.scrape_all_bet_types_for_date(date)


def scrape_mlb_odds_range(start_date, end_date, bet_types=None):
    """
    Convenience function to scrape MLB odds for a date range

    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        bet_types (list): List of bet types to scrape (default: all)

    Returns:
        dict: Dictionary with bet types as keys and DataFrames as values
    """
    scraper = MLBOddsScraper()
    return scraper.scrape_date_range(start_date, end_date, bet_types)
