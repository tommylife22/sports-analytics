"""
MLB Data Pipeline
A modular pipeline for extracting, cleaning, and loading MLB data to Azure SQL Database
"""

from .extractors import (
    get_teams_by_season,
    get_all_players_by_season,
    get_games_by_season,
    get_team_boxscore,
    get_player_boxscore,
    get_all_boxscores_by_season,
)

from .extractors.odds_scraper import (
    scrape_mlb_odds,
    scrape_mlb_odds_range,
    MLBOddsScraper,
)

from .cleaners import (
    clean_duplicate_games,
    clean_teams_data,
    clean_players_data,
    clean_games_data,
    clean_team_boxscore_data,
    clean_player_boxscore_data,
)

from .orchestrators import (
    build_season_database,
    build_multi_season_database,
    build_boxscore_database,
    build_multi_season_boxscore_database,
)

from .orchestrators.odds_pipeline import (
    build_odds_database,
    build_odds_database_by_season,
    backfill_historical_odds,
    find_available_date_range,
)

from .orchestrators.daily_pipeline_update import (
    daily_pipeline_update,
    daily_update_with_retry,
    get_update_summary_message,
)

from .loaders import (
    load_teams_to_database,
    load_players_to_database,
    load_games_to_database,
    load_all_info_to_database,
    load_table_to_database,
    load_team_boxscore_to_database,
    load_player_boxscore_to_database,
    load_all_boxscores_to_database,
)

from .loaders.odds_loader import (
    load_odds_to_database,
    load_scraped_odds_to_database,
    prepare_odds_for_database,
)

from .utils import (
    parse_datetime,
    parse_date,
    get_primary_keys,
    get_data_columns,
)

__version__ = "2.1.0"
