import sys
import os

# Add project root to path for generic imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from generic.db import get_engine
from sports.cbb.pipeline.tables.ConferenceInfo              import loadConferenceInfo
from sports.cbb.pipeline.tables.GameBoxscorePlayer          import loadGameBoxscorePlayer
from sports.cbb.pipeline.tables.GameBoxscoreTeam            import loadGameBoxscoreTeam
from sports.cbb.pipeline.tables.GameInfo                    import loadGameInfo
from sports.cbb.pipeline.tables.GameLines                   import loadGameLines
from sports.cbb.pipeline.tables.TeamInfo                    import loadTeamInfo
from sports.cbb.pipeline.tables.VenueInfo                   import loadVenueInfo
from sports.cbb.pipeline.tables.PlayerInfo                  import loadPlayerInfo

from datetime import date, timedelta

def main():
    
    engine = get_engine('CBB')
    season = 2026
    startDate = date.today() - timedelta(days = 7)
    endDate = date.today() + timedelta(days = 1)

    loadTeamInfo(engine, season)
    print("TeamInfo Done")
    #loadPlayerInfo(engine, season)
    #print("PlayerInfo Done")
    loadConferenceInfo(engine)
    print("ConferenceInfo Done")
    loadVenueInfo(engine)
    print("VenueInfo Done")
    loadGameInfo(engine, startDate, endDate)
    print("GameInfo Done")
    loadGameBoxscoreTeam(engine, startDate, endDate)
    print("GameBoxscoreTeam Done")
    #loadGameBoxscorePlayer(engine, startDate, endDate)
    #print("GameBoxscorePlayer Done")
    loadGameLines(engine, startDate, endDate)
    print("GameLines Done")
    
if __name__ == "__main__":
    
    main()