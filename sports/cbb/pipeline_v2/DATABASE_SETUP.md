# CBB Pipeline V2 Database Setup

This guide walks through setting up the new `cbb` database with all pipeline tables.

## Prerequisites

1. SQL Server with Azure connection configured
2. `cbb` database already created
3. `.env` file with connection details
4. Python environment with dependencies installed

## Step 1: Verify Database Exists

Make sure the `cbb` database exists on your SQL Server:

```sql
SELECT name FROM sys.databases WHERE name = 'cbb';
```

If it doesn't exist, create it:

```sql
CREATE DATABASE cbb;
```

## Step 2: Create Schema and Tables

Run the setup script to create all tables in the `dbo` schema:

```bash
python sports/cbb/pipeline_v2/setup_database.py
```

This will:
- ✓ Connect to `cbb` database
- ✓ Execute the schema script
- ✓ Create all 8 tables in `dbo` schema with relationships
- ✓ Create indexes for performance
- ✓ Verify all tables were created

### Verify Tables Created

```sql
SELECT TABLE_NAME 
FROM INFORMATION_SCHEMA.TABLES 
WHERE TABLE_SCHEMA = 'dbo' AND TABLE_CATALOG = 'cbb'
ORDER BY TABLE_NAME;
```

Expected tables:
- `ConferenceInfo` - Conference information
- `VenueInfo` - Venue/arena information
- `TeamInfo` - Team information
- `PlayerInfo` - Player rosters
- `GameInfo` - Game data
- `GameBoxscoreTeam` - Team game statistics
- `GameBoxscorePlayer` - Player game statistics
- `GameLines` - Betting lines

## Step 3: (Optional) Migrate Old Data

If you want to copy existing data from the old `cbb` database:

```bash
python sports/cbb/pipeline_v2/setup_database.py --migrate
```

This will:
- ✓ Read from old `cbb` database tables
- ✓ Write to new `cbb` database tables
- ✓ Handle foreign key relationships
- ✓ Skip empty tables

## Database Schema

### ConferenceInfo
```
conferenceId (PK)      - Unique identifier
sourceId               - External API ID
name                   - Conference name
abbreviation           - Conference abbreviation
shortName              - Short conference name
```

### VenueInfo
```
venueId (PK)           - Unique identifier
sourceId               - External API ID
name                   - Venue name
city                   - City
state                  - State
country                - Country
latitude               - Latitude
longitude              - Longitude
```

### TeamInfo
```
teamId (PK)            - Team unique identifier
sourceId               - External source ID
school                 - School name
mascot                 - Team mascot
abbreviation           - Team abbreviation
displayName            - Display name
conferenceId (FK)      - Conference ID
conference             - Conference name
currentVenueId (FK)    - Current venue ID
currentVenue           - Current venue name
currentCity            - Current city
currentState           - Current state
```

### PlayerInfo
```
playerId (PK)          - Player unique identifier
sourceId               - External source ID
teamId (FK)            - Team ID
season                 - Season year
name                   - Full name
firstName              - First name
lastName               - Last name
jersey                 - Jersey number
position               - Position
height                 - Height (inches)
weight                 - Weight (lbs)
hometownCity           - Hometown city
hometownState          - Hometown state
startSeason            - First season
endSeason              - Last season
```

### GameInfo
```
gameId (PK)            - Game unique identifier
sourceId               - External source ID
season                 - Season year
seasonType             - Regular/Tournament/etc
startDate              - Game date/time
homeTeamId (FK)        - Home team ID
awayTeamId (FK)        - Away team ID
homePoints             - Home team score
awayPoints             - Away team score
status                 - Final/Live/Postponed/etc
neutralSite            - Boolean
conferenceGame         - Boolean
tournament             - Tournament name
venueId (FK)           - Venue ID
venue                  - Venue name
city                   - City
state                  - State
```

### GameBoxscoreTeam
```
boxscoreId (PK)        - Auto-incrementing ID
gameId (FK)            - Game ID
teamId (FK)            - Team ID
season                 - Season year
points                 - Points scored
fieldGoalsMade         - FG made
fieldGoalsAttempted    - FG attempted
threePointersMade      - 3P made
threePointersAttempted - 3P attempted
freeThrowsMade         - FT made
freeThrowsAttempted    - FT attempted
rebounds               - Rebounds
assists                - Assists
turnovers              - Turnovers
steals                 - Steals
blocks                 - Blocks
fouls                  - Fouls
```

### GameBoxscorePlayer
```
boxscoreId (PK)        - Auto-incrementing ID
gameId (FK)            - Game ID
playerId (FK)          - Player ID
teamId (FK)            - Team ID
season                 - Season year
points                 - Points scored
fieldGoalsMade         - FG made
fieldGoalsAttempted    - FG attempted
threePointersMade      - 3P made
threePointersAttempted - 3P attempted
freeThrowsMade         - FT made
freeThrowsAttempted    - FT attempted
rebounds               - Rebounds
assists                - Assists
turnovers              - Turnovers
steals                 - Steals
blocks                 - Blocks
fouls                  - Fouls
minutesPlayed          - Minutes played
```

### GameLines
```
linesId (PK)           - Auto-incrementing ID
gameId (FK)            - Game ID
season                 - Season year
spread                 - Point spread
overUnder              - Over/under total
homeMoneyline          - Home moneyline odds
awayMoneyline          - Away moneyline odds
homeOdds               - Home odds
awayOdds               - Away odds
```

## Indexes Created

Performance indexes on:
- `TeamInfo.conferenceId`
- `PlayerInfo.teamId`, `PlayerInfo.season`
- `GameInfo.season`, `GameInfo.homeTeamId`, `GameInfo.awayTeamId`, `GameInfo.startDate`
- `GameBoxscoreTeam.gameId`, `GameBoxscoreTeam.teamId`
- `GameBoxscorePlayer.gameId`, `GameBoxscorePlayer.playerId`, `GameBoxscorePlayer.teamId`
- `GameLines.gameId`

## Troubleshooting

### Connection Error
```
Error: mssql+pyodbc connection failed
```
**Solution:** Check `.env` file for correct:
- `SPORTS_SERVER_NAME` - SQL Server name
- `AZURE_USERNAME` - Username
- `AZURE_PASSWORD` - Password
- `CBB_DB` - Should be "cbb"

### Table Already Exists
```
Error: Table already exists
```
**Solution:** Either drop tables first or use `--migrate` flag to append

### Foreign Key Constraint Error During Migration
**Solution:** This means the parent table doesn't have required records. Create parent records first:
1. `ConferenceInfo`
2. `VenueInfo`
3. `TeamInfo`
4. Then migrate other tables

### Missing Driver
```
Error: pyodbc.Error: ('IM002', '[IM002]')
```
**Solution:** Install ODBC Driver for SQL Server:
```bash
# macOS
brew install freetds

# Windows
# Download from: https://learn.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server
```

## Next Steps

1. Run loaders to insert cleaned data
2. Monitor data quality using queries
3. Set up automated backups
4. Create views for common queries

## Rolling Back

To delete all tables in `dbo` schema:

```sql
-- Drop tables in reverse dependency order
DROP TABLE IF EXISTS [dbo].[GameLines];
DROP TABLE IF EXISTS [dbo].[GameBoxscorePlayer];
DROP TABLE IF EXISTS [dbo].[GameBoxscoreTeam];
DROP TABLE IF EXISTS [dbo].[PlayerInfo];
DROP TABLE IF EXISTS [dbo].[GameInfo];
DROP TABLE IF EXISTS [dbo].[TeamInfo];
DROP TABLE IF EXISTS [dbo].[VenueInfo];
DROP TABLE IF EXISTS [dbo].[ConferenceInfo];
```
