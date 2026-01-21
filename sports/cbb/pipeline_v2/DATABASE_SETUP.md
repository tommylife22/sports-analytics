# CBB Pipeline V2 Database Setup

This guide walks through setting up the new `CBB_V2` database for the refactored CBB pipeline.

## Prerequisites

1. SQL Server with Azure connection configured
2. `.env` file with connection details
3. Python environment with dependencies installed

## Step 1: Create the Database

First, you need to create the `CBB_V2` database on your SQL Server (manual step):

### Option A: Using SQL Server Management Studio
1. Connect to your Azure SQL Server
2. Right-click "Databases" → "New Database"
3. Name it `CBB_V2`
4. Click "OK"

### Option B: Using SQL Query
```sql
CREATE DATABASE CBB_V2;
```

## Step 2: Create Schema and Tables

Run the setup script to create all tables:

```bash
python sports/cbb/pipeline_v2/setup_database.py
```

This will:
- ✓ Connect to `CBB_V2` database
- ✓ Execute the schema script
- ✓ Create all tables with relationships
- ✓ Create indexes for performance

### Verify Tables Created

```sql
SELECT TABLE_NAME 
FROM INFORMATION_SCHEMA.TABLES 
WHERE TABLE_SCHEMA = 'CBB_V2'
ORDER BY TABLE_NAME;
```

Expected tables:
- `Team` - Team information
- `Game` - Game data
- `Player` - Player rosters
- `TeamBoxscore` - Team game statistics
- `PlayerBoxscore` - Player game statistics
- `Conference` - Conference information

## Step 3: (Optional) Migrate Old Data

If you want to copy data from the old `CBB` database:

```bash
python sports/cbb/pipeline_v2/setup_database.py --migrate
```

This will:
- ✓ Read from old `CBB` database
- ✓ Write to new `CBB_V2` database
- ✓ Skip tables that don't exist in old database

## Step 4: Update .env

Add the new database name to your `.env` file:

```env
CBB_V2_DB=CBB_V2
```

## Database Schema

### Team Table
```
team_id (PK)           - Team unique identifier
source_id              - External source ID
school_name            - Official school name
team_abbr              - Team abbreviation
mascot                 - Team mascot
display_name           - Display name
conference_id          - Conference ID
conference_name        - Conference name
```

### Game Table
```
game_id (PK)           - Game unique identifier
source_id              - External source ID
season                 - Season year
season_type            - Regular, Tournament, etc.
game_date              - Game date/time
home_team_id (FK)      - Home team ID
away_team_id (FK)      - Away team ID
home_points            - Home team score
away_points            - Away team score
status                 - Final, Live, Postponed, etc.
neutral_site           - Boolean
conference_game        - Boolean
```

### Player Table
```
player_id (PK)         - Player unique identifier
source_id              - External source ID
team_id (FK)           - Team ID
season                 - Season year
name                   - Full name
first_name             - First name
last_name              - Last name
jersey_number          - Jersey #
position               - Guard, Forward, Center
height_inches          - Height in inches
weight_lbs             - Weight in pounds
start_season           - First season
end_season             - Last season
```

### Indexes

Performance indexes created on:
- `Game.season`, `Game.home_team_id`, `Game.away_team_id`, `Game.game_date`
- `Player.team_id + season`, `Player.season`
- `TeamBoxscore.game_id + team_id`, `TeamBoxscore.season`
- `PlayerBoxscore.game_id + player_id`, `PlayerBoxscore.season`, `PlayerBoxscore.team_id`

## Troubleshooting

### Connection Error
```
Error: mssql+pyodbc connection failed
```
**Solution:** Check `.env` file for correct server name, username, password

### Table Already Exists
```
Error: ProgrammingError: Table 'Team' already exists
```
**Solution:** The schema has already been created. Skip this step or drop tables first.

### Missing Driver
```
Error: pyodbc.Error: ('IM002', '[IM002]')
```
**Solution:** Install ODBC Driver for SQL Server:
```bash
# macOS
brew install freetds --with-odbc

# Windows
# Download from: https://learn.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server
```

## Next Steps

1. Run the loaders to insert cleaned data
2. Monitor data quality and relationships
3. Verify foreign key constraints are working
4. Set up backup strategy

## Rolling Back

To delete all tables in `CBB_V2`:

```sql
-- Drop foreign keys first
ALTER TABLE CBB_V2.PlayerBoxscore DROP CONSTRAINT FK_PlayerBoxscore_Game;
ALTER TABLE CBB_V2.PlayerBoxscore DROP CONSTRAINT FK_PlayerBoxscore_Player;
ALTER TABLE CBB_V2.PlayerBoxscore DROP CONSTRAINT FK_PlayerBoxscore_Team;
ALTER TABLE CBB_V2.TeamBoxscore DROP CONSTRAINT FK_TeamBoxscore_Game;
ALTER TABLE CBB_V2.TeamBoxscore DROP CONSTRAINT FK_TeamBoxscore_Team;
ALTER TABLE CBB_V2.Game DROP CONSTRAINT FK_Game_HomeTeam;
ALTER TABLE CBB_V2.Game DROP CONSTRAINT FK_Game_AwayTeam;
ALTER TABLE CBB_V2.Player DROP CONSTRAINT FK_Player_Team;

-- Drop tables
DROP TABLE CBB_V2.PlayerBoxscore;
DROP TABLE CBB_V2.TeamBoxscore;
DROP TABLE CBB_V2.Game;
DROP TABLE CBB_V2.Player;
DROP TABLE CBB_V2.Team;
DROP TABLE CBB_V2.Conference;
```
