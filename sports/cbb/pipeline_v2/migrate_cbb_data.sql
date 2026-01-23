-- ============================================
-- CBB Data Migration Script
-- ============================================
-- Migrates data from: nhlpipe-sqlsvr.CBB schema
-- Migrates to: cbb.dbo schema
--
-- Run this script in SSMS or Azure Data Studio
-- connected to the TARGET (cbb) database
-- ============================================

-- NOTE: Before running this script:
-- 1. Make sure the target tables exist (run cbb_v2_schema.sql first)
-- 2. Update the source server/database name if different
-- 3. Tables must be migrated in order due to foreign key constraints

-- Source database reference (adjust if needed)
-- Using 3-part naming: [database].[schema].[table]

-- ============================================
-- STEP 1: MIGRATE CONFERENCE INFO (No dependencies)
-- ============================================
PRINT 'Migrating ConferenceInfo...';

SET IDENTITY_INSERT [dbo].[ConferenceInfo] OFF;

INSERT INTO [dbo].[ConferenceInfo] (
    [conferenceId],
    [sourceId],
    [name],
    [abbreviation],
    [shortName],
    [insert_date],
    [update_date]
)
SELECT
    [conferenceId],
    [sourceId],
    [name],
    [abbreviation],
    [shortName],
    ISNULL([insert_date], GETDATE()),
    ISNULL([update_date], GETDATE())
FROM [nhlpipe-sqlsvr].[CBB].[ConferenceInfo]
WHERE [conferenceId] NOT IN (SELECT [conferenceId] FROM [dbo].[ConferenceInfo]);

PRINT 'ConferenceInfo migration complete.';

-- ============================================
-- STEP 2: MIGRATE VENUE INFO (No dependencies)
-- ============================================
PRINT 'Migrating VenueInfo...';

INSERT INTO [dbo].[VenueInfo] (
    [venueId],
    [sourceId],
    [name],
    [city],
    [state],
    [country],
    [latitude],
    [longitude],
    [insert_date],
    [update_date]
)
SELECT
    [venueId],
    [sourceId],
    [name],
    [city],
    [state],
    [country],
    [latitude],
    [longitude],
    ISNULL([insert_date], GETDATE()),
    ISNULL([update_date], GETDATE())
FROM [nhlpipe-sqlsvr].[CBB].[VenueInfo]
WHERE [venueId] NOT IN (SELECT [venueId] FROM [dbo].[VenueInfo]);

PRINT 'VenueInfo migration complete.';

-- ============================================
-- STEP 3: MIGRATE TEAM INFO (Depends on Conference, Venue)
-- ============================================
PRINT 'Migrating TeamInfo...';

INSERT INTO [dbo].[TeamInfo] (
    [teamId],
    [sourceId],
    [school],
    [mascot],
    [abbreviation],
    [displayName],
    [conferenceId],
    [conference],
    [currentVenueId],
    [currentVenue],
    [currentCity],
    [currentState],
    [insert_date],
    [update_date]
)
SELECT
    [teamId],
    [sourceId],
    [school],
    [mascot],
    [abbreviation],
    [displayName],
    [conferenceId],
    [conference],
    [currentVenueId],
    [currentVenue],
    [currentCity],
    [currentState],
    ISNULL([insert_date], GETDATE()),
    ISNULL([update_date], GETDATE())
FROM [nhlpipe-sqlsvr].[CBB].[TeamInfo]
WHERE [teamId] NOT IN (SELECT [teamId] FROM [dbo].[TeamInfo]);

PRINT 'TeamInfo migration complete.';

-- ============================================
-- STEP 4: MIGRATE PLAYER INFO (Depends on Team)
-- ============================================
PRINT 'Migrating PlayerInfo...';

INSERT INTO [dbo].[PlayerInfo] (
    [playerId],
    [sourceId],
    [teamId],
    [season],
    [name],
    [firstName],
    [lastName],
    [jersey],
    [position],
    [height],
    [weight],
    [hometownCity],
    [hometownState],
    [startSeason],
    [endSeason],
    [insert_date],
    [update_date]
)
SELECT
    [playerId],
    [sourceId],
    [teamId],
    [season],
    [name],
    [firstName],
    [lastName],
    [jersey],
    [position],
    [height],
    [weight],
    [hometownCity],
    [hometownState],
    [startSeason],
    [endSeason],
    ISNULL([insert_date], GETDATE()),
    ISNULL([update_date], GETDATE())
FROM [nhlpipe-sqlsvr].[CBB].[PlayerInfo]
WHERE [playerId] NOT IN (SELECT [playerId] FROM [dbo].[PlayerInfo]);

PRINT 'PlayerInfo migration complete.';

-- ============================================
-- STEP 5: MIGRATE GAME INFO (Depends on Team, Venue)
-- ============================================
PRINT 'Migrating GameInfo...';

INSERT INTO [dbo].[GameInfo] (
    [gameId],
    [sourceId],
    [season],
    [seasonType],
    [startDate],
    [homeTeamId],
    [awayTeamId],
    [homePoints],
    [awayPoints],
    [status],
    [neutralSite],
    [conferenceGame],
    [tournament],
    [venueId],
    [venue],
    [city],
    [state],
    [insert_date],
    [update_date]
)
SELECT
    [gameId],
    [sourceId],
    [season],
    [seasonType],
    [startDate],
    [homeTeamId],
    [awayTeamId],
    [homePoints],
    [awayPoints],
    [status],
    [neutralSite],
    [conferenceGame],
    [tournament],
    [venueId],
    [venue],
    [city],
    [state],
    ISNULL([insert_date], GETDATE()),
    ISNULL([update_date], GETDATE())
FROM [nhlpipe-sqlsvr].[CBB].[GameInfo]
WHERE [gameId] NOT IN (SELECT [gameId] FROM [dbo].[GameInfo]);

PRINT 'GameInfo migration complete.';

-- ============================================
-- STEP 6: MIGRATE GAME BOXSCORE TEAM (Depends on Game, Team)
-- ============================================
PRINT 'Migrating GameBoxscoreTeam...';

-- Note: boxscoreId is IDENTITY, so we don't insert it
INSERT INTO [dbo].[GameBoxscoreTeam] (
    [gameId],
    [teamId],
    [season],
    [points],
    [fieldGoalsMade],
    [fieldGoalsAttempted],
    [threePointersMade],
    [threePointersAttempted],
    [freeThrowsMade],
    [freeThrowsAttempted],
    [rebounds],
    [assists],
    [turnovers],
    [steals],
    [blocks],
    [fouls],
    [insert_date],
    [update_date]
)
SELECT
    src.[gameId],
    src.[teamId],
    src.[season],
    src.[points],
    src.[fieldGoalsMade],
    src.[fieldGoalsAttempted],
    src.[threePointersMade],
    src.[threePointersAttempted],
    src.[freeThrowsMade],
    src.[freeThrowsAttempted],
    src.[rebounds],
    src.[assists],
    src.[turnovers],
    src.[steals],
    src.[blocks],
    src.[fouls],
    ISNULL(src.[insert_date], GETDATE()),
    ISNULL(src.[update_date], GETDATE())
FROM [nhlpipe-sqlsvr].[CBB].[GameBoxscoreTeam] src
WHERE NOT EXISTS (
    SELECT 1 FROM [dbo].[GameBoxscoreTeam] tgt
    WHERE tgt.[gameId] = src.[gameId] AND tgt.[teamId] = src.[teamId]
);

PRINT 'GameBoxscoreTeam migration complete.';

-- ============================================
-- STEP 7: MIGRATE GAME BOXSCORE PLAYER (Depends on Game, Player, Team)
-- ============================================
PRINT 'Migrating GameBoxscorePlayer...';

-- Note: boxscoreId is IDENTITY, so we don't insert it
INSERT INTO [dbo].[GameBoxscorePlayer] (
    [gameId],
    [playerId],
    [teamId],
    [season],
    [points],
    [fieldGoalsMade],
    [fieldGoalsAttempted],
    [threePointersMade],
    [threePointersAttempted],
    [freeThrowsMade],
    [freeThrowsAttempted],
    [rebounds],
    [assists],
    [turnovers],
    [steals],
    [blocks],
    [fouls],
    [minutesPlayed],
    [insert_date],
    [update_date]
)
SELECT
    src.[gameId],
    src.[playerId],
    src.[teamId],
    src.[season],
    src.[points],
    src.[fieldGoalsMade],
    src.[fieldGoalsAttempted],
    src.[threePointersMade],
    src.[threePointersAttempted],
    src.[freeThrowsMade],
    src.[freeThrowsAttempted],
    src.[rebounds],
    src.[assists],
    src.[turnovers],
    src.[steals],
    src.[blocks],
    src.[fouls],
    src.[minutesPlayed],
    ISNULL(src.[insert_date], GETDATE()),
    ISNULL(src.[update_date], GETDATE())
FROM [nhlpipe-sqlsvr].[CBB].[GameBoxscorePlayer] src
WHERE NOT EXISTS (
    SELECT 1 FROM [dbo].[GameBoxscorePlayer] tgt
    WHERE tgt.[gameId] = src.[gameId] AND tgt.[playerId] = src.[playerId]
);

PRINT 'GameBoxscorePlayer migration complete.';

-- ============================================
-- STEP 8: MIGRATE GAME LINES (Depends on Game)
-- ============================================
PRINT 'Migrating GameLines...';

-- Note: linesId is IDENTITY, so we don't insert it
INSERT INTO [dbo].[GameLines] (
    [gameId],
    [season],
    [spread],
    [overUnder],
    [homeMoneyline],
    [awayMoneyline],
    [homeOdds],
    [awayOdds],
    [insert_date],
    [update_date]
)
SELECT
    src.[gameId],
    src.[season],
    src.[spread],
    src.[overUnder],
    src.[homeMoneyline],
    src.[awayMoneyline],
    src.[homeOdds],
    src.[awayOdds],
    ISNULL(src.[insert_date], GETDATE()),
    ISNULL(src.[update_date], GETDATE())
FROM [nhlpipe-sqlsvr].[CBB].[GameLines] src
WHERE NOT EXISTS (
    SELECT 1 FROM [dbo].[GameLines] tgt
    WHERE tgt.[gameId] = src.[gameId]
);

PRINT 'GameLines migration complete.';

-- ============================================
-- VERIFICATION QUERIES
-- ============================================
PRINT '';
PRINT '============================================';
PRINT 'MIGRATION SUMMARY';
PRINT '============================================';

SELECT 'ConferenceInfo' AS TableName, COUNT(*) AS RowCount FROM [dbo].[ConferenceInfo]
UNION ALL
SELECT 'VenueInfo', COUNT(*) FROM [dbo].[VenueInfo]
UNION ALL
SELECT 'TeamInfo', COUNT(*) FROM [dbo].[TeamInfo]
UNION ALL
SELECT 'PlayerInfo', COUNT(*) FROM [dbo].[PlayerInfo]
UNION ALL
SELECT 'GameInfo', COUNT(*) FROM [dbo].[GameInfo]
UNION ALL
SELECT 'GameBoxscoreTeam', COUNT(*) FROM [dbo].[GameBoxscoreTeam]
UNION ALL
SELECT 'GameBoxscorePlayer', COUNT(*) FROM [dbo].[GameBoxscorePlayer]
UNION ALL
SELECT 'GameLines', COUNT(*) FROM [dbo].[GameLines]
ORDER BY TableName;

PRINT '';
PRINT 'Migration complete!';
