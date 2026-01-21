-- CBB Pipeline V2 Database Schema
-- Migrates all tables from old pipeline to new dbo schema
-- Tables: ConferenceInfo, GameBoxscorePlayer, GameBoxscoreTeam, GameInfo, GameLines, PlayerInfo, TeamInfo, VenueInfo

-- ============================================
-- CONFERENCE INFO TABLE
-- ============================================
CREATE TABLE [dbo].[ConferenceInfo] (
    [conferenceId] VARCHAR(25) PRIMARY KEY NOT NULL,
    [sourceId] VARCHAR(50),
    [name] VARCHAR(255),
    [abbreviation] VARCHAR(10),
    [shortName] VARCHAR(50),
    [insert_date] DATETIME DEFAULT GETDATE(),
    [update_date] DATETIME DEFAULT GETDATE()
);

-- ============================================
-- VENUE INFO TABLE
-- ============================================
CREATE TABLE [dbo].[VenueInfo] (
    [venueId] VARCHAR(25) PRIMARY KEY NOT NULL,
    [sourceId] VARCHAR(255),
    [name] VARCHAR(255),
    [city] VARCHAR(100),
    [state] VARCHAR(50),
    [country] VARCHAR(100),
    [latitude] DECIMAL(10,6),
    [longitude] DECIMAL(10,6),
    [insert_date] DATETIME DEFAULT GETDATE(),
    [update_date] DATETIME DEFAULT GETDATE()
);

-- ============================================
-- TEAM INFO TABLE
-- ============================================
CREATE TABLE [dbo].[TeamInfo] (
    [teamId] VARCHAR(25) PRIMARY KEY NOT NULL,
    [sourceId] VARCHAR(50),
    [school] VARCHAR(255) NOT NULL,
    [mascot] VARCHAR(100),
    [abbreviation] VARCHAR(10),
    [displayName] VARCHAR(255),
    [conferenceId] VARCHAR(25),
    [conference] VARCHAR(100),
    [currentVenueId] VARCHAR(25),
    [currentVenue] VARCHAR(255),
    [currentCity] VARCHAR(100),
    [currentState] VARCHAR(50),
    [insert_date] DATETIME DEFAULT GETDATE(),
    [update_date] DATETIME DEFAULT GETDATE(),
    CONSTRAINT FK_TeamInfo_Conference FOREIGN KEY ([conferenceId]) REFERENCES [dbo].[ConferenceInfo]([conferenceId]),
    CONSTRAINT FK_TeamInfo_Venue FOREIGN KEY ([currentVenueId]) REFERENCES [dbo].[VenueInfo]([venueId])
);

-- ============================================
-- PLAYER INFO TABLE
-- ============================================
CREATE TABLE [dbo].[PlayerInfo] (
    [playerId] VARCHAR(25) PRIMARY KEY NOT NULL,
    [sourceId] VARCHAR(50),
    [teamId] VARCHAR(25) NOT NULL,
    [season] INT NOT NULL,
    [name] VARCHAR(255),
    [firstName] VARCHAR(100),
    [lastName] VARCHAR(100),
    [jersey] VARCHAR(10),
    [position] VARCHAR(50),
    [height] INT,
    [weight] INT,
    [hometownCity] VARCHAR(100),
    [hometownState] VARCHAR(50),
    [startSeason] INT,
    [endSeason] INT,
    [insert_date] DATETIME DEFAULT GETDATE(),
    [update_date] DATETIME DEFAULT GETDATE(),
    CONSTRAINT FK_PlayerInfo_Team FOREIGN KEY ([teamId]) REFERENCES [dbo].[TeamInfo]([teamId]),
    CONSTRAINT UC_PlayerInfo_Season UNIQUE ([playerId], [teamId], [season])
);

-- ============================================
-- GAME INFO TABLE
-- ============================================
CREATE TABLE [dbo].[GameInfo] (
    [gameId] VARCHAR(25) PRIMARY KEY NOT NULL,
    [sourceId] VARCHAR(50),
    [season] INT NOT NULL,
    [seasonType] VARCHAR(50),
    [startDate] DATETIME2,
    [homeTeamId] VARCHAR(25) NOT NULL,
    [awayTeamId] VARCHAR(25) NOT NULL,
    [homePoints] INT,
    [awayPoints] INT,
    [status] VARCHAR(50),
    [neutralSite] BIT,
    [conferenceGame] BIT,
    [tournament] VARCHAR(100),
    [venueId] VARCHAR(25),
    [venue] VARCHAR(255),
    [city] VARCHAR(100),
    [state] VARCHAR(50),
    [insert_date] DATETIME DEFAULT GETDATE(),
    [update_date] DATETIME DEFAULT GETDATE(),
    CONSTRAINT FK_GameInfo_HomeTeam FOREIGN KEY ([homeTeamId]) REFERENCES [dbo].[TeamInfo]([teamId]),
    CONSTRAINT FK_GameInfo_AwayTeam FOREIGN KEY ([awayTeamId]) REFERENCES [dbo].[TeamInfo]([teamId]),
    CONSTRAINT FK_GameInfo_Venue FOREIGN KEY ([venueId]) REFERENCES [dbo].[VenueInfo]([venueId])
);

-- ============================================
-- GAME BOXSCORE TEAM TABLE
-- ============================================
CREATE TABLE [dbo].[GameBoxscoreTeam] (
    [boxscoreId] INT PRIMARY KEY IDENTITY(1,1),
    [gameId] VARCHAR(25) NOT NULL,
    [teamId] VARCHAR(25) NOT NULL,
    [season] INT NOT NULL,
    [points] INT,
    [fieldGoalsMade] INT,
    [fieldGoalsAttempted] INT,
    [threePointersMade] INT,
    [threePointersAttempted] INT,
    [freeThrowsMade] INT,
    [freeThrowsAttempted] INT,
    [rebounds] INT,
    [assists] INT,
    [turnovers] INT,
    [steals] INT,
    [blocks] INT,
    [fouls] INT,
    [insert_date] DATETIME DEFAULT GETDATE(),
    [update_date] DATETIME DEFAULT GETDATE(),
    CONSTRAINT FK_GameBoxscoreTeam_Game FOREIGN KEY ([gameId]) REFERENCES [dbo].[GameInfo]([gameId]),
    CONSTRAINT FK_GameBoxscoreTeam_Team FOREIGN KEY ([teamId]) REFERENCES [dbo].[TeamInfo]([teamId])
);

-- ============================================
-- GAME BOXSCORE PLAYER TABLE
-- ============================================
CREATE TABLE [dbo].[GameBoxscorePlayer] (
    [boxscoreId] INT PRIMARY KEY IDENTITY(1,1),
    [gameId] VARCHAR(25) NOT NULL,
    [playerId] VARCHAR(25) NOT NULL,
    [teamId] VARCHAR(25) NOT NULL,
    [season] INT NOT NULL,
    [points] INT,
    [fieldGoalsMade] INT,
    [fieldGoalsAttempted] INT,
    [threePointersMade] INT,
    [threePointersAttempted] INT,
    [freeThrowsMade] INT,
    [freeThrowsAttempted] INT,
    [rebounds] INT,
    [assists] INT,
    [turnovers] INT,
    [steals] INT,
    [blocks] INT,
    [fouls] INT,
    [minutesPlayed] INT,
    [insert_date] DATETIME DEFAULT GETDATE(),
    [update_date] DATETIME DEFAULT GETDATE(),
    CONSTRAINT FK_GameBoxscorePlayer_Game FOREIGN KEY ([gameId]) REFERENCES [dbo].[GameInfo]([gameId]),
    CONSTRAINT FK_GameBoxscorePlayer_Player FOREIGN KEY ([playerId]) REFERENCES [dbo].[PlayerInfo]([playerId]),
    CONSTRAINT FK_GameBoxscorePlayer_Team FOREIGN KEY ([teamId]) REFERENCES [dbo].[TeamInfo]([teamId])
);

-- ============================================
-- GAME LINES TABLE
-- ============================================
CREATE TABLE [dbo].[GameLines] (
    [linesId] INT PRIMARY KEY IDENTITY(1,1),
    [gameId] VARCHAR(25) NOT NULL,
    [season] INT NOT NULL,
    [spread] DECIMAL(5,2),
    [overUnder] DECIMAL(5,2),
    [homeMoneyline] INT,
    [awayMoneyline] INT,
    [homeOdds] DECIMAL(5,2),
    [awayOdds] DECIMAL(5,2),
    [insert_date] DATETIME DEFAULT GETDATE(),
    [update_date] DATETIME DEFAULT GETDATE(),
    CONSTRAINT FK_GameLines_Game FOREIGN KEY ([gameId]) REFERENCES [dbo].[GameInfo]([gameId])
);

-- ============================================
-- INDEXES FOR PERFORMANCE
-- ============================================
CREATE INDEX IX_TeamInfo_Conference ON [dbo].[TeamInfo]([conferenceId]);
CREATE INDEX IX_PlayerInfo_Team ON [dbo].[PlayerInfo]([teamId]);
CREATE INDEX IX_PlayerInfo_Season ON [dbo].[PlayerInfo]([season]);
CREATE INDEX IX_GameInfo_Season ON [dbo].[GameInfo]([season]);
CREATE INDEX IX_GameInfo_HomeTeam ON [dbo].[GameInfo]([homeTeamId]);
CREATE INDEX IX_GameInfo_AwayTeam ON [dbo].[GameInfo]([awayTeamId]);
CREATE INDEX IX_GameInfo_StartDate ON [dbo].[GameInfo]([startDate]);
CREATE INDEX IX_GameBoxscoreTeam_Game ON [dbo].[GameBoxscoreTeam]([gameId]);
CREATE INDEX IX_GameBoxscoreTeam_Team ON [dbo].[GameBoxscoreTeam]([teamId]);
CREATE INDEX IX_GameBoxscorePlayer_Game ON [dbo].[GameBoxscorePlayer]([gameId]);
CREATE INDEX IX_GameBoxscorePlayer_Player ON [dbo].[GameBoxscorePlayer]([playerId]);
CREATE INDEX IX_GameBoxscorePlayer_Team ON [dbo].[GameBoxscorePlayer]([teamId]);
CREATE INDEX IX_GameLines_Game ON [dbo].[GameLines]([gameId]);
