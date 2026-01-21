-- CBB Pipeline V2 Database Schema
-- This script creates the refactored CBB pipeline database with optimized tables

-- ============================================
-- TEAM DATA TABLE
-- ============================================
CREATE TABLE [CBB_V2].[Team] (
    [team_id] INT PRIMARY KEY NOT NULL,
    [source_id] INT,
    [school_name] VARCHAR(255) NOT NULL,
    [team_abbr] VARCHAR(10) NOT NULL,
    [mascot] VARCHAR(100),
    [display_name] VARCHAR(255),
    [conference_id] INT,
    [conference_name] VARCHAR(100),
    [insert_date] DATETIME DEFAULT GETDATE(),
    [update_date] DATETIME DEFAULT GETDATE()
);

-- ============================================
-- GAME DATA TABLE
-- ============================================
CREATE TABLE [CBB_V2].[Game] (
    [game_id] INT PRIMARY KEY NOT NULL,
    [source_id] INT,
    [season] INT NOT NULL,
    [season_type] VARCHAR(50),
    [game_date] DATETIME2,
    [home_team_id] INT NOT NULL,
    [away_team_id] INT NOT NULL,
    [home_points] INT,
    [away_points] INT,
    [status] VARCHAR(50),
    [neutral_site] BIT,
    [conference_game] BIT,
    [insert_date] DATETIME DEFAULT GETDATE(),
    [update_date] DATETIME DEFAULT GETDATE(),
    CONSTRAINT FK_Game_HomeTeam FOREIGN KEY ([home_team_id]) REFERENCES [CBB_V2].[Team]([team_id]),
    CONSTRAINT FK_Game_AwayTeam FOREIGN KEY ([away_team_id]) REFERENCES [CBB_V2].[Team]([team_id])
);

-- ============================================
-- PLAYER DATA TABLE
-- ============================================
CREATE TABLE [CBB_V2].[Player] (
    [player_id] INT PRIMARY KEY NOT NULL,
    [source_id] INT,
    [team_id] INT NOT NULL,
    [season] INT NOT NULL,
    [name] VARCHAR(255),
    [first_name] VARCHAR(100),
    [last_name] VARCHAR(100),
    [jersey_number] INT,
    [position] VARCHAR(50),
    [height_inches] INT,
    [weight_lbs] INT,
    [start_season] INT,
    [end_season] INT,
    [team_name] VARCHAR(255),
    [conference] VARCHAR(100),
    [insert_date] DATETIME DEFAULT GETDATE(),
    [update_date] DATETIME DEFAULT GETDATE(),
    CONSTRAINT FK_Player_Team FOREIGN KEY ([team_id]) REFERENCES [CBB_V2].[Team]([team_id]),
    CONSTRAINT UC_Player_Season UNIQUE ([player_id], [team_id], [season])
);

-- ============================================
-- TEAM BOXSCORE TABLE
-- ============================================
CREATE TABLE [CBB_V2].[TeamBoxscore] (
    [boxscore_id] INT PRIMARY KEY IDENTITY(1,1),
    [game_id] INT NOT NULL,
    [team_id] INT NOT NULL,
    [season] INT NOT NULL,
    [points] INT,
    [field_goals_made] INT,
    [field_goals_attempted] INT,
    [three_pointers_made] INT,
    [three_pointers_attempted] INT,
    [free_throws_made] INT,
    [free_throws_attempted] INT,
    [rebounds] INT,
    [assists] INT,
    [turnovers] INT,
    [steals] INT,
    [blocks] INT,
    [fouls] INT,
    [insert_date] DATETIME DEFAULT GETDATE(),
    [update_date] DATETIME DEFAULT GETDATE(),
    CONSTRAINT FK_TeamBoxscore_Game FOREIGN KEY ([game_id]) REFERENCES [CBB_V2].[Game]([game_id]),
    CONSTRAINT FK_TeamBoxscore_Team FOREIGN KEY ([team_id]) REFERENCES [CBB_V2].[Team]([team_id])
);

-- ============================================
-- PLAYER BOXSCORE TABLE
-- ============================================
CREATE TABLE [CBB_V2].[PlayerBoxscore] (
    [boxscore_id] INT PRIMARY KEY IDENTITY(1,1),
    [game_id] INT NOT NULL,
    [player_id] INT NOT NULL,
    [team_id] INT NOT NULL,
    [season] INT NOT NULL,
    [points] INT,
    [field_goals_made] INT,
    [field_goals_attempted] INT,
    [three_pointers_made] INT,
    [three_pointers_attempted] INT,
    [free_throws_made] INT,
    [free_throws_attempted] INT,
    [rebounds] INT,
    [assists] INT,
    [turnovers] INT,
    [steals] INT,
    [blocks] INT,
    [fouls] INT,
    [minutes_played] INT,
    [insert_date] DATETIME DEFAULT GETDATE(),
    [update_date] DATETIME DEFAULT GETDATE(),
    CONSTRAINT FK_PlayerBoxscore_Game FOREIGN KEY ([game_id]) REFERENCES [CBB_V2].[Game]([game_id]),
    CONSTRAINT FK_PlayerBoxscore_Player FOREIGN KEY ([player_id]) REFERENCES [CBB_V2].[Player]([player_id]),
    CONSTRAINT FK_PlayerBoxscore_Team FOREIGN KEY ([team_id]) REFERENCES [CBB_V2].[Team]([team_id])
);

-- ============================================
-- CONFERENCE DATA TABLE
-- ============================================
CREATE TABLE [CBB_V2].[Conference] (
    [conference_id] INT PRIMARY KEY NOT NULL,
    [conference_name] VARCHAR(100) NOT NULL,
    [insert_date] DATETIME DEFAULT GETDATE(),
    [update_date] DATETIME DEFAULT GETDATE()
);

-- ============================================
-- INDEXES FOR PERFORMANCE
-- ============================================
CREATE INDEX IX_Game_Season ON [CBB_V2].[Game]([season]);
CREATE INDEX IX_Game_HomeTeam ON [CBB_V2].[Game]([home_team_id]);
CREATE INDEX IX_Game_AwayTeam ON [CBB_V2].[Game]([away_team_id]);
CREATE INDEX IX_Game_GameDate ON [CBB_V2].[Game]([game_date]);

CREATE INDEX IX_Player_TeamSeason ON [CBB_V2].[Player]([team_id], [season]);
CREATE INDEX IX_Player_Season ON [CBB_V2].[Player]([season]);

CREATE INDEX IX_TeamBoxscore_GameTeam ON [CBB_V2].[TeamBoxscore]([game_id], [team_id]);
CREATE INDEX IX_TeamBoxscore_Season ON [CBB_V2].[TeamBoxscore]([season]);

CREATE INDEX IX_PlayerBoxscore_GamePlayer ON [CBB_V2].[PlayerBoxscore]([game_id], [player_id]);
CREATE INDEX IX_PlayerBoxscore_Season ON [CBB_V2].[PlayerBoxscore]([season]);
CREATE INDEX IX_PlayerBoxscore_Team ON [CBB_V2].[PlayerBoxscore]([team_id]);
