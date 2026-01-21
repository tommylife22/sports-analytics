IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'TeamInfo')
BEGIN
CREATE TABLE [dbo].[TeamInfo] (
    [team_id] VARCHAR(10),
    [season] VARCHAR(10),
    [team_name] VARCHAR(255),
    [team_abbr] VARCHAR(10),
    [franchise_name] VARCHAR(255),
    [club_name] VARCHAR(255),
    [division] VARCHAR(100),
    [league] VARCHAR(100),
    [venue_id] VARCHAR(10),
    [venue_name] VARCHAR(255),
    [first_year] VARCHAR(10),
    [insert_date] DATETIME DEFAULT GETDATE(),
    [update_date] DATETIME DEFAULT GETDATE(),
    CONSTRAINT PK_TeamInfo PRIMARY KEY ([team_id], [season])
)
END;

IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'GameInfo')
BEGIN
CREATE TABLE [dbo].[GameInfo] (
    [game_id] VARCHAR(10),
    [season] VARCHAR(10),
    [game_datetime] DATETIME2,
    [away_id] VARCHAR(10),
    [away_name] VARCHAR(255),
    [home_id] VARCHAR(10),
    [home_name] VARCHAR(255),
    [away_score] INT,
    [home_score] INT,
    [status] VARCHAR(100),
    [game_type] VARCHAR(10),
    [venue_id] VARCHAR(10),
    [venue_name] VARCHAR(255),
    [insert_date] DATETIME DEFAULT GETDATE(),
    [update_date] DATETIME DEFAULT GETDATE(),
    CONSTRAINT PK_GameInfo PRIMARY KEY ([game_id])
)
END;

IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'PlayerInfo')
BEGIN
CREATE TABLE [dbo].[PlayerInfo] (
    [player_id] VARCHAR(10),
    [team_id] VARCHAR(10),
    [season] VARCHAR(10),
    [full_name] VARCHAR(255),
    [jersey_number] VARCHAR(10),
    [position_code] VARCHAR(255),
    [position_name] VARCHAR(255),
    [position_type] VARCHAR(255),
    [status] VARCHAR(255),
    [roster_type] VARCHAR(255),
    [insert_date] DATETIME DEFAULT GETDATE(),
    [update_date] DATETIME DEFAULT GETDATE(),
    CONSTRAINT PK_PlayerInfo PRIMARY KEY ([player_id], [team_id], [season])
)
END;

IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'TeamBoxscore')
BEGIN
CREATE TABLE [dbo].[TeamBoxscore] (
    -- Primary Keys (for joins)
    [game_id] VARCHAR(10),
    [team_id] VARCHAR(10),

    -- Game context
    [is_home] BIT,

    -- Batting Stats
    [runs] INT,
    [hits] INT,
    [doubles] INT,
    [triples] INT,
    [home_runs] INT,
    [rbi] INT,
    [walks] INT,
    [strikeouts] INT,
    [stolen_bases] INT,
    [caught_stealing] INT,
    [left_on_base] INT,
    [hit_by_pitch] INT,
    [avg] DECIMAL(5,3),
    [obp] DECIMAL(5,3),
    [slg] DECIMAL(5,3),
    [ops] DECIMAL(5,3),

    -- Pitching Stats
    [earned_runs] INT,
    [hits_allowed] INT,
    [home_runs_allowed] INT,
    [walks_allowed] INT,
    [strikeouts_pitched] INT,
    [pitches_thrown] INT,
    [strikes] INT,
    [era] DECIMAL(5,2),

    -- Metadata
    [insert_date] DATETIME DEFAULT GETDATE(),
    [update_date] DATETIME DEFAULT GETDATE(),

    -- Constraints
    CONSTRAINT PK_TeamBoxscore PRIMARY KEY ([game_id], [team_id]),
    CONSTRAINT FK_TeamBoxscore_Game FOREIGN KEY ([game_id])
        REFERENCES [dbo].[GameInfo]([game_id])
)
END;

IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'PlayerBoxscore')
BEGIN
CREATE TABLE [dbo].[PlayerBoxscore] (
    -- Primary Keys (for joins)
    [game_id] VARCHAR(10),
    [player_id] VARCHAR(10),
    [team_id] VARCHAR(10),

    -- Game context
    [is_home] BIT,
    [position] VARCHAR(10),
    [batting_order] VARCHAR(10),

    -- Batting Stats (NULL for pitchers who didn't bat)
    [at_bats] INT,
    [runs] INT,
    [hits] INT,
    [doubles] INT,
    [triples] INT,
    [home_runs] INT,
    [rbi] INT,
    [walks] INT,
    [strikeouts] INT,
    [stolen_bases] INT,
    [caught_stealing] INT,
    [hit_by_pitch] INT,
    [avg] DECIMAL(5,3),
    [obp] DECIMAL(5,3),
    [slg] DECIMAL(5,3),
    [ops] DECIMAL(5,3),

    -- Pitching Stats (NULL for batters)
    [innings_pitched] DECIMAL(5,2),
    [hits_allowed] INT,
    [runs_allowed] INT,
    [earned_runs] INT,
    [walks_allowed] INT,
    [strikeouts_pitched] INT,
    [home_runs_allowed] INT,
    [pitches_thrown] INT,
    [strikes] INT,
    [era] DECIMAL(5,2),
    [win] INT,
    [loss] INT,
    [save] INT,
    [blown_save] INT,
    [hold] INT,

    -- Metadata
    [insert_date] DATETIME DEFAULT GETDATE(),
    [update_date] DATETIME DEFAULT GETDATE(),

    -- Constraints
    CONSTRAINT PK_PlayerBoxscore PRIMARY KEY ([game_id], [player_id]),
    CONSTRAINT FK_PlayerBoxscore_Game FOREIGN KEY ([game_id])
        REFERENCES [dbo].[GameInfo]([game_id])
)
END;

IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'BettingOdds')
BEGIN
CREATE TABLE [dbo].[BettingOdds] (
    [date] DATE NOT NULL,
    [game_id] VARCHAR(50),
    [game_time] DATETIME,
    [away_team] VARCHAR(50),
    [home_team] VARCHAR(50),
    [sportsbook] VARCHAR(50),
    [bet_type] VARCHAR(20),

    -- Moneyline
    [away_line] INT,
    [home_line] INT,

    -- Spread
    [away_spread] DECIMAL(5,2),
    [away_spread_odds] INT,
    [home_spread] DECIMAL(5,2),
    [home_spread_odds] INT,

    -- Total
    [total] DECIMAL(5,2),
    [over_odds] INT,
    [under_odds] INT,

    -- Metadata
    [insert_date] DATETIME DEFAULT GETDATE(),
    [update_date] DATETIME DEFAULT GETDATE(),

    CONSTRAINT PK_BettingOdds PRIMARY KEY ([game_id], [sportsbook], [bet_type])
)
END;

IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'StatcastPitches')
BEGIN
CREATE TABLE [dbo].[StatcastPitches] (
    -- Identifiers
    [pitch_id] BIGINT IDENTITY(1,1),
    [game_pk] VARCHAR(10),
    [game_date] DATE,
    [game_year] INT,
    [at_bat_number] INT,
    [pitch_number] INT,

    -- Players (MLBAM IDs)
    [pitcher] INT,
    [batter] INT,
    [pitcher_name] VARCHAR(100),
    [batter_name] VARCHAR(100),

    -- Game State
    [inning] INT,
    [inning_topbot] VARCHAR(10),
    [outs_when_up] INT,
    [balls] INT,
    [strikes] INT,
    [on_1b] INT,
    [on_2b] INT,
    [on_3b] INT,
    [stand] VARCHAR(1),                    -- L/R batter stance
    [p_throws] VARCHAR(1),                 -- L/R pitcher throws

    -- Pitch Characteristics
    [pitch_type] VARCHAR(10),              -- FF, SL, CH, CU, etc.
    [pitch_name] VARCHAR(50),
    [release_speed] DECIMAL(5,2),
    [release_pos_x] DECIMAL(5,2),
    [release_pos_y] DECIMAL(5,2),
    [release_pos_z] DECIMAL(5,2),
    [release_spin_rate] INT,
    [release_extension] DECIMAL(5,2),
    [spin_axis] INT,

    -- Pitch Movement (inches)
    [pfx_x] DECIMAL(5,2),                  -- Horizontal break
    [pfx_z] DECIMAL(5,2),                  -- Vertical break
    [plate_x] DECIMAL(5,2),                -- Location at plate
    [plate_z] DECIMAL(5,2),

    -- Velocity Components (ft/s)
    [vx0] DECIMAL(6,3),
    [vy0] DECIMAL(6,3),
    [vz0] DECIMAL(6,3),

    -- Acceleration Components (ft/s^2)
    [ax] DECIMAL(6,3),
    [ay] DECIMAL(6,3),
    [az] DECIMAL(6,3),

    -- Strike Zone Info
    [zone] INT,
    [sz_top] DECIMAL(5,2),
    [sz_bot] DECIMAL(5,2),

    -- Batted Ball Data (2015+)
    [launch_speed] DECIMAL(5,2),           -- Exit velocity
    [launch_angle] DECIMAL(5,2),
    [hit_distance_sc] INT,
    [bb_type] VARCHAR(20),                 -- fly_ball, ground_ball, line_drive, popup
    [hc_x] DECIMAL(6,2),                   -- Hit coordinate X
    [hc_y] DECIMAL(6,2),                   -- Hit coordinate Y

    -- Barrel/Expected Stats
    [barrel] BIT,
    [estimated_ba_using_speedangle] DECIMAL(5,3),
    [estimated_woba_using_speedangle] DECIMAL(5,3),
    [woba_value] DECIMAL(5,3),
    [woba_denom] INT,
    [babip_value] DECIMAL(5,3),
    [iso_value] DECIMAL(5,3),

    -- Outcome
    [type] VARCHAR(2),                     -- S (strike), B (ball), X (in play)
    [description] VARCHAR(100),            -- Called strike, swinging strike, etc.
    [events] VARCHAR(50),                  -- single, double, home_run, strikeout, etc.
    [des] VARCHAR(500),                    -- Full play description

    -- Home Run Details
    [home_run] BIT,
    [hit_location] VARCHAR(10),

    -- Additional Context
    [if_fielding_alignment] VARCHAR(20),
    [of_fielding_alignment] VARCHAR(20),

    -- Metadata
    [insert_date] DATETIME DEFAULT GETDATE(),
    [update_date] DATETIME DEFAULT GETDATE(),

    CONSTRAINT PK_StatcastPitches PRIMARY KEY ([pitch_id])
)
END;

-- Indexes for performance
CREATE INDEX IX_StatcastPitches_GameDate  ON [dbo].[StatcastPitches]([game_date]);
CREATE INDEX IX_StatcastPitches_GamePK    ON [dbo].[StatcastPitches]([game_pk]);
CREATE INDEX IX_StatcastPitches_Pitcher   ON [dbo].[StatcastPitches]([pitcher]);
CREATE INDEX IX_StatcastPitches_Batter    ON [dbo].[StatcastPitches]([batter]);
CREATE INDEX IX_StatcastPitches_PitchType ON [dbo].[StatcastPitches]([pitch_type]);
CREATE INDEX IX_StatcastPitches_Events    ON [dbo].[StatcastPitches]([events]);

SELECT
    g.game_id,
    g.game_datetime,
    g.home_name AS full_home_name,
    g.away_name AS full_away_name,
    g.home_score,
    g.away_score,
    b.sportsbook,
    b.home_line,
    b.away_line,
    b.total,
    b.over_odds,
    b.under_odds,
    b.home_spread,
    b.home_spread_odds,
    b.away_spread,
    b.away_spread_odds
FROM GameInfo g
JOIN BettingOdds b
    ON g.game_datetime = b.game_time
    AND g.home_name LIKE '%' + b.home_team + '%'
    AND g.away_name LIKE '%' + b.away_team + '%'
WHERE date = '2025-07-27' 
    AND home_team = 'Chicago'

SELECT *
FROM StatcastPitches
WHERE pitcher = '445276'
    AND pitch_type IS NULL