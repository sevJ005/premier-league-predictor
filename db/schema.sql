CREATE TABLE IF NOT EXISTS teams (
    team_id int NOT NULL PRIMARY KEY,
    name varchar(255) NOT NULL,
    short_name varchar(255),
    tla varchar(5)
);

CREATE TABLE IF NOT EXISTS standings (
    team_id int NOT NULL,
    season_year int NOT NULL,
    position int,
    played int,
    won int,
    draw int,
    loss int,
    points int, 
    goals_for int, 
    goals_against int,
    goal_difference int, 
    form varchar(9),
    FOREIGN KEY (team_id) REFERENCES teams(team_id),
    PRIMARY KEY (team_id, season_year)
);

CREATE TABLE IF NOT EXISTS matches (
    match_id int NOT NULL PRIMARY KEY,
    season_year int NOT NULL,
    matchday int,
    utc_date varchar(20),
    status varchar(10),
    home_team_id int,
    away_team_id int,
    home_goals_fulltime int,
    away_goals_fulltime int,
    home_goals_halftime int,
    away_goals_halftime int,
    home_shots INTEGER,
    away_shots INTEGER,
    home_shots_target INTEGER,
    away_shots_target INTEGER,
    home_fouls INTEGER,
    away_fouls INTEGER,
    home_corners INTEGER,
    away_corners INTEGER,
    home_yellow INTEGER,
    away_yellow INTEGER,
    home_red INTEGER,
    away_red INTEGER,
    referee TEXT,
    b365_home_odds REAL,
    b365_draw_odds REAL,
    b365_away_odds REAL,
    winner varchar(9),
    FOREIGN KEY (home_team_id) REFERENCES teams(team_id),
    FOREIGN KEY (away_team_id) REFERENCES teams(team_id)
);