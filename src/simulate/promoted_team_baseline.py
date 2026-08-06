import sqlite3
import pandas as pd

connect = sqlite3.connect("db/pl_data.db")
matches = pd.read_sql_query("SELECT * FROM matches", connect)
connect.close()

matches["utc_date"] = pd.to_datetime(matches["utc_date"], dayfirst=True)

# rebuild team-perspective table 

home = matches[[
    "season_year", "utc_date", "home_team_id",
    "home_goals_fulltime", "away_goals_fulltime",
    "home_shots", "away_shots",
    "home_shots_target", "away_shots_target",
    "home_corners", "away_corners",
    "winner"
]].copy()
home.columns = [
    "season_year", "date", "team_id",
    "goals_for", "goals_against",
    "shots_for", "shots_against",
    "shots_target_for", "shots_target_against",
    "corners_for", "corners_against",
    "winner"
]

away = matches[[
    "season_year", "utc_date", "away_team_id",
    "away_goals_fulltime", "home_goals_fulltime",
    "away_shots", "home_shots",
    "away_shots_target", "home_shots_target",
    "away_corners", "home_corners",
    "winner"
]].copy()
away.columns = [
    "season_year", "date", "team_id",
    "goals_for", "goals_against",
    "shots_for", "shots_against",
    "shots_target_for", "shots_target_against",
    "corners_for", "corners_against",
    "winner"
]

home_result_map = {"HOME_TEAM": "WIN", "AWAY_TEAM": "LOSS", "DRAW": "DRAW"}
away_result_map = {"HOME_TEAM": "LOSS", "AWAY_TEAM": "WIN", "DRAW": "DRAW"}
points_map = {"WIN": 3, "DRAW": 1, "LOSS": 0}

home["points"] = home["winner"].map(home_result_map).map(points_map)
away["points"] = away["winner"].map(away_result_map).map(points_map)

team_matches = pd.concat([home, away], ignore_index=True)
team_matches = team_matches.sort_values(["team_id", "season_year", "date"]).reset_index(drop=True)

# identify promoted team-seasons 

team_seasons = pd.concat([
    matches[["home_team_id", "season_year"]].rename(columns={"home_team_id": "team_id"}),
    matches[["away_team_id", "season_year"]].rename(columns={"away_team_id": "team_id"})
], ignore_index=True).drop_duplicates()

shifted = team_seasons.copy()
shifted["season_year"] = shifted["season_year"] + 1

promoted = pd.merge(team_seasons, shifted, on=["team_id", "season_year"], how="left", indicator=True)
promoted_teams = promoted[promoted["_merge"] == "left_only"]
promoted_teams = promoted_teams[promoted_teams["season_year"] != 2018]  # exclude dataset start artifact

# pull each promoted team's first 5 real matches of that season 

first_5_rows = []

for _, row in promoted_teams.iterrows():
    team_id = row["team_id"]
    season_year = row["season_year"]

    team_season_matches = team_matches[
        (team_matches["team_id"] == team_id) & (team_matches["season_year"] == season_year)
    ].sort_values("date").head(5)

    first_5_rows.append(team_season_matches)

promoted_first_5 = pd.concat(first_5_rows, ignore_index=True)

# average across all promoted teams' first 5 games, pooled together 

promoted_baseline = promoted_first_5[[
    "points", "goals_for", "goals_against",
    "shots_for", "shots_against",
    "shots_target_for", "shots_target_against",
    "corners_for", "corners_against"
]].mean()

print(f"Pooled from {len(promoted_teams)} promoted team-seasons, {len(promoted_first_5)} total matches")
print("\nPromoted team baseline (average per match, first 5 games back in the PL):")
print(promoted_baseline)

promoted_baseline.to_csv("data/processed/promoted_team_baseline.csv")
print("\nSaved to data/processed/promoted_team_baseline.csv")