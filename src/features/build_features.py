import pandas as pd
import sqlite3

# load raw matches

connect = sqlite3.connect("db/pl_data.db")
matches = pd.read_sql_query("SELECT * FROM matches", connect)
connect.close()

matches["utc_date"] = pd.to_datetime(matches["utc_date"], dayfirst=True)

# one row per match

matches = matches.sort_values(["season_year", "utc_date"]).reset_index(drop=True)
matches["matchday"] = matches.groupby("season_year").cumcount() + 1

# odds, simple form

matches["implied_prob_home"] = 1 / matches["b365_home_odds"]
matches["implied_prob_draw"] = 1 / matches["b365_draw_odds"]
matches["implied_prob_away"] = 1 / matches["b365_away_odds"]


# BUILD HOME / AWAY TEAM-PERSPECTIVE TABLES
# each match becomes 2 rows: one for the home team's experience,
# one for the away team's experience, using shared generic column names

home = matches[[
    "match_id", "season_year", "matchday", "utc_date",
    "home_team_id", "away_team_id",
    "home_goals_fulltime", "away_goals_fulltime",
    "home_shots", "away_shots",
    "home_shots_target", "away_shots_target",
    "home_corners", "away_corners",
    "home_yellow", "away_yellow",
    "home_red", "away_red",
    "winner"
]].copy()
home.columns = [
    "match_id", "season_year", "matchday", "date",
    "team_id", "opponent_id",
    "goals_for", "goals_against",
    "shots_for", "shots_against",
    "shots_target_for", "shots_target_against",
    "corners_for", "corners_against",
    "yellow_for", "yellow_against",
    "red_for", "red_against",
    "winner"
]
home["venue"] = "HOME"

away = matches[[
    "match_id", "season_year", "matchday", "utc_date",
    "away_team_id", "home_team_id",
    "away_goals_fulltime", "home_goals_fulltime",
    "away_shots", "home_shots",
    "away_shots_target", "home_shots_target",
    "away_corners", "home_corners",
    "away_yellow", "home_yellow",
    "away_red", "home_red",
    "winner"
]].copy()
away.columns = [
    "match_id", "season_year", "matchday", "date",
    "team_id", "opponent_id",
    "goals_for", "goals_against",
    "shots_for", "shots_against",
    "shots_target_for", "shots_target_against",
    "corners_for", "corners_against",
    "yellow_for", "yellow_against",
    "red_for", "red_against",
    "winner"
]
away["venue"] = "AWAY"

# translate raw HOME_TEAM/AWAY_TEAM/DRAW into each side's own result
home_result_map = {"HOME_TEAM": "WIN", "AWAY_TEAM": "LOSS", "DRAW": "DRAW"}
away_result_map = {"HOME_TEAM": "LOSS", "AWAY_TEAM": "WIN", "DRAW": "DRAW"}

home["result"] = home["winner"].map(home_result_map)
away["result"] = away["winner"].map(away_result_map)

points_map = {"WIN": 3, "DRAW": 1, "LOSS": 0}
home["points"] = home["result"].map(points_map)
away["points"] = away["result"].map(points_map)

team_matches = pd.concat([home, away], ignore_index=True)
team_matches = team_matches.sort_values(["team_id", "date"]).reset_index(drop=True)

# rest days between games
team_matches["rest_days"] = (
    team_matches.groupby("team_id")["date"].diff().dt.days
)

# cap rest_days at a reasonable ceiling after ~30 days, "more rest" stops
# being a meaningful signal (it's either a season opener or a team returning
# from relegation, not a fatigue difference)
team_matches["long_layoff"] = team_matches["rest_days"] > 30
team_matches["rest_days"] = team_matches["rest_days"].clip(upper=30)


## rolling form excluding current game

rolling_cols = [
    "points", "goals_for", "goals_against",
    "shots_for", "shots_against",
    "shots_target_for", "shots_target_against",
    "corners_for", "corners_against",
    "yellow_for", "yellow_against",
    "red_for", "red_against",
]

for col in rolling_cols:
    team_matches[f"rolling_{col}_5"] = (
        team_matches.groupby("team_id")[col].shift(1).rolling(window=5).sum()
    )

# HOME/AWAY specific rolling form

for col in ["points", "goals_for", "goals_against"]:
    team_matches[f"rolling_{col}_5_by_venue"] = (
        team_matches.groupby(["team_id", "venue"])[col]
        .shift(1)
        .rolling(window=5)
        .sum()
    )

# Head 2 Head form

team_matches = team_matches.sort_values(["team_id", "opponent_id", "date"])
team_matches["h2h_points_avg_3"] = (
    team_matches.groupby(["team_id", "opponent_id"])["points"]
    .shift(1)
    .rolling(window=3)
    .mean()
)

# restore chronological order per team before moving on
team_matches = team_matches.sort_values(["team_id", "date"]).reset_index(drop=True)


# RESHAPE BACK TO ONE ROW PER MATCH
# merge the home-side and away-side feature rows back together,
# using match_id + which team was home/away in the original matches table

home_features = matches[["match_id", "home_team_id"]].merge(
    team_matches,
    left_on=["match_id", "home_team_id"],
    right_on=["match_id", "team_id"]
)

away_features = matches[["match_id", "away_team_id"]].merge(
    team_matches,
    left_on=["match_id", "away_team_id"],
    right_on=["match_id", "team_id"]
)

match_features = home_features.merge(
    away_features,
    on="match_id",
    suffixes=("_home", "_away")
)

# attach match-level fields that don't belong to either side specifically
match_level_cols = matches[[
    "match_id", "season_year", "matchday", "utc_date", "status", "referee",
    "home_goals_fulltime", "away_goals_fulltime", "winner",
    "implied_prob_home", "implied_prob_draw", "implied_prob_away"
]]

match_features = match_level_cols.merge(match_features, on="match_id")

# Saving to processed data

match_features.to_csv("data/processed/match_features.csv", index=False)

print(f"Built match_features with shape {match_features.shape}")
print(f"Saved to data/processed/match_features.csv")