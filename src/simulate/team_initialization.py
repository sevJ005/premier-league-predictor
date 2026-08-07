import sqlite3
import pandas as pd

connect = sqlite3.connect("db/pl_data.db")
matches = pd.read_sql_query("SELECT * FROM matches", connect)
connect.close()

matches["utc_date"] = pd.to_datetime(matches["utc_date"], dayfirst=True)

home = matches[[
    "season_year", "utc_date", "home_team_id",
    "home_goals_fulltime", "away_goals_fulltime",
    "home_shots", "away_shots",
    "home_shots_target", "away_shots_target",
    "home_corners", "away_corners",
    "home_yellow", "away_yellow",
    "home_red", "away_red",
    "winner"
]].copy()
home.columns = [
    "season_year", "date", "team_id",
    "goals_for", "goals_against",
    "shots_for", "shots_against",
    "shots_target_for", "shots_target_against",
    "corners_for", "corners_against",
    "yellow_for", "yellow_against",
    "red_for", "red_against",
    "winner"
]

away = matches[[
    "season_year", "utc_date", "away_team_id",
    "away_goals_fulltime", "home_goals_fulltime",
    "away_shots", "home_shots",
    "away_shots_target", "home_shots_target",
    "away_corners", "home_corners",
    "away_yellow", "home_yellow",
    "away_red", "home_red",
    "winner"
]].copy()
away.columns = [
    "season_year", "date", "team_id",
    "goals_for", "goals_against",
    "shots_for", "shots_against",
    "shots_target_for", "shots_target_against",
    "corners_for", "corners_against",
    "yellow_for", "yellow_against",
    "red_for", "red_against",
    "winner"
]

home_result_map = {"HOME_TEAM": "WIN", "AWAY_TEAM": "LOSS", "DRAW": "DRAW"}
away_result_map = {"HOME_TEAM": "LOSS", "AWAY_TEAM": "WIN", "DRAW": "DRAW"}
points_map = {"WIN": 3, "DRAW": 1, "LOSS": 0}

home["points"] = home["winner"].map(home_result_map).map(points_map)
away["points"] = away["winner"].map(away_result_map).map(points_map)

team_matches = pd.concat([home, away], ignore_index=True)
team_matches = team_matches.sort_values(["team_id", "date"]).reset_index(drop=True)

# rolling stats -- min_periods=1 
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
        team_matches.groupby("team_id")[col]
        .shift(0)  # include the match itself this time, we want form AS OF their most recent match, not excluding it
        .rolling(window=5, min_periods=1)
        .sum()
    )

# get each team's single most recent real match 

most_recent = team_matches.sort_values("date").groupby("team_id").tail(1)
most_recent = most_recent.set_index("team_id")

# teams with no real history at all use the promoted-team baseline 

promoted_baseline = pd.read_csv("data/processed/promoted_team_baseline.csv", index_col=0).iloc[:, 0]

# multiply per-match averages by 5 to match the "sum over last 5" scale used
# by the rolling features everywhere else in this project
NO_HISTORY_TEAMS = [9010, 9011]  # Coventry City, Hull City

# build the initial state dictionary 

team_state = {}

all_2026_27_teams = [
    57, 58, 1044, 402, 397, 61, 9010, 354, 62, 63,
    9011, 9008, 9005, 64, 65, 66, 67, 351, 9009, 73,
]

for team_id in all_2026_27_teams:
    if team_id in NO_HISTORY_TEAMS:
        team_state[team_id] = {
            "rolling_points_5": promoted_baseline["points"] * 5,
            "rolling_goals_for_5": promoted_baseline["goals_for"] * 5,
            "rolling_goals_against_5": promoted_baseline["goals_against"] * 5,
            "rolling_shots_for_5": promoted_baseline["shots_for"] * 5,
            "rolling_shots_against_5": promoted_baseline["shots_against"] * 5,
            "rolling_shots_target_for_5": promoted_baseline["shots_target_for"] * 5,
            "rolling_shots_target_against_5": promoted_baseline["shots_target_against"] * 5,
            "rolling_corners_for_5": promoted_baseline["corners_for"] * 5,
            "rolling_corners_against_5": promoted_baseline["corners_against"] * 5,
            "rolling_yellow_for_5": 0,  # not tracked in the baseline -- neutral default
            "rolling_yellow_against_5": 0,
            "rolling_red_for_5": 0,
            "rolling_red_against_5": 0,
            "source": "promoted_baseline",
        }
    else:
        row = most_recent.loc[team_id]
        team_state[team_id] = {
            "rolling_points_5": row["rolling_points_5"],
            "rolling_goals_for_5": row["rolling_goals_for_5"],
            "rolling_goals_against_5": row["rolling_goals_against_5"],
            "rolling_shots_for_5": row["rolling_shots_for_5"],
            "rolling_shots_against_5": row["rolling_shots_against_5"],
            "rolling_shots_target_for_5": row["rolling_shots_target_for_5"],
            "rolling_shots_target_against_5": row["rolling_shots_target_against_5"],
            "rolling_corners_for_5": row["rolling_corners_for_5"],
            "rolling_corners_against_5": row["rolling_corners_against_5"],
            "rolling_yellow_for_5": row["rolling_yellow_for_5"],
            "rolling_yellow_against_5": row["rolling_yellow_against_5"],
            "rolling_red_for_5": row["rolling_red_for_5"],
            "rolling_red_against_5": row["rolling_red_against_5"],
            "source": "real_history",
        }

league_avg = team_matches[rolling_cols].mean() * 5

for team_id, state in team_state.items():
    if state["source"] == "real_history":
        for col in rolling_cols:
            key = f"rolling_{col}_5"
            state[key] = 0.7 * state[key] + 0.3 * league_avg[col]

import sqlite3
import pandas as pd

connect = sqlite3.connect("db/pl_data.db")
matches = pd.read_sql_query("SELECT * FROM matches", connect)
connect.close()

# columns we'll need to "use" when simulating a match's actual stats
stat_cols = [
    "home_goals_fulltime", "away_goals_fulltime",
    "home_shots", "away_shots",
    "home_shots_target", "away_shots_target",
    "home_corners", "away_corners",
    "home_yellow", "away_yellow",
    "home_red", "away_red",
]

# three pools of real historical matches, grouped by outcome, used to
# sample a realistic scoreline/stat-line for a simulated match with that
# same outcome, rather than inventing a fake score
outcome_pools = {
    "HOME_TEAM": matches[matches["winner"] == "HOME_TEAM"][stat_cols].reset_index(drop=True),
    "DRAW": matches[matches["winner"] == "DRAW"][stat_cols].reset_index(drop=True),
    "AWAY_TEAM": matches[matches["winner"] == "AWAY_TEAM"][stat_cols].reset_index(drop=True),
}

for outcome, pool in outcome_pools.items():
    print(outcome, "-", len(pool), "historical matches available to sample from")

for team_id, state in team_state.items():
    print(team_id, state["source"], "points:", state["rolling_points_5"])