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

rolling_cols = [
    "points", "goals_for", "goals_against",
    "shots_for", "shots_against",
    "shots_target_for", "shots_target_against",
    "corners_for", "corners_against",
    "yellow_for", "yellow_against",
    "red_for", "red_against",
]

# get each team's last 5 real matches, as a list of dicts, oldest first 

most_recent_dict = (
    team_matches.sort_values("date")
    .groupby("team_id")
    .tail(5)
    .groupby("team_id")
    .apply(lambda x: x.sort_values("date").to_dict(orient="records"))
    .to_dict()
)

# teams with no real history at all use the promoted-team baseline 

promoted_baseline = pd.read_csv("data/processed/promoted_team_baseline.csv", index_col=0).iloc[:, 0]

NO_HISTORY_TEAMS = [9010, 9011]  # Coventry City, Hull City

all_2026_27_teams = [
    57, 58, 1044, 402, 397, 61, 9010, 354, 62, 63,
    9011, 9008, 9005, 64, 65, 66, 67, 351, 9009, 73,
]

# build the initial state: each team holds a list of its recent matches 

team_state = {}

baseline_match = {
    "points": promoted_baseline["points"],
    "goals_for": promoted_baseline["goals_for"],
    "goals_against": promoted_baseline["goals_against"],
    "shots_for": promoted_baseline["shots_for"],
    "shots_against": promoted_baseline["shots_against"],
    "shots_target_for": promoted_baseline["shots_target_for"],
    "shots_target_against": promoted_baseline["shots_target_against"],
    "corners_for": promoted_baseline["corners_for"],
    "corners_against": promoted_baseline["corners_against"],
    "yellow_for": 0,
    "yellow_against": 0,
    "red_for": 0,
    "red_against": 0,
}

for team_id in all_2026_27_teams:
    if team_id in NO_HISTORY_TEAMS:
        team_state[team_id] = {
            "recent_matches": [baseline_match.copy() for _ in range(5)],
            "source": "promoted_baseline",
        }
    else:
        team_state[team_id] = {
            "recent_matches": most_recent_dict[team_id],
            "source": "real_history",
        }

# helper: calculate a rolling sum on demand, blending real-history teams
# 70% their own recent form / 30% league average, to soften the carryover
# of last season's form into a brand new season 

league_avg = team_matches[rolling_cols].mean() * 5

def get_rolling_sum(team_state, team_id, stat, league_avg):
    matches = team_state[team_id]["recent_matches"]
    raw_sum = sum(m[stat] for m in matches)

    if team_state[team_id]["source"] == "real_history":
        return 0.7 * raw_sum + 0.3 * league_avg[stat]
    else:
        return raw_sum

# sanity check 

for team_id, state in team_state.items():
    print(team_id, state["source"], "points:", get_rolling_sum(team_state, team_id, "points", league_avg))