import sqlite3
import pandas as pd

connect = sqlite3.connect("db/pl_data.db")
matches = pd.read_sql_query("SELECT * FROM matches", connect)
connect.close()

matches["utc_date"] = pd.to_datetime(matches["utc_date"], dayfirst=True)

# rebuild team-perspective table, now tagging each row with its venue
# so we can build proper venue-specific rolling history later

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
home["venue"] = "HOME"

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
away["venue"] = "AWAY"

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

# each team's last 5 real matches overall (any venue), for the main
# rolling features matches how rolling_points_5 etc. 

most_recent_dict = (
    team_matches.sort_values("date")
    .groupby("team_id")
    .tail(5)
    .groupby("team_id")
    .apply(lambda x: x.sort_values("date").to_dict(orient="records"))
    .to_dict()
)

# each team's last 5 real HOME matches, and last 5 real AWAY matches,
# separately for the venue-specific rolling features 

most_recent_home_dict = (
    team_matches[team_matches["venue"] == "HOME"]
    .sort_values("date")
    .groupby("team_id")
    .tail(5)
    .groupby("team_id")
    .apply(lambda x: x.sort_values("date").to_dict(orient="records"))
    .to_dict()
)

most_recent_away_dict = (
    team_matches[team_matches["venue"] == "AWAY"]
    .sort_values("date")
    .groupby("team_id")
    .tail(5)
    .groupby("team_id")
    .apply(lambda x: x.sort_values("date").to_dict(orient="records"))
    .to_dict()
)

# teams with no real history at all: use the promoted-team baseline 

promoted_baseline = pd.read_csv("data/processed/promoted_team_baseline.csv", index_col=0).iloc[:, 0]

# league-average cards, used as a defensible stand-in for promoted teams
# instead of hardcoding zero (which would understate their likely discipline record)
league_avg_yellow_for = team_matches["yellow_for"].mean()
league_avg_yellow_against = team_matches["yellow_against"].mean()
league_avg_red_for = team_matches["red_for"].mean()
league_avg_red_against = team_matches["red_against"].mean()

NO_HISTORY_TEAMS = [9010, 9011]  # Coventry City, Hull City

all_2026_27_teams = [
    57, 58, 1044, 402, 397, 61, 9010, 354, 62, 63,
    9011, 9008, 9005, 64, 65, 66, 67, 351, 9009, 73,
]

# build the initial state 

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
    "yellow_for": league_avg_yellow_for,
    "yellow_against": league_avg_yellow_against,
    "red_for": league_avg_red_for,
    "red_against": league_avg_red_against,
}

for team_id in all_2026_27_teams:
    if team_id in NO_HISTORY_TEAMS:
        team_state[team_id] = {
            "recent_matches": [baseline_match.copy() for _ in range(5)],
            "recent_home_matches": [baseline_match.copy() for _ in range(5)],
            "recent_away_matches": [baseline_match.copy() for _ in range(5)],
            "source": "promoted_baseline",
            "season_total_points": 0,
        }
    else:
        team_state[team_id] = {
            "recent_matches": most_recent_dict[team_id],
            "recent_home_matches": most_recent_home_dict.get(team_id, most_recent_dict[team_id]),
            "recent_away_matches": most_recent_away_dict.get(team_id, most_recent_dict[team_id]),
            "source": "real_history",
            "season_total_points": 0,
        }

# helper: calculate a rolling sum on demand from a given list of matches,
# blending real-history teams 70% their own recent form / 30% league average

league_avg = team_matches[rolling_cols].mean() * 5

def get_rolling_sum(team_state, team_id, stat, league_avg, venue=None):
    if venue == "HOME":
        matches = team_state[team_id]["recent_home_matches"]
    elif venue == "AWAY":
        matches = team_state[team_id]["recent_away_matches"]
    else:
        matches = team_state[team_id]["recent_matches"]

    raw_sum = sum(m[stat] for m in matches)

    if team_state[team_id]["source"] == "real_history":
        return 0.85 * raw_sum + 0.15 * league_avg[stat]
        # return raw_sum + 0 * league_avg[stat]
    else:
        return raw_sum

stat_cols = [
    "home_goals_fulltime", "away_goals_fulltime",
    "home_shots", "away_shots",
    "home_shots_target", "away_shots_target",
    "home_corners", "away_corners",
    "home_yellow", "away_yellow",
    "home_red", "away_red",
]

outcome_pools = {
    "HOME_TEAM": matches[matches["winner"] == "HOME_TEAM"][stat_cols].reset_index(drop=True),
    "DRAW": matches[matches["winner"] == "DRAW"][stat_cols].reset_index(drop=True),
    "AWAY_TEAM": matches[matches["winner"] == "AWAY_TEAM"][stat_cols].reset_index(drop=True),
}

outcome_arrays = {
    outcome: pool.to_numpy()
    for outcome, pool in outcome_pools.items()
}

# sanity check
for team_id, state in team_state.items():
    overall = get_rolling_sum(team_state, team_id, "points", league_avg)
    home_only = get_rolling_sum(team_state, team_id, "points", league_avg, venue="HOME")
    away_only = get_rolling_sum(team_state, team_id, "points", league_avg, venue="AWAY")
    print(team_id, state["source"], "overall:", round(overall, 1), "home:", round(home_only, 1), "away:", round(away_only, 1))