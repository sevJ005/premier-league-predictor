import pandas as pd
import sqlite3

# connect and pull data
connect = sqlite3.connect("db/pl_data.db")
cursor = connect.cursor()
matches = pd.read_sql_query("SELECT * FROM matches", connect)

# reshape: one row per team per match, instead of one row per match 

# home DataFrame
home = matches[["season_year", "home_team_id", "home_goals_fulltime", "away_goals_fulltime", "winner"]].copy()
home.columns = ["season_year", "team_id", "goals_for", "goals_against", "winner"]

# away DataFrame
away = matches[["season_year", "away_team_id", "away_goals_fulltime", "home_goals_fulltime", "winner"]].copy()
away.columns = ["season_year", "team_id", "goals_for", "goals_against", "winner"]

# translate the raw "winner" field (HOME_TEAM/AWAY_TEAM/DRAW) into each team's
# own personal result (WIN/LOSS/DRAW)
home_result_map = {
    "HOME_TEAM": "WIN",
    "AWAY_TEAM": "LOSS",
    "DRAW": "DRAW"
}

away_result_map = {
    "HOME_TEAM": "LOSS",
    "AWAY_TEAM": "WIN",
    "DRAW": "DRAW"
}

home["result"] = home["winner"].map(home_result_map)
away["result"] = away["winner"].map(away_result_map)

# stack home and away perspectives into one table
team_matches = pd.concat([home, away], ignore_index=True)

# collapse match-level rows into one summary row per team per season 
seasons = team_matches.groupby(["team_id", "season_year"]).agg(
    matches_played=("winner", "count"),
    total_goals_for=("goals_for", "sum"),
    total_goals_against=("goals_against", "sum"),
    wins=("result", lambda x: (x == "WIN").sum()),
    draws=("result", lambda x: (x == "DRAW").sum()),
    losses=("result", lambda x: (x == "LOSS").sum()),
)

seasons["points"] = seasons["wins"] * 3 + seasons["draws"]
seasons["goal_difference"] = seasons["total_goals_for"] - seasons["total_goals_against"]

# rank teams within each season 
# sort by season, then points (highest first), then goal difference as the
# tiebreaker (also highest first)
seasons = seasons.sort_values(
    ["season_year", "points", "goal_difference"],
    ascending=[True, False, False]
)

# running count  + 1 to make it accurate to an actual table
seasons["position"] = seasons.groupby("season_year").cumcount() + 1

# reorder columns for readability (purely cosmetic, doesn't affect the data)
seasons = seasons[[
    "matches_played", "total_goals_for", "total_goals_against",
    "goal_difference", "wins", "draws", "losses", "points", "position"
]]

# turn team_id/season_year back from the groupby index into regular columns
seasons_flat = seasons.reset_index()

# wipe out old standings data 
cursor.execute("DELETE FROM standings")

for _, row in seasons_flat.iterrows():
    cursor.execute("""
        INSERT INTO standings (
            team_id, season_year, position, played, won, draw, loss,
            points, goals_for, goals_against, goal_difference, form
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        int(row["team_id"]), int(row["season_year"]), int(row["position"]), int(row["matches_played"]),
        int(row["wins"]), int(row["draws"]), int(row["losses"]), int(row["points"]),
        int(row["total_goals_for"]), int(row["total_goals_against"]), int(row["goal_difference"]),
        None  # "form" (last 5 results, e.g. "W,W,D,L,W") 
    ))

connect.commit()
connect.close()

print(f"Inserted {len(seasons_flat)} team-season standings rows.")