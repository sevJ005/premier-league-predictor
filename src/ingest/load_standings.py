import json
import sqlite3

connect = sqlite3.connect("db/pl_data.db")
cursor = connect.cursor()
cursor.execute("DELETE FROM standings")

with open("data/raw/pl_standings_2023.json", "r") as file:
    data = json.load(file)

for stand in data["standings"]:
    if stand["type"] != "TOTAL":
        continue    
    for row in stand["table"]:
        team_id = row["team"]["id"]
        position = row["position"]
        played = row["playedGames"]
        won = row["won"]
        draw = row["draw"]
        loss = row["lost"]
        points = row["points"]
        goals_for = row["goalsFor"]
        goals_against = row["goalsAgainst"]
        goal_difference = row["goalDifference"]
        form = row["form"]
        season_year = 2023
        cursor.execute("INSERT INTO standings (team_id, season_year, position, played, won, draw, loss, points, goals_for, goals_against, goal_difference, form) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (team_id, season_year, position, played, won, draw, loss, points, goals_for, goals_against, goal_difference, form))

connect.commit()
connect.close()