import json
import sqlite3

connect = sqlite3.connect("db/pl_data.db")
cursor = connect.cursor()

with open("data/raw/pl_teams_2023.json", "r") as file:
    data = json.load(file)

for team in data["teams"]:
    team_id = team["id"]
    name = team["name"]
    short_name = team["shortName"]
    tla = team["tla"]
    cursor.execute("INSERT INTO teams (team_id, name, short_name, tla) VALUES (?, ?, ?, ?)", (team_id, name, short_name, tla))

connect.commit()
connect.close()
