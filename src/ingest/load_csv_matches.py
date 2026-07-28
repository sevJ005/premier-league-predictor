import csv
import sqlite3

connect = sqlite3.connect("db/pl_data.db")
cursor = connect.cursor()

files = [
    ("pl_2018-2019.csv", 2018),
    ("pl_2019-2020.csv", 2019),
    ("pl_2020-2021.csv", 2020),
    ("pl_2021-2022.csv", 2021),
    ("pl_2022-2023.csv", 2022),
]

team_name_mapping = {
    "Arsenal": 57,
    "Aston Villa": 58,
    "Chelsea": 61,
    "Everton": 62,
    "Fulham": 63,
    "Liverpool": 64,
    "Man City": 65,
    "Man United": 66,
    "Newcastle": 67,
    "Tottenham": 73,
    "Wolves": 76,
    "Burnley": 328,
    "Nott'm Forest": 351,
    "Crystal Palace": 354,
    "Sheffield United": 356,
    "Norwich": 9000,
    "Brighton": 397,
    "Brentford": 402,
    "West Ham": 563,
    "Bournemouth": 1044,
    "Southampton": 9001,
    "West Brom": 9002,
    "Cardiff": 9003,
    "Huddersfield": 9004,
    "Leeds": 9005,
    "Watford": 9006,
    "Leicester": 9007
}

result_mapping = {
    "H": "HOME_TEAM",
    "A": "AWAY_TEAM",
    "D": "DRAW"
}
#generating match ID since CSV doesn't provide
match_id_counter = 20000  

for file, season_year in files:
    with open(f"data/raw/{file}", "r") as doc:
        data = csv.DictReader(doc)

        for row in data:
            home_team_id = team_name_mapping[row["HomeTeam"]]
            away_team_id = team_name_mapping[row["AwayTeam"]]
            home_fulltime = int(row["FTHG"])
            away_fulltime = int(row["FTAG"])
            home_halftime = int(row["HTHG"])
            away_halftime = int(row["HTAG"])
            winner = result_mapping[row["FTR"]]
            match_id = match_id_counter
            match_id_counter += 1
            status = "FINISHED"
            utc_date = row["Date"]
            cursor.execute("INSERT INTO matches (match_id, season_year, matchday, utc_date, status, home_team_id, away_team_id, home_goals_fulltime, away_goals_fulltime, home_goals_halftime, away_goals_halftime, winner) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (match_id, season_year, None, utc_date, status, home_team_id, away_team_id, home_fulltime, away_fulltime, home_halftime, away_halftime, winner)
)
connect.commit()
connect.close()
