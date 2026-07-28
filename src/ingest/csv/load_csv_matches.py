import csv
import sqlite3

connect = sqlite3.connect("db/pl_data.db")
cursor = connect.cursor()

# rebuild the matches table fresh, using the updated schema
cursor.execute("DROP TABLE IF EXISTS matches")
with open("db/schema.sql", "r") as f:
    schema_sql = f.read()
cursor.executescript(schema_sql)

files = [
    ("pl_2018-2019.csv", 2018),
    ("pl_2019-2020.csv", 2019),
    ("pl_2020-2021.csv", 2020),
    ("pl_2021-2022.csv", 2021),
    ("pl_2022-2023.csv", 2022),
    ("pl_2023-2024.csv", 2023),
    ("pl_2024-2025.csv", 2024),
    ("pl_2025-2026.csv", 2025),
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
    "Luton": 389,
    "Ipswich": 9008,
    "Sunderland": 9009,
    "Leeds": 9005,
    "Watford": 9006,
    "Leicester": 9007
}

result_mapping = {
    "H": "HOME_TEAM",
    "A": "AWAY_TEAM",
    "D": "DRAW"
}

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
            referee = row["Referee"]

            home_shots = int(row["HS"])
            away_shots = int(row["AS"])
            home_shots_target = int(row["HST"])
            away_shots_target = int(row["AST"])
            home_fouls = int(row["HF"])
            away_fouls = int(row["AF"])
            home_corners = int(row["HC"])
            away_corners = int(row["AC"])
            home_yellow = int(row["HY"])
            away_yellow = int(row["AY"])
            home_red = int(row["HR"])
            away_red = int(row["AR"])

            b365_home_odds = float(row["B365H"])
            b365_draw_odds = float(row["B365D"])
            b365_away_odds = float(row["B365A"])

            cursor.execute("""
                INSERT INTO matches (
                    match_id, season_year, matchday, utc_date, status,
                    home_team_id, away_team_id,
                    home_goals_fulltime, away_goals_fulltime,
                    home_goals_halftime, away_goals_halftime,
                    home_shots, away_shots, home_shots_target, away_shots_target,
                    home_fouls, away_fouls, home_corners, away_corners,
                    home_yellow, away_yellow, home_red, away_red,
                    referee, b365_home_odds, b365_draw_odds, b365_away_odds,
                    winner
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                match_id, season_year, None, utc_date, status,
                home_team_id, away_team_id,
                home_fulltime, away_fulltime,
                home_halftime, away_halftime,
                home_shots, away_shots, home_shots_target, away_shots_target,
                home_fouls, away_fouls, home_corners, away_corners,
                home_yellow, away_yellow, home_red, away_red,
                referee, b365_home_odds, b365_draw_odds, b365_away_odds,
                winner
            ))

connect.commit()
connect.close()