import json
import sqlite3

connect = sqlite3.connect("db/pl_data.db")
cursor = connect.cursor()
years = ["2023", "2024", "2025"]

for year in years:
    with open(f"data/raw/pl_matches_{year}.json", "r") as file:
        data = json.load(file)

    for match in data["matches"]:
        match_id = match["id"]
        season = int(year)
        matchday = match["matchday"]
        utcDate = match["utcDate"]
        status = match["status"]
        homeTeam_id = match["homeTeam"]["id"]
        awayTeam_id = match["awayTeam"]["id"]
        score_fullTime_home = match["score"]["fullTime"]["home"]
        score_fullTime_away = match["score"]["fullTime"]["away"]
        score_halfTime_home = match["score"]["halfTime"]["home"]
        score_halfTime_away = match["score"]["halfTime"]["away"]
        score_winner = match["score"]["winner"]
        cursor.execute("INSERT INTO matches (match_id, season_year, matchday, utc_date, status, home_team_id, away_team_id, home_goals_fulltime, away_goals_fulltime, home_goals_halftime, away_goals_halftime, winner) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", ( match_id, season, matchday, utcDate, status, homeTeam_id, awayTeam_id, score_fullTime_home, score_fullTime_away, score_halfTime_home, score_halfTime_away, score_winner))

connect.commit()
connect.close()
