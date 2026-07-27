import os 
import json
import requests
import time
from dotenv import load_dotenv

#loading the .env file
load_dotenv(dotenv_path="config/.env")

#read api key
api_key = os.getenv("FOOTBALL_DATA_API_KEY")
print("Key loaded:", api_key is not None)

# build request for all season
teams = "https://api.football-data.org/v4/competitions/PL/teams"
matches =  "https://api.football-data.org/v4/competitions/PL/matches"
standings = "https://api.football-data.org/v4/competitions/PL/standings"

seasons = [2023, 2024, 2025]

for season in seasons:
    params = {"season": season}
    headers = {"X-Auth-Token": api_key}

    #calling api
    teams_response = requests.get(teams, headers=headers, params=params)
    teams_data = teams_response.json()
    time.sleep(5)

    matches_response = requests.get(matches, headers=headers, params=params)
    matches_data = matches_response.json()
    time.sleep(5)

    standings_response = requests.get(standings, headers=headers, params=params)
    standings_data = standings_response.json()
    time.sleep(5)

    #checking status
    if teams_response.status_code != 200:
        print(f"Failed to fetch teams for {season}: {teams_response.status_code}")

    if matches_response.status_code != 200:
        print(f"Failed to fetch matches for {season}: {matches_response.status_code}")

    if standings_response.status_code != 200:
        print(f"Failed to fetch standings for {season}: {standings_response.status_code}")

    with open(f"data/raw/pl_teams_{season}.json", "w") as f:
        json.dump(teams_data, f, indent=2)

    with open(f"data/raw/pl_matches_{season}.json", "w") as f:
        json.dump(matches_data, f, indent=2)

    with open(f"data/raw/pl_standings_{season}.json", "w") as f:
        json.dump(standings_data, f, indent=2)

print("----- Finished fetching all seasons. -----")