import os 
import json
import requests
from dotenv import load_dotenv

#loading the .env file
load_dotenv(dotenv_path="config/.env")

#read api key
api_key = os.getenv("FOOTBALL_DATA_API_KEY")
print("Key loaded:", api_key is not None)

# build request for 2023 season
url = "https://api.football-data.org/v4/competitions/PL/matches" 
params = {"season": 2023}

headers = {"X-Auth-Token": api_key}

#calling api
response = requests.get(url, headers=headers, params=params)

#checking status
print("Status code:", response.status_code)

data = response.json()

print("Top level keys:", data.keys())

with open("data/raw/pl_matches_2023.json", "w") as f:
    json.dump(data, f, indent=2)

print("Saved full response to data/raw/pl_matches_2023.json")