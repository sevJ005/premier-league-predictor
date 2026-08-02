import sqlite3

connect = sqlite3.connect("db/pl_data.db")
cursor = connect.cursor()

cursor.execute("DELETE FROM teams")

all_teams = [
    # Originally sourced from football-data.org's API (2023/24 season)
    (57, "Arsenal FC", "Arsenal", "ARS"),
    (58, "Aston Villa FC", "Aston Villa", "AVL"),
    (61, "Chelsea FC", "Chelsea", "CHE"),
    (62, "Everton FC", "Everton", "EVE"),
    (63, "Fulham FC", "Fulham", "FUL"),
    (64, "Liverpool FC", "Liverpool", "LIV"),
    (65, "Manchester City FC", "Man City", "MCI"),
    (66, "Manchester United FC", "Man United", "MUN"),
    (67, "Newcastle United FC", "Newcastle", "NEW"),
    (73, "Tottenham Hotspur FC", "Tottenham", "TOT"),
    (76, "Wolverhampton Wanderers FC", "Wolves", "WOL"),
    (328, "Burnley FC", "Burnley", "BUR"),
    (351, "Nottingham Forest FC", "Nott'm Forest", "NFO"),
    (354, "Crystal Palace FC", "Crystal Palace", "CRY"),
    (356, "Sheffield United FC", "Sheffield United", "SHU"),
    (389, "Luton Town FC", "Luton", "LUT"),
    (397, "Brighton & Hove Albion FC", "Brighton", "BHA"),
    (402, "Brentford FC", "Brentford", "BRE"),
    (563, "West Ham United FC", "West Ham", "WHU"),
    (1044, "AFC Bournemouth", "Bournemouth", "BOU"),

    # Self-assigned IDs — teams not covered by the API's free tier (relegated/promoted
    # in seasons outside 2023-2025), added manually to support CSV-sourced match data
    (9000, "Norwich City FC", "Norwich", "NOR"),
    (9001, "Southampton FC", "Southampton", "SOU"),
    (9002, "West Bromwich Albion FC", "West Brom", "WBA"),
    (9003, "Cardiff City FC", "Cardiff", "CDF"),
    (9004, "Huddersfield FC", "Huddersfield", "HDF"),
    (9005, "Leeds United FC", "Leeds", "LDS"),
    (9006, "Watford FC", "Watford", "WFD"),
    (9007, "Leicester City FC", "Leicester", "LEI"),
    (9008, "Ipswich Town FC", "Ipswich", "IPS"),
    (9009, "Sunderland AFC", "Sunderland", "SUN"),
]

for team in all_teams:
    cursor.execute(
        "INSERT INTO teams (team_id, name, short_name, tla) VALUES (?, ?, ?, ?)",
        team
    )

connect.commit()
connect.close()

print(f"Loaded {len(all_teams)} teams.")