import sqlite3

connect = sqlite3.connect("db/pl_data.db")
cursor = connect.cursor()

new_teams = [
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
    (9008, "Ipswich Town FC", "Ipswich", "IPS"),
    (9009, "Sunderland AFC", "Sunderland", "SUN")
] 

for team in new_teams:
    cursor.execute("INSERT INTO teams (team_id, name, short_name, tla) VALUES (?, ?, ?, ?)", team)

connect.commit()
connect.close()