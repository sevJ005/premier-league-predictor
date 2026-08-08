import re
import csv

# full-name -> team_id mapping, matching the names used in the official
# fixture list (different style than the short CSV names used elsewhere)
fixture_team_mapping = {
    "Arsenal": 57,
    "Aston Villa": 58,
    "AFC Bournemouth": 1044,
    "Brentford": 402,
    "Brighton & Hove Albion": 397,
    "Chelsea": 61,
    "Coventry City": 9010,
    "Crystal Palace": 354,
    "Everton": 62,
    "Fulham": 63,
    "Hull City": 9011,
    "Ipswich Town": 9008,
    "Leeds United": 9005,
    "Liverpool": 64,
    "Manchester City": 65,
    "Manchester United": 66,
    "Newcastle United": 67,
    "Nottingham Forest": 351,
    "Sunderland": 9009,
    "Tottenham Hotspur": 73,
}

months = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
}

weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# matches lines like "Friday 21 August 2026" or "Saturday 22 August" (year omitted)
date_pattern = re.compile(
    r"^(" + "|".join(weekdays) + r")\s+(\d{1,2})\s+([A-Za-z]+)(?:\s+(\d{4}))?$"
)

# matches a fixture line, with optional leading time and trailing broadcaster/footnote junk
fixture_pattern = re.compile(
    r"^(?:(\d{1,2}:\d{2})\s+)?(.+?)\s+v\s+(.+?)(?:\s*\(.*\))?(?:\*+)?$"
)

fixtures = []
current_date = None
current_year = 2026 
with open("data/raw/fixtures_2026-27.txt", "r", encoding="utf-8") as f:
    for raw_line in f:
        line = raw_line.strip()
        if not line:
            continue

        date_match = date_pattern.match(line)
        if date_match:
            weekday, day, month_name, year = date_match.groups()
            if year:
                current_year = int(year)
            month = months.get(month_name)
            if month:
                current_date = f"{current_year}-{month:02d}-{int(day):02d}"
            else:
                current_date = None  # unrecognized month name, skip fixtures until next valid date
            continue

        if current_date is None:
            continue  # skip stray lines (footnotes, headers) before any date is set

        fixture_match = fixture_pattern.match(line)
        if fixture_match:
            _, home_name, away_name = fixture_match.groups()
            home_name = home_name.strip()
            away_name = away_name.strip()

            home_id = fixture_team_mapping.get(home_name)
            away_id = fixture_team_mapping.get(away_name)

            if home_id is not None and away_id is not None:
                fixtures.append({
                    "date": current_date,
                    "home_team": home_name,
                    "away_team": away_name,
                    "home_team_id": home_id,
                    "away_team_id": away_id,
                })
            else:
                print(f"Skipped (unmatched team name): {line}")

print(f"Parsed {len(fixtures)} fixtures")

with open("data/raw/fixtures_2026-27.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["date", "home_team", "away_team", "home_team_id", "away_team_id"])
    writer.writeheader()
    writer.writerows(fixtures)

print("Saved to data/raw/fixtures_2026-27.csv")