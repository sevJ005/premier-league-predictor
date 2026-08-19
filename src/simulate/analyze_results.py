import json
import pandas as pd

with open("data/processed/simulation_results_50k_final.json", "r") as f:
    all_results = json.load(f)

results_df = pd.DataFrame(all_results)

top4_counts = {team_id: 0 for team_id in results_df.columns}

for _, row in results_df.iterrows():
    top4_teams = row.sort_values(ascending=False).head(4).index
    for team_id in top4_teams:
        top4_counts[team_id] += 1

top4_odds = pd.Series(top4_counts).sort_values(ascending=False) / len(results_df) * 100

relegation_counts = {team_id: 0 for team_id in results_df.columns}

for _, row in results_df.iterrows():
    bottom3_teams = row.sort_values(ascending=True).head(3).index
    for team_id in bottom3_teams:
        relegation_counts[team_id] += 1

relegation_odds = pd.Series(relegation_counts).sort_values(ascending=False) / len(results_df) * 100

winners = results_df.idxmax(axis=1)
title_odds = winners.value_counts(normalize=True) * 100

team_names = {
    57: "Arsenal", 58: "Aston Villa", 1044: "AFC Bournemouth", 402: "Brentford",
    397: "Brighton & Hove Albion", 61: "Chelsea", 9010: "Coventry City",
    354: "Crystal Palace", 62: "Everton", 63: "Fulham", 9011: "Hull City",
    9008: "Ipswich Town", 9005: "Leeds United", 64: "Liverpool", 65: "Manchester City",
    66: "Manchester United", 67: "Newcastle United", 351: "Nottingham Forest",
    9009: "Sunderland", 73: "Tottenham Hotspur",
}

summary = pd.DataFrame({
    "team": [team_names[int(t)] for t in results_df.columns],
    "avg_points": results_df.mean().values,
    "title_pct": title_odds.reindex(results_df.columns).values,
    "top4_pct": top4_odds.reindex(results_df.columns).values,
    "relegation_pct": relegation_odds.reindex(results_df.columns).values,
})

summary = summary.sort_values("top4_pct", ascending=False).reset_index(drop=True)

summary["avg_points"] = summary["avg_points"].round(1)
summary["title_pct"] = summary["title_pct"].round(1)
summary["top4_pct"] = summary["top4_pct"].round(1)
summary["relegation_pct"] = summary["relegation_pct"].round(1)

predicted_table = summary.sort_values("top4_pct", ascending=False).reset_index(drop=True)
predicted_table.insert(0, "predicted_position", range(1, 21))

print(predicted_table[["predicted_position", "team", "title_pct", "top4_pct", "relegation_pct"]])
predicted_table.to_csv("data/processed/predicted_final_table.csv", index=False)
print("Saved to data/processed/predicted_final_table.csv")

summary.to_csv("data/processed/season_predictions_summary.csv", index=False)
print("Saved to data/processed/season_predictions_summary.csv")