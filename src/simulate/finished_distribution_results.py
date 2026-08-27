import json
import pandas as pd
import matplotlib.pyplot as plt

with open("data/processed/simulation_results_50k_final.json", "r") as f:
    all_results = json.load(f)

results_df = pd.DataFrame(all_results)
ranks = results_df.rank(axis=1, ascending=False, method="first")

team_names = {
    57: "Arsenal", 58: "Aston Villa", 1044: "AFC Bournemouth", 402: "Brentford",
    397: "Brighton", 61: "Chelsea", 9010: "Coventry City",
    354: "Crystal Palace", 62: "Everton", 63: "Fulham", 9011: "Hull City",
    9008: "Ipswich Town", 9005: "Leeds United", 64: "Liverpool", 65: "Manchester City",
    66: "Manchester United", 67: "Newcastle United", 351: "Nottingham Forest",
    9009: "Sunderland", 73: "Tottenham",
}

def tier(pos):
    if pos <= 4:
        return "Top 4"
    elif pos <= 7:
        return "European spots (5-7)"
    elif pos <= 17:
        return "Mid-table (8-17)"
    else:
        return "Relegation zone (18-20)"

tier_counts = {}
for team_id in ranks.columns:
    counts = ranks[team_id].apply(tier).value_counts(normalize=True) * 100
    tier_counts[team_names[int(team_id)]] = counts

tier_df = pd.DataFrame(tier_counts).T.fillna(0)
tier_order = ["Top 4", "European spots (5-7)", "Mid-table (8-17)", "Relegation zone (18-20)"]
tier_df = tier_df[tier_order]

# sort teams by Top 4 % for a sensible visual order
tier_df = tier_df.sort_values("Top 4", ascending=True)

colors = ["#2563eb", "#60a5fa", "#9ca3af", "#dc2626"]

fig, ax = plt.subplots(figsize=(10, 9))
tier_df.plot(kind="barh", stacked=True, color=colors, ax=ax, width=0.75)

ax.set_xlabel("Share of Simulated Seasons (%)")
ax.set_title("2026/27 Premier League — Predicted Finish Distribution\n(50,000 simulated seasons)", fontsize=13, fontweight="bold")
ax.legend(loc="lower right", fontsize=8)
ax.set_xlim(0, 100)

plt.tight_layout()
plt.savefig("data/processed/charts/finish_distribution.png", dpi=200)
plt.close()

print("Saved finish_distribution.png")