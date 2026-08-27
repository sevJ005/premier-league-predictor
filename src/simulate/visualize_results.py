import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

summary = pd.read_csv("data/processed/season_predictions_summary.csv")

# top 4 = blue, bot 3 = red, rest = grey
def get_colors(df, highlight_top=0, highlight_bottom=0):
    colors = []
    n = len(df)
    for i in range(n):
        if i < highlight_top:
            colors.append("#2563eb")  # blue
        elif i >= n - highlight_bottom:
            colors.append("#dc2626")  # red
        else:
            colors.append("#9ca3af")  # grey
    return colors


# Chart 1: Predicted table ranked by top-4 odds 

table = summary.sort_values("top4_pct", ascending=False).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(9, 8))
colors = get_colors(table, highlight_top=4, highlight_bottom=3)
bars = ax.barh(table["team"][::-1], table["top4_pct"][::-1], color=colors[::-1])

ax.set_xlabel("Top 4 Finish Probability (%)")
ax.set_title("2026/27 Premier League — Predicted Top 4 Odds\n(50,000 simulated seasons)", fontsize=13, fontweight="bold")
ax.xaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))

for bar, pct in zip(bars, table["top4_pct"][::-1]):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2, f"{pct:.1f}%", va="center", fontsize=9)

plt.tight_layout()
plt.savefig("data/processed/charts/predicted_table_top4.png", dpi=200)
plt.close()

# Chart 2: Title race (top 8 contenders) 

title_race = summary.sort_values("title_pct", ascending=False).head(8).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.barh(title_race["team"][::-1], title_race["title_pct"][::-1], color="#2563eb")

ax.set_xlabel("Title Win Probability (%)")
ax.set_title("2026/27 Premier League — Title Race\n(Top 8 contenders, 50,000 simulated seasons)", fontsize=13, fontweight="bold")

for bar, pct in zip(bars, title_race["title_pct"][::-1]):
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2, f"{pct:.1f}%", va="center", fontsize=9)

plt.tight_layout()
plt.savefig("data/processed/charts/title_race.png", dpi=200)
plt.close()

# Chart 3: Relegation risk (bottom 8) 

relegation_risk = summary.sort_values("relegation_pct", ascending=False).head(8).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.barh(relegation_risk["team"][::-1], relegation_risk["relegation_pct"][::-1], color="#dc2626")

ax.set_xlabel("Relegation Probability (%)")
ax.set_title("2026/27 Premier League — Relegation Risk\n(Highest-risk 8 teams, 50,000 simulated seasons)", fontsize=13, fontweight="bold")

for bar, pct in zip(bars, relegation_risk["relegation_pct"][::-1]):
    ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2, f"{pct:.1f}%", va="center", fontsize=9)

plt.tight_layout()
plt.savefig("data/processed/charts/relegation_risk.png", dpi=200)
plt.close()

print("Saved 3 charts to data/processed/charts/")