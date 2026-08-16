import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

match_features = pd.read_csv("data/processed/match_features.csv")

# small feature set: just points and goals for/against, matching what
# synthetic_implied_probs currently has available during simulation
prob_feature_cols = [
    "rolling_points_5_home", "rolling_points_5_away",
    "rolling_goals_for_5_home", "rolling_goals_against_5_home",
    "rolling_goals_for_5_away", "rolling_goals_against_5_away",
]

X = match_features[prob_feature_cols]
y = match_features["winner"]

prob_model = LogisticRegression()
prob_model.fit(X, y)

print("Classes:", prob_model.classes_)

# ---- test on a range of scenarios, from even to a big mismatch ----

test_cases = [
    {"label": "Evenly matched", "home_points": 8, "away_points": 8, "home_gf": 6, "home_ga": 6, "away_gf": 6, "away_ga": 6},
    {"label": "Moderate home edge", "home_points": 12, "away_points": 6, "home_gf": 9, "home_ga": 4, "away_gf": 5, "away_ga": 8},
    {"label": "Big home edge", "home_points": 15, "away_points": 2, "home_gf": 13, "home_ga": 2, "away_gf": 2, "away_ga": 13},
    {"label": "Extreme mismatch", "home_points": 15, "away_points": 0, "home_gf": 15, "home_ga": 1, "away_gf": 1, "away_ga": 15},
]

for case in test_cases:
    row = np.array([[
        case["home_points"], case["away_points"],
        case["home_gf"], case["home_ga"],
        case["away_gf"], case["away_ga"],
    ]])
    probs = prob_model.predict_proba(row)[0]
    class_probs = dict(zip(prob_model.classes_, probs))
    print(f"\n{case['label']}:")
    print(f"  Home win: {class_probs['HOME_TEAM']:.1%}")
    print(f"  Draw:     {class_probs['DRAW']:.1%}")
    print(f"  Away win: {class_probs['AWAY_TEAM']:.1%}")