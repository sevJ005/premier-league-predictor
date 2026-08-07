import sqlite3
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

match_features = pd.read_csv("data/processed/match_features.csv")

feature_cols = [
    "matchday",
    "implied_prob_home", "implied_prob_draw", "implied_prob_away",

    # home side rolling form
    "rolling_points_5_home",
    "rolling_goals_for_5_home", "rolling_goals_against_5_home",
    "rolling_shots_for_5_home", "rolling_shots_against_5_home",
    "rolling_shots_target_for_5_home", "rolling_shots_target_against_5_home",
    "rolling_corners_for_5_home", "rolling_corners_against_5_home",
    "rolling_yellow_for_5_home", "rolling_yellow_against_5_home",
    "rolling_red_for_5_home", "rolling_red_against_5_home",
    "rolling_points_5_by_venue_home",
    "rolling_goals_for_5_by_venue_home", "rolling_goals_against_5_by_venue_home",
    "rest_days_home", "long_layoff_home",
    "h2h_points_avg_3_home",

    # away side rolling form
    "rolling_points_5_away",
    "rolling_goals_for_5_away", "rolling_goals_against_5_away",
    "rolling_shots_for_5_away", "rolling_shots_against_5_away",
    "rolling_shots_target_for_5_away", "rolling_shots_target_against_5_away",
    "rolling_corners_for_5_away", "rolling_corners_against_5_away",
    "rolling_yellow_for_5_away", "rolling_yellow_against_5_away",
    "rolling_red_for_5_away", "rolling_red_against_5_away",
    "rolling_points_5_by_venue_away",
    "rolling_goals_for_5_by_venue_away", "rolling_goals_against_5_by_venue_away",
    "rest_days_away", "long_layoff_away",
    "h2h_points_avg_3_away",
]

train = match_features[match_features["season_year"].isin([2018, 2019, 2020, 2021, 2022, 2023, 2024])]
test = match_features[match_features["season_year"] == 2025]

X_train = train[feature_cols]
y_train = train["winner"]
X_test = test[feature_cols]

scaler = StandardScaler()
scaled_train_data = scaler.fit_transform(X_train)
scaled_test_data = scaler.transform(X_test)

base_model = LogisticRegression()
model = CalibratedClassifierCV(base_model, method="sigmoid", cv=5)
model.fit(scaled_train_data, y_train)

# test the sampling mechanism on one real match's probabilities
probabilities = model.predict_proba(scaled_test_data)

# adding fake odds for new season matches based off larger rolling point average / momentum + H2H rolling  
def synthetic_implied_probs(home_rolling_points, away_rolling_points):
    diff = home_rolling_points - away_rolling_points
    
    # logistic curve: bigger point advantage -> higher win probability
    home_advantage = 1 / (1 + 2.71828 ** (-diff / 5))
    
    # crude 3-way split: push toward home/away based on advantage,
    # keep a baseline draw probability that shrinks slightly as the
    # gap between teams grows (blowouts are less likely to end level)
    draw_prob = 0.25 - abs(diff) * 0.01
    draw_prob = max(draw_prob, 0.10)  # never let draw probability vanish entirely
    
    home_prob = home_advantage * (1 - draw_prob)
    away_prob = (1 - home_advantage) * (1 - draw_prob)
    
    return away_prob, draw_prob, home_prob

def simulate_match(home_team_id, away_team_id, team_state, matchday, model, scaler, feature_cols, outcome_pools):
    home = team_state[home_team_id]
    away = team_state[away_team_id]

    # synthetic odds, derived from current rolling points 
    # stands in for real market odds, which don't exist yet for a future match
    away_prob, draw_prob, home_prob = synthetic_implied_probs(
        home["rolling_points_5"], away["rolling_points_5"]
    )

    # build one feature row, matching feature_cols exactly
    row = {
        "matchday": matchday,
        "implied_prob_home": home_prob,
        "implied_prob_draw": draw_prob,
        "implied_prob_away": away_prob,

        "rolling_points_5_home": home["rolling_points_5"],
        "rolling_goals_for_5_home": home["rolling_goals_for_5"],
        "rolling_goals_against_5_home": home["rolling_goals_against_5"],
        "rolling_shots_for_5_home": home["rolling_shots_for_5"],
        "rolling_shots_against_5_home": home["rolling_shots_against_5"],
        "rolling_shots_target_for_5_home": home["rolling_shots_target_for_5"],
        "rolling_shots_target_against_5_home": home["rolling_shots_target_against_5"],
        "rolling_corners_for_5_home": home["rolling_corners_for_5"],
        "rolling_corners_against_5_home": home["rolling_corners_against_5"],
        "rolling_yellow_for_5_home": home["rolling_yellow_for_5"],
        "rolling_yellow_against_5_home": home["rolling_yellow_against_5"],
        "rolling_red_for_5_home": home["rolling_red_for_5"],
        "rolling_red_against_5_home": home["rolling_red_against_5"],
        "rolling_points_5_by_venue_home": home["rolling_points_5"],  # simplification, could build on afterward
        "rolling_goals_for_5_by_venue_home": home["rolling_goals_for_5"],
        "rolling_goals_against_5_by_venue_home": home["rolling_goals_against_5"],
        "rest_days_home": 7,  # simplification could build on afterward
        "long_layoff_home": False,
        "h2h_points_avg_3_home": 1.38,  # simplification, could build on afterward 

        "rolling_points_5_away": away["rolling_points_5"],
        "rolling_goals_for_5_away": away["rolling_goals_for_5"],
        "rolling_goals_against_5_away": away["rolling_goals_against_5"],
        "rolling_shots_for_5_away": away["rolling_shots_for_5"],
        "rolling_shots_against_5_away": away["rolling_shots_against_5"],
        "rolling_shots_target_for_5_away": away["rolling_shots_target_for_5"],
        "rolling_shots_target_against_5_away": away["rolling_shots_target_against_5"],
        "rolling_corners_for_5_away": away["rolling_corners_for_5"],
        "rolling_corners_against_5_away": away["rolling_corners_against_5"],
        "rolling_yellow_for_5_away": away["rolling_yellow_for_5"],
        "rolling_yellow_against_5_away": away["rolling_yellow_against_5"],
        "rolling_red_for_5_away": away["rolling_red_for_5"],
        "rolling_red_against_5_away": away["rolling_red_against_5"],
        "rolling_points_5_by_venue_away": away["rolling_points_5"],
        "rolling_goals_for_5_by_venue_away": away["rolling_goals_for_5"],
        "rolling_goals_against_5_by_venue_away": away["rolling_goals_against_5"],
        "rest_days_away": 7,
        "long_layoff_away": False,
        "h2h_points_avg_3_away": 1.38,
    }

    row_df = pd.DataFrame([row])[feature_cols]
    scaled_row = scaler.transform(row_df)

    probs = model.predict_proba(scaled_row)[0]
    outcome = np.random.choice(model.classes_, p=probs)

    # borrow a real historical stat-line matching this outcome
    borrowed = outcome_pools[outcome].sample(1).iloc[0]

    return {
        "outcome": outcome,
        "home_goals": borrowed["home_goals_fulltime"],
        "away_goals": borrowed["away_goals_fulltime"],
        "home_shots": borrowed["home_shots"],
        "away_shots": borrowed["away_shots"],
        "home_shots_target": borrowed["home_shots_target"],
        "away_shots_target": borrowed["away_shots_target"],
        "home_corners": borrowed["home_corners"],
        "away_corners": borrowed["away_corners"],
        "home_yellow": borrowed["home_yellow"],
        "away_yellow": borrowed["away_yellow"],
        "home_red": borrowed["home_red"],
        "away_red": borrowed["away_red"],
    }
