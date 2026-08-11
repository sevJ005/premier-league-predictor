import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from src.simulate.team_initialization import team_state, get_rolling_sum, league_avg, outcome_pools

match_features = pd.read_csv("data/processed/match_features.csv")
fixtures = pd.read_csv("data/raw/fixtures_2026-27.csv")

# odds columns removed entirely -- no synthetic odds needed, since the
# model was never trained on them in this version
feature_cols = [
    "matchday",

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
y_test = test["winner"]

scaler = StandardScaler()
scaled_train_data = scaler.fit_transform(X_train)
scaled_test_data = scaler.transform(X_test)

base_model = LogisticRegression()
model = CalibratedClassifierCV(base_model, method="sigmoid", cv=5)
model.fit(scaled_train_data, y_train)

# quick check: how does the no-odds model compare to the full model?
from sklearn.metrics import accuracy_score
prediction = model.predict(scaled_test_data)
print("No-odds model accuracy:", accuracy_score(y_test, prediction))


def simulate_match(home_team_id, away_team_id, team_state, matchday, model, scaler, feature_cols, outcome_pools, league_avg):
    def r(team_id, stat, venue=None):
        return get_rolling_sum(team_state, team_id, stat, league_avg, venue=venue)

    # build one feature row -- now using REAL venue-specific rolling form
    # for the *_by_venue features, instead of reusing overall form
    row = {
        "matchday": matchday,

        "rolling_points_5_home": r(home_team_id, "points"),
        "rolling_goals_for_5_home": r(home_team_id, "goals_for"),
        "rolling_goals_against_5_home": r(home_team_id, "goals_against"),
        "rolling_shots_for_5_home": r(home_team_id, "shots_for"),
        "rolling_shots_against_5_home": r(home_team_id, "shots_against"),
        "rolling_shots_target_for_5_home": r(home_team_id, "shots_target_for"),
        "rolling_shots_target_against_5_home": r(home_team_id, "shots_target_against"),
        "rolling_corners_for_5_home": r(home_team_id, "corners_for"),
        "rolling_corners_against_5_home": r(home_team_id, "corners_against"),
        "rolling_yellow_for_5_home": r(home_team_id, "yellow_for"),
        "rolling_yellow_against_5_home": r(home_team_id, "yellow_against"),
        "rolling_red_for_5_home": r(home_team_id, "red_for"),
        "rolling_red_against_5_home": r(home_team_id, "red_against"),
        "rolling_points_5_by_venue_home": r(home_team_id, "points", venue="HOME"),
        "rolling_goals_for_5_by_venue_home": r(home_team_id, "goals_for", venue="HOME"),
        "rolling_goals_against_5_by_venue_home": r(home_team_id, "goals_against", venue="HOME"),
        "rest_days_home": 7,  # documented simplification
        "long_layoff_home": False,
        "h2h_points_avg_3_home": 1.38,  # documented simplification

        "rolling_points_5_away": r(away_team_id, "points"),
        "rolling_goals_for_5_away": r(away_team_id, "goals_for"),
        "rolling_goals_against_5_away": r(away_team_id, "goals_against"),
        "rolling_shots_for_5_away": r(away_team_id, "shots_for"),
        "rolling_shots_against_5_away": r(away_team_id, "shots_against"),
        "rolling_shots_target_for_5_away": r(away_team_id, "shots_target_for"),
        "rolling_shots_target_against_5_away": r(away_team_id, "shots_target_against"),
        "rolling_corners_for_5_away": r(away_team_id, "corners_for"),
        "rolling_corners_against_5_away": r(away_team_id, "corners_against"),
        "rolling_yellow_for_5_away": r(away_team_id, "yellow_for"),
        "rolling_yellow_against_5_away": r(away_team_id, "yellow_against"),
        "rolling_red_for_5_away": r(away_team_id, "red_for"),
        "rolling_red_against_5_away": r(away_team_id, "red_against"),
        "rolling_points_5_by_venue_away": r(away_team_id, "points", venue="AWAY"),
        "rolling_goals_for_5_by_venue_away": r(away_team_id, "goals_for", venue="AWAY"),
        "rolling_goals_against_5_by_venue_away": r(away_team_id, "goals_against", venue="AWAY"),
        "rest_days_away": 7,
        "long_layoff_away": False,
        "h2h_points_avg_3_away": 1.38,
    }

    row_df = pd.DataFrame([row])[feature_cols]
    scaled_row = scaler.transform(row_df)

    probs = model.predict_proba(scaled_row)[0]
    outcome = np.random.choice(model.classes_, p=probs)

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


fixtures["date"] = pd.to_datetime(fixtures["date"])
fixtures = fixtures.sort_values("date").reset_index(drop=True)
fixtures["matchday"] = fixtures.index // 10 + 1


def simulate_season(fixtures, team_state, model, scaler, feature_cols, outcome_pools, league_avg):
    for _, match in fixtures.iterrows():
        home_id = match["home_team_id"]
        away_id = match["away_team_id"]
        matchday = match["matchday"]

        result = simulate_match(home_id, away_id, team_state, matchday, model, scaler, feature_cols, outcome_pools, league_avg)

        home_points_map = {"HOME_TEAM": 3, "DRAW": 1, "AWAY_TEAM": 0}
        home_new_match = {
            "points": home_points_map[result["outcome"]],
            "goals_for": result["home_goals"],
            "goals_against": result["away_goals"],
            "shots_for": result["home_shots"],
            "shots_against": result["away_shots"],
            "shots_target_for": result["home_shots_target"],
            "shots_target_against": result["away_shots_target"],
            "corners_for": result["home_corners"],
            "corners_against": result["away_corners"],
            "yellow_for": result["home_yellow"],
            "yellow_against": result["away_yellow"],
            "red_for": result["home_red"],
            "red_against": result["away_red"],
        }

        away_points_map = {"HOME_TEAM": 0, "DRAW": 1, "AWAY_TEAM": 3}
        away_new_match = {
            "points": away_points_map[result["outcome"]],
            "goals_for": result["away_goals"],
            "goals_against": result["home_goals"],
            "shots_for": result["away_shots"],
            "shots_against": result["home_shots"],
            "shots_target_for": result["away_shots_target"],
            "shots_target_against": result["home_shots_target"],
            "corners_for": result["away_corners"],
            "corners_against": result["home_corners"],
            "yellow_for": result["away_yellow"],
            "yellow_against": result["home_yellow"],
            "red_for": result["away_red"],
            "red_against": result["home_red"],
        }

        # update overall AND venue-specific histories
        team_state[home_id]["recent_matches"].append(home_new_match)
        if len(team_state[home_id]["recent_matches"]) > 5:
            team_state[home_id]["recent_matches"].pop(0)
        team_state[home_id]["recent_home_matches"].append(home_new_match)
        if len(team_state[home_id]["recent_home_matches"]) > 5:
            team_state[home_id]["recent_home_matches"].pop(0)
        team_state[home_id]["season_total_points"] += home_new_match["points"]

        team_state[away_id]["recent_matches"].append(away_new_match)
        if len(team_state[away_id]["recent_matches"]) > 5:
            team_state[away_id]["recent_matches"].pop(0)
        team_state[away_id]["recent_away_matches"].append(away_new_match)
        if len(team_state[away_id]["recent_away_matches"]) > 5:
            team_state[away_id]["recent_away_matches"].pop(0)
        team_state[away_id]["season_total_points"] += away_new_match["points"]

    return team_state


final_state = simulate_season(fixtures, team_state, model, scaler, feature_cols, outcome_pools, league_avg)
print("Season simulation complete.")

for team_id, state in final_state.items():
    print(team_id, "season total points:", state["season_total_points"])