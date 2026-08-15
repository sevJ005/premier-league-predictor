import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from src.simulate.team_initialization import team_state, get_rolling_sum, league_avg, outcome_pools, outcome_arrays
import copy, time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

match_features = pd.read_csv("data/processed/match_features.csv")
fixtures = pd.read_csv("data/raw/fixtures_2026-27.csv")

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

# calibrated version
base_model = LogisticRegression()
model = CalibratedClassifierCV(base_model, method="sigmoid", cv=5)
model.fit(scaled_train_data, y_train)

# uncalibrated version, for speed comparison
plain_model = LogisticRegression()
plain_model.fit(scaled_train_data, y_train)

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

def simulate_match(home_team_id, away_team_id, team_state, matchday, model, scaler, feature_cols, outcome_pools, league_avg):
    def r(team_id, stat, venue=None):
        return get_rolling_sum(team_state, team_id, stat, league_avg, venue=venue)

    home_points = r(home_team_id, "points")
    away_points = r(away_team_id, "points")

    # synthetic odds, derived from current rolling points
    # stands in for real market odds, which don't exist yet for a future match
    away_prob, draw_prob, home_prob = synthetic_implied_probs(home_points, away_points)

    # build one feature row, matching feature_cols exactly
    row = {
        "matchday": matchday,
        "implied_prob_home": home_prob,
        "implied_prob_draw": draw_prob,
        "implied_prob_away": away_prob,

        "rolling_points_5_home": home_points,
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
        "rest_days_home": 7,  # simplification, could build on afterward
        "long_layoff_home": False,
        "h2h_points_avg_3_home": 1.38,  # simplification, could build on afterward

        "rolling_points_5_away": away_points,
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

    row_values = np.array([[row[col] for col in feature_cols]])
    scaled_row = scaler.transform(row_values)

    probs = model.predict_proba(scaled_row)[0]
    outcome = np.random.choice(model.classes_, p=probs)

    # borrow a real historical stat-line matching this outcome
    pool = outcome_arrays[outcome]
    borrowed = pool[np.random.randint(len(pool))]

    return {
        "outcome": outcome,
        "home_goals": borrowed[0],
        "away_goals": borrowed[1],
        "home_shots": borrowed[2],
        "away_shots": borrowed[3],
        "home_shots_target": borrowed[4],
        "away_shots_target": borrowed[5],
        "home_corners": borrowed[6],
        "away_corners": borrowed[7],
        "home_yellow": borrowed[8],
        "away_yellow": borrowed[9],
        "home_red": borrowed[10],
        "away_red": borrowed[11],
    }

fixtures["date"] = pd.to_datetime(fixtures["date"])
fixtures = fixtures.sort_values("date").reset_index(drop=True)
fixtures["matchday"] = fixtures.index + 1

fixture_data = list(
    fixtures[["home_team_id", "away_team_id", "matchday"]].itertuples(index=False, name=None)
)

def simulate_season(fixture_data, team_state, model, scaler, feature_cols, outcome_pools, league_avg):
    for home_id, away_id, matchday in fixture_data:

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

        for team_id, new_match in [(home_id, home_new_match), (away_id, away_new_match)]:
            team_state[team_id]["recent_matches"].append(new_match)
            if len(team_state[team_id]["recent_matches"]) > 5:
                team_state[team_id]["recent_matches"].pop(0)
            team_state[team_id]["season_total_points"] += new_match["points"]

    return team_state

start = time.time()
full_results = []

ran = 50
for i in range(ran):
    fresh_state = copy.deepcopy(team_state)
    final_state= simulate_season(fixture_data, fresh_state, model, scaler, feature_cols, outcome_pools, league_avg)

    season_result = {team_id: state["season_total_points"] for team_id, state in final_state.items()}
    full_results.append(season_result)

elapsed = time.time() - start
print(f"Calibrated: {elapsed:.1f}s for 50 seasons")

start = time.time()
for i in range(ran):
    fresh_state = copy.deepcopy(team_state)
    simulate_season(fixture_data, fresh_state, plain_model, scaler, feature_cols, outcome_pools, league_avg)
plain_time = time.time() - start


# final_state = simulate_season(fixtures, team_state, model, scaler, feature_cols, outcome_pools, league_avg)
print("Season simulation complete.")
"""
for team_id, state in final_state.items():
    print(team_id, "final rolling points:", get_rolling_sum(final_state, team_id, "points", league_avg))
print("-------------------------------")
for team_id, state in final_state.items():
    print(team_id, "season total points:", state["season_total_points"])
"""

print(f"Plain: {plain_time:.1f}s for 50 seasons")
print(f"Speedup: {elapsed / plain_time:.1f}x")
