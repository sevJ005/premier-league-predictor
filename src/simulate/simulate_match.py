import pandas as pd
import numpy as np
import copy
import time
import warnings
import json
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from src.simulate.team_initialization import team_state, league_avg, outcome_arrays

warnings.filterwarnings("ignore", category=UserWarning)

match_features = pd.read_csv("data/processed/match_features.csv")
fixtures = pd.read_csv("data/raw/fixtures_2026-27.csv")

feature_cols = [
    "matchday",
    "implied_prob_home", "implied_prob_draw", "implied_prob_away",

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

scaler = StandardScaler()
scaled_train_data = scaler.fit_transform(X_train)

base_model = LogisticRegression()
model = CalibratedClassifierCV(base_model, method="sigmoid", cv=5)
model.fit(scaled_train_data, y_train)

def synthetic_implied_probs(home_rolling_points, away_rolling_points):
    diff = home_rolling_points - away_rolling_points
    home_advantage = 1 / (1 + 2.71828 ** (-diff / 5))
    draw_prob = 0.25 - abs(diff) * 0.01
    draw_prob = max(draw_prob, 0.10)
    home_prob = home_advantage * (1 - draw_prob)
    away_prob = (1 - home_advantage) * (1 - draw_prob)
    return away_prob, draw_prob, home_prob


ROLLING_STATS = [
    "points", "goals_for", "goals_against",
    "shots_for", "shots_against",
    "shots_target_for", "shots_target_against",
    "corners_for", "corners_against",
    "yellow_for", "yellow_against",
    "red_for", "red_against",
]

VENUE_STATS = ["points", "goals_for", "goals_against"]


def get_team_features(team_state, team_id, league_avg, venue):
    #Compute every rolling stat this team needs for one match, in one pass,
    # instead of calling get_rolling_sum() 16 separate times.
    state = team_state[team_id]
    is_real = state["source"] == "real_history"

    overall_matches = state["recent_matches"]
    venue_matches = state["recent_home_matches"] if venue == "HOME" else state["recent_away_matches"]

    overall = {}
    for stat in ROLLING_STATS:
        raw = sum(m[stat] for m in overall_matches)
        overall[stat] = 0.8 * raw + 0.2 * league_avg[stat] if is_real else raw

    venue_form = {}
    for stat in VENUE_STATS:
        raw = sum(m[stat] for m in venue_matches)
        venue_form[stat] = 0.8 * raw + 0.2 * league_avg[stat] if is_real else raw

    return overall, venue_form


def simulate_match(home_team_id, away_team_id, team_state, matchday, model, scaler, outcome_arrays, league_avg):
    home_overall, home_venue = get_team_features(team_state, home_team_id, league_avg, "HOME")
    away_overall, away_venue = get_team_features(team_state, away_team_id, league_avg, "AWAY")

    away_prob, draw_prob, home_prob = synthetic_implied_probs(home_overall["points"], away_overall["points"])

    row_values = np.array([[
        matchday,
        home_prob, draw_prob, away_prob,

        home_overall["points"], home_overall["goals_for"], home_overall["goals_against"],
        home_overall["shots_for"], home_overall["shots_against"],
        home_overall["shots_target_for"], home_overall["shots_target_against"],
        home_overall["corners_for"], home_overall["corners_against"],
        home_overall["yellow_for"], home_overall["yellow_against"],
        home_overall["red_for"], home_overall["red_against"],
        home_venue["points"], home_venue["goals_for"], home_venue["goals_against"],
        7, False, 1.38,

        away_overall["points"], away_overall["goals_for"], away_overall["goals_against"],
        away_overall["shots_for"], away_overall["shots_against"],
        away_overall["shots_target_for"], away_overall["shots_target_against"],
        away_overall["corners_for"], away_overall["corners_against"],
        away_overall["yellow_for"], away_overall["yellow_against"],
        away_overall["red_for"], away_overall["red_against"],
        away_venue["points"], away_venue["goals_for"], away_venue["goals_against"],
        7, False, 1.38,
    ]])

    scaled_row = scaler.transform(row_values)
    probs = model.predict_proba(scaled_row)[0]
    outcome = np.random.choice(model.classes_, p=probs)

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


def simulate_season(fixture_data, team_state, model, scaler, outcome_arrays, league_avg):
    for home_id, away_id, matchday in fixture_data:
        result = simulate_match(home_id, away_id, team_state, matchday, model, scaler, outcome_arrays, league_avg)

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

# benchmark seasons with the calibrated model 
start = time.time()
NUM_SIMULATIONS = 50000

all_results = []

start = time.time()
for i in range(NUM_SIMULATIONS):
    fresh_state = copy.deepcopy(team_state)
    final_state = simulate_season(fixture_data, fresh_state, model, scaler, outcome_arrays, league_avg)
    
    season_result = {team_id: state["season_total_points"] for team_id, state in final_state.items()}
    all_results.append(season_result)
    
    if (i + 1) % 500 == 0:
        elapsed = time.time() - start
        print(f"Completed {i + 1}/{NUM_SIMULATIONS} simulations ({elapsed/60:.1f} minutes elapsed)")
        with open("data/processed/simulation_results_50k_final.json", "w") as f:
            json.dump(all_results, f)

elapsed = time.time() - start
print(f"Finished all {NUM_SIMULATIONS} simulations in {elapsed/3600:.2f} hours")

with open("data/processed/simulation_results_50k_final.json", "w") as f:
    json.dump(all_results, f)
print("Results saved to data/processed/simulation_results_50k_final.json")