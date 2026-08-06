from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

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

# chronological split -- train on 2018-2024, test on 2025
train = match_features[match_features["season_year"].isin([2018, 2019, 2020, 2021, 2022, 2023, 2024])]
test = match_features[match_features["season_year"] == 2025]

X_train = train[feature_cols]
y_train = train["winner"]

X_test = test[feature_cols]
y_test = test["winner"]

scaler = StandardScaler()
scaled_train_data = scaler.fit_transform(X_train)
scaled_test_data = scaler.transform(X_test)

# wrap an unfit LogisticRegression in a calibrator: sigmoid instead of isotonic because of the modest dataset size.
# CalibratedClassifierCV internally splits the training data further to learn the correction, so we just fit it once on the full training set.
base_model = LogisticRegression()
calibrated_model = CalibratedClassifierCV(base_model, method="sigmoid", cv=5)
calibrated_model.fit(scaled_train_data, y_train)

prediction = calibrated_model.predict(scaled_test_data)
probabilities = calibrated_model.predict_proba(scaled_test_data)

print("Calibrated model classes order:", calibrated_model.classes_)
print("Accuracy (calibrated):", accuracy_score(y_test, prediction))
print(classification_report(y_test, prediction))

# re-check calibration curves for all three classes
y_test_home_binary = (y_test == "HOME_TEAM")
y_test_away_binary = (y_test == "AWAY_TEAM")
y_test_draw_binary = (y_test == "DRAW")

home_probs = probabilities[:, 2]
away_probs = probabilities[:, 0]
draw_probs = probabilities[:, 1]

print("\nHOME calibration (actual, predicted):")
print(calibration_curve(y_test_home_binary, home_probs))

print("\nAWAY calibration (actual, predicted):")
print(calibration_curve(y_test_away_binary, away_probs))

print("\nDRAW calibration (actual, predicted):")
print(calibration_curve(y_test_draw_binary, draw_probs))