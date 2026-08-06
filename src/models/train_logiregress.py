from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.calibration import calibration_curve
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
# full feature/target set (not used for training directly, kept for reference)
X = match_features[feature_cols]
y = match_features["winner"]

# chronological split -- train on 2018-2024, test on 2025
train = match_features[match_features["season_year"].isin([2018, 2019, 2020, 2021, 2022, 2023, 2024])]
test = match_features[match_features["season_year"] == 2025]

X_train = train[feature_cols]
y_train = train["winner"]

X_test = test[feature_cols]
y_test = test["winner"]

# scale features -- logistic regression needs comparable scales to converge properly
scaler = StandardScaler()
scaled_train_data= scaler.fit_transform(X_train)
scaled_test_data = scaler.transform(X_test)

# train baseline model (LR)
model = LogisticRegression()
model.fit(scaled_train_data, y_train)

# confirmation of execution
print("LogisticRegression Model trained successfully.")

# prediction and score
prediction = model.predict(scaled_test_data)
probabilities = model.predict_proba(scaled_test_data)
accuracy = accuracy_score(y_test, prediction)

# check calibration HOME_TEAM
y_test_home_binary = (y_test == "HOME_TEAM")
HOME_TEAM = probabilities[:,2]
calibrated_home = calibration_curve(y_test_home_binary, HOME_TEAM)

# check calibration AWAY_TEAM
y_test_home_binary = (y_test == "AWAY_TEAM")
AWAY_TEAM = probabilities[:,0]
calibrated_away = calibration_curve(y_test_home_binary, AWAY_TEAM)

# check calibration DRAW
y_test_home_binary = (y_test == "DRAW")
DRAW = probabilities[:,1]
calibrated_draw = calibration_curve(y_test_home_binary, DRAW)

# gettings results
print("Accuracy for LR:", accuracy)
print(classification_report(y_test, prediction))
print("------------Shape------------")
print(probabilities.shape)
print("------------------------------")
print("0 = AWAY, 1 = DRAW, 2 = HOME")
print(probabilities[:5])
print("----------Calibrated----------")
print("Home", calibrated_home)
print("DRAW", calibrated_draw)
print("AWAY", calibrated_away)
