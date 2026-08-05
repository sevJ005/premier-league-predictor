from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
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

# XGBoost requires numeric labels, not text (HOME_TEAM/AWAY_TEAM/DRAW) --
# fit the encoder on training labels only, then reuse it (never re-fit)
# to transform the test labels, same principle as the scaler earlier
encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)
y_test_encoded = encoder.transform(y_test)

# tree-based model -- no scaling needed, handles nonlinear patterns
# and feature interactions natively
model_xgb = XGBClassifier(eval_metric="mlogloss", max_depth=10, n_estimators=150, learning_rate=0.05)
model_xgb.fit(X_train, y_train_encoded)

prediction_encoded = model_xgb.predict(X_test)

# convert numeric predictions back into readable labels for comparison/reporting
prediction_xgb = encoder.inverse_transform(prediction_encoded)

print("Accuracy for XGBoost:", accuracy_score(y_test, prediction_xgb))
print(classification_report(y_test, prediction_xgb))