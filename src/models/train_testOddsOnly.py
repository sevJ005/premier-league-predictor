from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

match_features = pd.read_csv("data/processed/match_features.csv")

# odds-only feature set -- testing whether market-implied probabilities
# alone already capture most of the predictive signal, independent of
# our engineered rolling/form features
odds_only_cols = ["implied_prob_home", "implied_prob_draw", "implied_prob_away"]

train = match_features[match_features["season_year"].isin([2018, 2019, 2020, 2021, 2022, 2023, 2024])]
test = match_features[match_features["season_year"] == 2025]

X_train = train[odds_only_cols]
y_train = train["winner"]

X_test = test[odds_only_cols]
y_test = test["winner"]

# no scaling needed -- odds are already on a comparable, bounded 0-1 scale
model = LogisticRegression()
model.fit(X_train, y_train)

prediction = model.predict(X_test)

print("Odds-only accuracy:", accuracy_score(y_test, prediction))
print(classification_report(y_test, prediction))