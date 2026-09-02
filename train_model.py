import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error

# Load dataset
df = pd.read_csv("land_cases.csv")

# Features used by the model
features = [
    "land_area_acres",
    "affected_families",
    "disputes",
    "documents_pending",
    "compensation_delay_months",
    "court_cases"
]

X = df[features]

# =====================================================
# MODEL 1: WILL THE CASE BE DELAYED?
# =====================================================

y_classification = df["will_be_delayed"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_classification,
    test_size=0.2,
    random_state=42
)

classifier = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\n===================================")
print("LAND ACQUISITION DELAY PREDICTOR")
print("===================================")

print("\nClassification Model")
print("--------------------")
print("Accuracy:", round(accuracy * 100, 2), "%")


# =====================================================
# MODEL 2: HOW MANY DAYS DELAY?
# =====================================================

# Only use cases that were actually delayed
delayed_df = df[df["will_be_delayed"] == 1]

X_delay = delayed_df[features]
y_delay = delayed_df["delay_days"]

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X_delay,
    y_delay,
    test_size=0.2,
    random_state=42
)

regressor = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

regressor.fit(X_train2, y_train2)

delay_predictions = regressor.predict(X_test2)

mae = mean_absolute_error(y_test2, delay_predictions)

print("\nRegression Model")
print("----------------")
print("Average prediction error:", round(mae, 2), "days")

print("\nModels trained successfully!")