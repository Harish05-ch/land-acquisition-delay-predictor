import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Load dataset
df = pd.read_csv("land_cases.csv")

features = [
    "land_area_acres",
    "affected_families",
    "disputes",
    "documents_pending",
    "compensation_delay_months",
    "court_cases"
]

X = df[features]

# ============================
# TRAIN CLASSIFICATION MODEL
# ============================

classifier = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

classifier.fit(X, df["will_be_delayed"])


# ============================
# TRAIN REGRESSION MODEL
# ============================

delayed_df = df[df["will_be_delayed"] == 1]

regressor = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

regressor.fit(
    delayed_df[features],
    delayed_df["delay_days"]
)


# ============================
# GET USER INPUT
# ============================

print("\n========================================")
print(" LAND ACQUISITION DELAY PREDICTOR")
print("========================================")

print("\nEnter details of the land acquisition case:\n")

land_area = float(input("Land area (acres): "))
families = int(input("Affected families: "))
disputes = int(input("Number of disputes: "))
documents = int(input("Pending documents: "))
compensation = int(input("Compensation delay (months): "))
court_cases = int(input("Number of court cases: "))

case = pd.DataFrame([[
    land_area,
    families,
    disputes,
    documents,
    compensation,
    court_cases
]], columns=features)


# ============================
# MAKE PREDICTION
# ============================

prediction = classifier.predict(case)[0]

probability = classifier.predict_proba(case)[0][1]


print("\n========================================")
print("              RESULT")
print("========================================")

print(
    "Probability of delay:",
    round(probability * 100, 1),
    "%"
)


if prediction == 1:

    estimated_days = regressor.predict(case)[0]

    print("\n⚠️  HIGH RISK OF DEADLINE DELAY")
    print(
        "Estimated additional delay:",
        round(estimated_days),
        "days"
    )

else:

    print("\n✅ LOW RISK OF DEADLINE DELAY")
    print("Case is likely to finish within the deadline.")


print("\n========================================")