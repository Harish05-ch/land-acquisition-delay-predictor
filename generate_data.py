import pandas as pd
import random

random.seed(42)

data = []

for i in range(500):
    land_area = random.randint(1, 100)
    affected_families = random.randint(1, 50)
    disputes = random.randint(0, 10)
    documents_pending = random.randint(0, 8)
    compensation_delay = random.randint(0, 12)
    court_cases = random.randint(0, 5)

    # Calculate a realistic delay score
    score = (
        disputes * 3
        + documents_pending * 2
        + compensation_delay * 2
        + court_cases * 4
        + affected_families * 0.2
    )

    # Whether the case will miss its deadline
    delayed = 1 if score > 35 else 0

    # Approximate number of delayed days
    delay_days = 0

    if delayed:
        delay_days = int(score * 2 + random.randint(5, 30))

    data.append([
        land_area,
        affected_families,
        disputes,
        documents_pending,
        compensation_delay,
        court_cases,
        delayed,
        delay_days
    ])

columns = [
    "land_area_acres",
    "affected_families",
    "disputes",
    "documents_pending",
    "compensation_delay_months",
    "court_cases",
    "will_be_delayed",
    "delay_days"
]

df = pd.DataFrame(data, columns=columns)

df.to_csv("land_cases.csv", index=False)

print("Dataset created successfully!")
print(df.head())
print("\nTotal cases:", len(df))
print("Delayed cases:", df["will_be_delayed"].sum())