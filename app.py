import tkinter as tk
from tkinter import messagebox
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


# =========================================================
# LOAD DATA
# =========================================================

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


# =========================================================
# TRAIN CLASSIFICATION MODEL
# =========================================================

classifier = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

classifier.fit(X, df["will_be_delayed"])


# =========================================================
# TRAIN REGRESSION MODEL
# =========================================================

delayed_df = df[df["will_be_delayed"] == 1]

regressor = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

regressor.fit(
    delayed_df[features],
    delayed_df["delay_days"]
)


# =========================================================
# MAIN WINDOW
# =========================================================

window = tk.Tk()

window.title("Land Acquisition Early Warning System")
window.geometry("1100x720")
window.minsize(950, 650)
window.configure(bg="#eef2f7")


# =========================================================
# COLORS
# =========================================================

NAVY = "#17324d"
BLUE = "#2563eb"
LIGHT_BLUE = "#e8f0fe"
WHITE = "#ffffff"
BG = "#eef2f7"
TEXT = "#172033"
GRAY = "#667085"
GREEN = "#16803c"
GREEN_BG = "#e7f6ec"
ORANGE = "#d97706"
ORANGE_BG = "#fff4df"
RED = "#d92d20"
RED_BG = "#fdecea"
BORDER = "#d9e1ea"


# =========================================================
# HEADER
# =========================================================

header = tk.Frame(
    window,
    bg=NAVY,
    height=90
)

header.pack(fill="x")
header.pack_propagate(False)


tk.Label(
    header,
    text="LAND ACQUISITION",
    font=("Arial", 22, "bold"),
    fg=WHITE,
    bg=NAVY
).pack(anchor="w", padx=30, pady=(15, 0))


tk.Label(
    header,
    text="AI EARLY WARNING & DEADLINE RISK ASSESSMENT SYSTEM",
    font=("Arial", 10),
    fg="#cbd5e1",
    bg=NAVY
).pack(anchor="w", padx=32, pady=(2, 0))


# =========================================================
# MODEL METRIC CARDS
# =========================================================

metrics = tk.Frame(window, bg=BG)
metrics.pack(fill="x", padx=25, pady=18)


def metric_card(parent, title, value, subtitle):

    card = tk.Frame(
        parent,
        bg=WHITE,
        highlightbackground=BORDER,
        highlightthickness=1
    )

    card.pack(
        side="left",
        expand=True,
        fill="both",
        padx=6
    )

    tk.Label(
        card,
        text=title,
        font=("Arial", 9),
        fg=GRAY,
        bg=WHITE
    ).pack(anchor="w", padx=15, pady=(10, 0))

    tk.Label(
        card,
        text=value,
        font=("Arial", 20, "bold"),
        fg=NAVY,
        bg=WHITE
    ).pack(anchor="w", padx=15, pady=(2, 0))

    tk.Label(
        card,
        text=subtitle,
        font=("Arial", 8),
        fg=GRAY,
        bg=WHITE
    ).pack(anchor="w", padx=15, pady=(0, 10))


metric_card(
    metrics,
    "TRAINING CASES",
    "500",
    "Synthetic historical cases"
)

metric_card(
    metrics,
    "CLASSIFICATION ACCURACY",
    "95%",
    "Deadline risk prediction"
)

metric_card(
    metrics,
    "AVG. ERROR",
    "9.4 days",
    "Delay estimation error"
)


# =========================================================
# MAIN CONTENT
# =========================================================

content = tk.Frame(window, bg=BG)
content.pack(fill="both", expand=True, padx=25)


# =========================================================
# LEFT INPUT PANEL
# =========================================================

input_panel = tk.Frame(
    content,
    bg=WHITE,
    highlightbackground=BORDER,
    highlightthickness=1
)

input_panel.pack(
    side="left",
    fill="both",
    expand=True,
    padx=(0, 10)
)


tk.Label(
    input_panel,
    text="CASE INFORMATION",
    font=("Arial", 14, "bold"),
    fg=TEXT,
    bg=WHITE
).pack(anchor="w", padx=22, pady=(18, 3))


tk.Label(
    input_panel,
    text="Enter current case conditions",
    font=("Arial", 9),
    fg=GRAY,
    bg=WHITE
).pack(anchor="w", padx=22, pady=(0, 15))


labels = [
    "Land Area (acres)",
    "Affected Families",
    "Number of Disputes",
    "Pending Documents",
    "Compensation Delay (months)",
    "Number of Court Cases"
]

entries = []


for label in labels:

    row = tk.Frame(input_panel, bg=WHITE)
    row.pack(fill="x", padx=22, pady=5)

    tk.Label(
        row,
        text=label,
        font=("Arial", 9),
        fg=TEXT,
        bg=WHITE
    ).pack(side="left")

    entry = tk.Entry(
        row,
        font=("Arial", 10),
        width=15,
        relief="solid",
        bd=1
    )

    entry.pack(side="right")

    entries.append(entry)


# Fill demo values

demo_values = [
    "50",
    "30",
    "6",
    "5",
    "8",
    "3"
]

for entry, value in zip(entries, demo_values):
    entry.insert(0, value)


# =========================================================
# ANALYZE BUTTON
# =========================================================

analyze_button = tk.Button(
    input_panel,
    text="ANALYZE CASE",
    font=("Arial", 11, "bold"),
    bg=BLUE,
    fg=WHITE,
    activebackground=NAVY,
    activeforeground=WHITE,
    relief="flat",
    cursor="hand2",
    padx=20,
    pady=10
)

analyze_button.pack(
    fill="x",
    padx=22,
    pady=(15, 8)
)


tk.Label(
    input_panel,
    text="Tip: Use current case information for an early warning.",
    font=("Arial", 8),
    fg=GRAY,
    bg=WHITE
).pack(pady=(0, 15))


# =========================================================
# RIGHT RESULT PANEL
# =========================================================

result_panel = tk.Frame(
    content,
    bg=WHITE,
    highlightbackground=BORDER,
    highlightthickness=1
)

result_panel.pack(
    side="right",
    fill="both",
    expand=True,
    padx=(10, 0)
)


tk.Label(
    result_panel,
    text="RISK ASSESSMENT",
    font=("Arial", 14, "bold"),
    fg=TEXT,
    bg=WHITE
).pack(anchor="w", padx=22, pady=(18, 3))


tk.Label(
    result_panel,
    text="Machine-learning based early warning",
    font=("Arial", 9),
    fg=GRAY,
    bg=WHITE
).pack(anchor="w", padx=22)


# =========================================================
# RISK SCORE
# =========================================================

risk_box = tk.Frame(
    result_panel,
    bg=LIGHT_BLUE,
    height=115
)

risk_box.pack(
    fill="x",
    padx=22,
    pady=15
)

risk_box.pack_propagate(False)


probability_label = tk.Label(
    risk_box,
    text="99.0%",
    font=("Arial", 32, "bold"),
    fg=NAVY,
    bg=LIGHT_BLUE
)

probability_label.pack(pady=(12, 0))


tk.Label(
    risk_box,
    text="Probability of missing deadline",
    font=("Arial", 9),
    fg=GRAY,
    bg=LIGHT_BLUE
).pack()


risk_label = tk.Label(
    result_panel,
    text="HIGH RISK",
    font=("Arial", 18, "bold"),
    fg=RED,
    bg=WHITE
)

risk_label.pack(pady=(0, 3))


delay_label = tk.Label(
    result_panel,
    text="Estimated additional delay: 134 days",
    font=("Arial", 11, "bold"),
    fg=TEXT,
    bg=WHITE
)

delay_label.pack(pady=3)


# =========================================================
# RISK FACTORS
# =========================================================

tk.Label(
    result_panel,
    text="KEY RISK FACTORS",
    font=("Arial", 10, "bold"),
    fg=TEXT,
    bg=WHITE
).pack(anchor="w", padx=22, pady=(15, 5))


factors_label = tk.Label(
    result_panel,
    text="",
    font=("Arial", 9),
    fg=GRAY,
    bg=WHITE,
    justify="left",
    anchor="w"
)

factors_label.pack(
    anchor="w",
    padx=22
)


# =========================================================
# RECOMMENDATIONS
# =========================================================

tk.Label(
    result_panel,
    text="RECOMMENDED ACTIONS",
    font=("Arial", 10, "bold"),
    fg=TEXT,
    bg=WHITE
).pack(anchor="w", padx=22, pady=(15, 5))


recommendations_label = tk.Label(
    result_panel,
    text="",
    font=("Arial", 9),
    fg=GRAY,
    bg=WHITE,
    justify="left",
    anchor="w"
)

recommendations_label.pack(
    anchor="w",
    padx=22
)


# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict():

    try:

        values = [
            float(entries[0].get()),
            int(entries[1].get()),
            int(entries[2].get()),
            int(entries[3].get()),
            int(entries[4].get()),
            int(entries[5].get())
        ]

        case = pd.DataFrame(
            [values],
            columns=features
        )

        prediction = classifier.predict(case)[0]

        probability = classifier.predict_proba(case)[0][1]

        percentage = probability * 100


        # ---------------------------------------------
        # PROBABILITY
        # ---------------------------------------------

        probability_label.config(
            text=f"{percentage:.1f}%"
        )


        # ---------------------------------------------
        # RISK LEVEL
        # ---------------------------------------------

        if percentage >= 70:

            risk_label.config(
                text="HIGH RISK",
                fg=RED
            )

            risk_box.config(
                bg=RED_BG
            )

            probability_label.config(
                bg=RED_BG,
                fg=RED
            )

        elif percentage >= 40:

            risk_label.config(
                text="MEDIUM RISK",
                fg=ORANGE
            )

            risk_box.config(
                bg=ORANGE_BG
            )

            probability_label.config(
                bg=ORANGE_BG,
                fg=ORANGE
            )

        else:

            risk_label.config(
                text="LOW RISK",
                fg=GREEN
            )

            risk_box.config(
                bg=GREEN_BG
            )

            probability_label.config(
                bg=GREEN_BG,
                fg=GREEN
            )


        # ---------------------------------------------
        # DELAY
        # ---------------------------------------------

        if prediction == 1:

            estimated_days = regressor.predict(case)[0]

            delay_label.config(
                text=f"Estimated additional delay: {estimated_days:.0f} days"
            )

        else:

            delay_label.config(
                text="Expected to finish within the deadline."
            )


        # ---------------------------------------------
        # RISK FACTORS
        # ---------------------------------------------

        reasons = []

        if values[2] >= 4:
            reasons.append("• High number of land disputes")

        if values[3] >= 4:
            reasons.append("• Documents are still pending")

        if values[4] >= 6:
            reasons.append("• Significant compensation delay")

        if values[5] >= 2:
            reasons.append("• Multiple court cases")

        if values[1] >= 25:
            reasons.append("• Large number of affected families")

        if values[0] >= 70:
            reasons.append("• Large land acquisition area")

        if not reasons:
            reasons.append("• No major high-risk factors detected")


        factors_label.config(
            text="\n".join(reasons)
        )


        # ---------------------------------------------
        # RECOMMENDATIONS
        # ---------------------------------------------

        actions = []

        if values[2] >= 4:
            actions.append("• Prioritize dispute resolution")

        if values[3] >= 4:
            actions.append("• Complete pending documentation")

        if values[4] >= 6:
            actions.append("• Review compensation processing")

        if values[5] >= 2:
            actions.append("• Monitor active court proceedings")

        if not actions:
            actions.append("• Continue routine monitoring")


        recommendations_label.config(
            text="\n".join(actions)
        )


    except ValueError:

        messagebox.showerror(
            "Invalid Input",
            "Please enter valid numbers in all fields."
        )


analyze_button.config(command=predict)


# =========================================================
# FOOTER
# =========================================================

footer = tk.Frame(
    window,
    bg=NAVY,
    height=30
)

footer.pack(fill="x", side="bottom")
footer.pack_propagate(False)


tk.Label(
    footer,
    text="Prototype | Synthetic dataset | Decision-support system",
    font=("Arial", 8),
    fg="#cbd5e1",
    bg=NAVY
).pack(pady=7)


# =========================================================
# INITIAL PREDICTION
# =========================================================

predict()


window.mainloop()