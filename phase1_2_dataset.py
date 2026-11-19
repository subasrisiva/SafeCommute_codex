import os
BASE = os.path.dirname(os.path.abspath(__file__))
# ============================================================
# SafeCommute: AI-Driven Route Safety Analyzer
# PHASE 1: Problem Definition
# PHASE 2: Dataset Creation (Synthetic)
# ============================================================
"""
PROBLEM STATEMENT:
Commuters (students, employees, women traveling late) often don't know
which route/road segment is safe to travel on. SafeCommute analyzes
road segments using factors like past accidents, lighting, crime reports,
traffic density, and time of day to predict a SAFETY LEVEL
(Safe / Moderate / Risky) and a numeric SAFETY SCORE (0-100).

WHY SYNTHETIC DATA:
Real accident/crime datasets at road-segment level are hard to get for
a specific city (privacy + availability). So we generate a realistic
synthetic dataset with the same statistical patterns real data would have.
This is a legitimate and common approach for academic mini-projects.
You can swap this later with a real Kaggle road-accident dataset if you find one.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 2000  # number of road segments

# ---- Base location: Tiruvannamalai, Tamil Nadu area (for map realism) ----
base_lat, base_lon = 12.2253, 79.0747

df = pd.DataFrame({
    "segment_id": [f"SEG{i:04d}" for i in range(N)],
    "latitude": base_lat + np.random.uniform(-0.05, 0.05, N),
    "longitude": base_lon + np.random.uniform(-0.05, 0.05, N),
    "time_of_day": np.random.choice(
        ["Morning", "Afternoon", "Evening", "Night"], N, p=[0.3, 0.3, 0.25, 0.15]
    ),
    "road_type": np.random.choice(
        ["Main Road", "Residential", "Highway", "Isolated Lane"], N, p=[0.35, 0.35, 0.2, 0.1]
    ),
    "traffic_density": np.random.choice(["Low", "Medium", "High"], N, p=[0.3, 0.4, 0.3]),
    "street_lighting": np.random.choice(["Poor", "Moderate", "Good"], N, p=[0.25, 0.35, 0.4]),
    "past_accidents_count": np.random.poisson(1.2, N),
    "crime_reports_count": np.random.poisson(0.8, N),
    "police_patrol_nearby": np.random.choice([0, 1], N, p=[0.6, 0.4]),
    "weather": np.random.choice(["Clear", "Rainy", "Foggy"], N, p=[0.7, 0.2, 0.1]),
})

# ---- Generate a realistic RISK SCORE using a weighted formula (ground truth logic) ----
def compute_risk(row):
    risk = 0
    risk += row["past_accidents_count"] * 8
    risk += row["crime_reports_count"] * 10
    risk += {"Poor": 20, "Moderate": 8, "Good": 0}[row["street_lighting"]]
    risk += {"Night": 18, "Evening": 8, "Afternoon": 2, "Morning": 0}[row["time_of_day"]]
    risk += {"Isolated Lane": 20, "Residential": 5, "Main Road": 2, "Highway": 6}[row["road_type"]]
    risk += {"Low": 5, "Medium": 2, "High": 0}[row["traffic_density"]]  # empty roads riskier
    risk += {"Foggy": 10, "Rainy": 6, "Clear": 0}[row["weather"]]
    risk -= row["police_patrol_nearby"] * 12
    risk += np.random.normal(0, 5)  # noise
    return max(0, min(100, risk))

df["risk_score_raw"] = df.apply(compute_risk, axis=1)

# Normalize to 0-100 safety score (inverse of risk)
df["safety_score"] = (100 - df["risk_score_raw"]).round(1)

# Label: Safe / Moderate / Risky
def label_risk(score):
    if score >= 70:
        return "Safe"
    elif score >= 45:
        return "Moderate"
    else:
        return "Risky"

df["safety_label"] = df["safety_score"].apply(label_risk)

df.to_csv(os.path.join(BASE, "safecommute_dataset.csv"), index=False)
print("Dataset created:", df.shape)
print(df["safety_label"].value_counts())
print(df.head())
