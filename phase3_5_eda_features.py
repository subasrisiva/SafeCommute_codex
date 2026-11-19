import os
BASE = os.path.dirname(os.path.abspath(__file__))
# ============================================================
# PHASE 3: Data Cleaning
# PHASE 4: Exploratory Data Analysis (EDA)
# PHASE 5: Feature Engineering
# ============================================================
import pandas as pd
import numpy as np

df = pd.read_csv(os.path.join(BASE, "safecommute_dataset.csv"))

# ---- PHASE 3: Cleaning ----
print("Missing values:\n", df.isnull().sum())
df = df.drop_duplicates(subset="segment_id")
print("\nShape after cleaning:", df.shape)

# ---- PHASE 4: EDA (key insights, printed as text for report/viva) ----
print("\n--- EDA Insights ---")
print("Avg safety score by road type:\n", df.groupby("road_type")["safety_score"].mean().round(1))
print("\nAvg safety score by time of day:\n", df.groupby("time_of_day")["safety_score"].mean().round(1))
print("\nAvg safety score by lighting:\n", df.groupby("street_lighting")["safety_score"].mean().round(1))
print("\nCorrelation with safety_score:\n",
      df[["past_accidents_count", "crime_reports_count", "police_patrol_nearby", "safety_score"]].corr()["safety_score"])

# ---- PHASE 5: Feature Engineering ----
# Encode categorical variables
df_model = df.copy()

ordinal_maps = {
    "street_lighting": {"Poor": 0, "Moderate": 1, "Good": 2},
    "traffic_density": {"Low": 0, "Medium": 1, "High": 2},
    "time_of_day": {"Morning": 0, "Afternoon": 1, "Evening": 2, "Night": 3},
}
for col, mapping in ordinal_maps.items():
    df_model[col + "_enc"] = df_model[col].map(mapping)

df_model = pd.get_dummies(df_model, columns=["road_type", "weather"], drop_first=True)

feature_cols = [
    "past_accidents_count", "crime_reports_count", "police_patrol_nearby",
    "street_lighting_enc", "traffic_density_enc", "time_of_day_enc"
] + [c for c in df_model.columns if c.startswith("road_type_") or c.startswith("weather_")]

df_model.to_csv(os.path.join(BASE, "safecommute_features.csv"), index=False)
print("\nFinal feature columns:", feature_cols)
print("Feature-engineered dataset saved. Shape:", df_model.shape)
