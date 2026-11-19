import os
BASE = os.path.dirname(os.path.abspath(__file__))
# ============================================================
# PHASE 6: Model Building (Random Forest + Logistic Regression)
# PHASE 7: Model Evaluation
# ============================================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_absolute_error, r2_score
)
import joblib

df = pd.read_csv(os.path.join(BASE, "safecommute_features.csv"))

feature_cols = [
    "past_accidents_count", "crime_reports_count", "police_patrol_nearby",
    "street_lighting_enc", "traffic_density_enc", "time_of_day_enc"
] + [c for c in df.columns if c.startswith("road_type_") or c.startswith("weather_")]

X = df[feature_cols]
y_class = df["safety_label"]        # for classification (Safe/Moderate/Risky)
y_score = df["safety_score"]        # for regression (0-100 score)

X_train, X_test, yclass_train, yclass_test, yscore_train, yscore_test = train_test_split(
    X, y_class, y_score, test_size=0.2, random_state=42, stratify=y_class
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---- MODEL 1: Logistic Regression (classification: Safe/Moderate/Risky) ----
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, yclass_train)
lr_preds = log_reg.predict(X_test_scaled)
print("=== Logistic Regression ===")
print("Accuracy:", round(accuracy_score(yclass_test, lr_preds), 3))
print(classification_report(yclass_test, lr_preds))

# ---- MODEL 2: Random Forest Classifier (classification) ----
rf_clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
rf_clf.fit(X_train, yclass_train)
rf_preds = rf_clf.predict(X_test)
print("\n=== Random Forest Classifier ===")
print("Accuracy:", round(accuracy_score(yclass_test, rf_preds), 3))
print(classification_report(yclass_test, rf_preds))

# ---- MODEL 3: Random Forest Regressor (predicts exact 0-100 safety score) ----
rf_reg = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
rf_reg.fit(X_train, yscore_train)
reg_preds = rf_reg.predict(X_test)
print("\n=== Random Forest Regressor (Safety Score) ===")
print("MAE:", round(mean_absolute_error(yscore_test, reg_preds), 2))
print("R2 Score:", round(r2_score(yscore_test, reg_preds), 3))

# ---- Feature importance (great for report/viva) ----
importances = pd.Series(rf_clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nFeature Importances (Random Forest Classifier):\n", importances)

# ---- Save best models ----
joblib.dump(rf_clf, os.path.join(BASE, "rf_classifier.pkl"))
joblib.dump(rf_reg, os.path.join(BASE, "rf_regressor.pkl"))
joblib.dump(scaler, os.path.join(BASE, "scaler.pkl"))
print("\nModels saved: rf_classifier.pkl, rf_regressor.pkl")
