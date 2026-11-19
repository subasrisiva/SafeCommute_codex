import os
BASE = os.path.dirname(os.path.abspath(__file__))
import networkx as nx
import pandas as pd
import numpy as np
import joblib
import json

rf_reg = joblib.load(os.path.join(BASE, "rf_regressor.pkl"))

ordinal_maps = {
    "street_lighting": {"Poor": 0, "Moderate": 1, "Good": 2},
    "traffic_density": {"Low": 0, "Medium": 1, "High": 2},
    "time_of_day": {"Morning": 0, "Afternoon": 1, "Evening": 2, "Night": 3},
}
feature_cols = [
    "past_accidents_count", "crime_reports_count", "police_patrol_nearby",
    "street_lighting_enc", "traffic_density_enc", "time_of_day_enc",
    "road_type_Isolated Lane", "road_type_Main Road", "road_type_Residential",
    "weather_Foggy", "weather_Rainy"
]

def predict_edge_safety(accidents, crime, patrol, lighting, traffic, road_type, weather, time_of_day):
    row = {c: 0 for c in feature_cols}
    row["past_accidents_count"] = accidents
    row["crime_reports_count"] = crime
    row["police_patrol_nearby"] = patrol
    row["street_lighting_enc"] = ordinal_maps["street_lighting"][lighting]
    row["traffic_density_enc"] = ordinal_maps["traffic_density"][traffic]
    row["time_of_day_enc"] = ordinal_maps["time_of_day"][time_of_day]
    if road_type == "Isolated Lane":
        row["road_type_Isolated Lane"] = 1
    elif road_type == "Main Road":
        row["road_type_Main Road"] = 1
    elif road_type == "Residential":
        row["road_type_Residential"] = 1
    if weather == "Foggy":
        row["weather_Foggy"] = 1
    elif weather == "Rainy":
        row["weather_Rainy"] = 1
    X = pd.DataFrame([row])[feature_cols]
    return round(float(rf_reg.predict(X)[0]), 1)

# ---- Major Tamil Nadu cities/towns with real coordinates ----
LOCATIONS = {
    # Tiruvannamalai district cluster (kept from Phase 12)
    "Tiruvannamalai": (12.2253, 79.0747),
    "Polur": (12.5117, 79.1261),
    "Chengam": (12.3000, 78.8000),
    "Arni": (12.6706, 79.2845),
    "Chetpet": (12.4644, 79.3475),
    "Vandavasi": (12.5000, 79.6200),
    "Thandarampattu": (12.1540, 78.9470),
    "Vettavalam": (12.1000, 79.2500),
    "Jamunamarathur": (12.5973, 78.8878),
    # Villupuram / Gingee area (small towns and commonly used connectors)
    "Gingee": (12.2520, 79.4170),
    "Tindivanam": (12.2340, 79.6550),
    "Melmalayanur": (12.2660, 79.0500),
    "Tirukkoyilur": (11.9660, 79.2020),
    "Kallakurichi": (11.7400, 78.9590),
    "Ulundurpet": (11.6980, 79.2880),
    # Additional smaller places on the existing state corridors
    "Chidambaram": (11.3990, 79.6910),
    "Mayiladuthurai": (11.1010, 79.6520),
    "Perambalur": (11.2340, 78.8800),
    "Palani": (10.4500, 77.5200),
    "Aruppukkottai": (9.5100, 78.0950),
    "Paramakudi": (9.5470, 78.5900),
    # Major Tamil Nadu cities
    "Chennai": (13.0827, 80.2707),
    "Kanchipuram": (12.8342, 79.7036),
    "Vellore": (12.9165, 79.1325),
    "Krishnagiri": (12.5186, 78.2137),
    "Hosur": (12.7409, 77.8253),
    "Salem": (11.6643, 78.1460),
    "Namakkal": (11.2189, 78.1677),
    "Erode": (11.3410, 77.7172),
    "Tiruppur": (11.1085, 77.3411),
    "Coimbatore": (11.0168, 76.9558),
    "Pollachi": (10.6600, 77.0080),
    "Udhagamandalam (Ooty)": (11.4102, 76.6950),
    "Karur": (10.9601, 78.0766),
    "Tiruchirappalli (Trichy)": (10.7905, 78.7047),
    "Thanjavur": (10.7870, 79.1378),
    "Kumbakonam": (10.9601, 79.3788),
    "Nagapattinam": (10.7672, 79.8449),
    "Pudukkottai": (10.3833, 78.8000),
    "Sivaganga": (9.8433, 78.4809),
    "Madurai": (9.9252, 78.1198),
    "Dindigul": (10.3624, 77.9695),
    "Theni": (10.0104, 77.4768),
    "Virudhunagar": (9.5680, 77.9624),
    "Sivakasi": (9.4491, 77.7972),
    "Rajapalayam": (9.4520, 77.5570),
    "Ramanathapuram": (9.3639, 78.8395),
    "Tirunelveli": (8.7139, 77.7567),
    "Thoothukudi": (8.7642, 78.1348),
    "Nagercoil": (8.1790, 77.4338),
    "Villupuram": (11.9401, 79.4861),
    "Cuddalore": (11.7480, 79.7714),
    "Pondicherry": (11.9139, 79.8145),
}

# ---- Roads: (from, to, accidents, crime, patrol, lighting, traffic, road_type, weather) ----
ROADS = [
    # Tiruvannamalai district cluster (from Phase 12)
    ("Tiruvannamalai", "Polur", 2, 1, 1, "Moderate", "Medium", "Main Road", "Clear"),
    ("Tiruvannamalai", "Chengam", 2, 1, 1, "Moderate", "Medium", "Main Road", "Clear"),
    ("Tiruvannamalai", "Vettavalam", 1, 0, 1, "Good", "High", "Main Road", "Clear"),
    ("Tiruvannamalai", "Thandarampattu", 1, 0, 1, "Moderate", "Medium", "Main Road", "Clear"),
    ("Tiruvannamalai", "Arni", 3, 1, 1, "Moderate", "High", "Main Road", "Clear"),
    ("Polur", "Chengam", 3, 1, 0, "Poor", "Low", "Isolated Lane", "Clear"),
    ("Polur", "Jamunamarathur", 4, 1, 0, "Poor", "Low", "Isolated Lane", "Foggy"),
    ("Polur", "Chetpet", 2, 1, 1, "Moderate", "Medium", "Main Road", "Clear"),
    ("Arni", "Chetpet", 2, 1, 1, "Moderate", "Medium", "Main Road", "Clear"),
    ("Arni", "Vandavasi", 1, 0, 1, "Good", "High", "Highway", "Clear"),
    ("Thandarampattu", "Vettavalam", 2, 1, 0, "Moderate", "Low", "Residential", "Clear"),
    ("Chengam", "Jamunamarathur", 3, 1, 0, "Poor", "Low", "Isolated Lane", "Clear"),
    # Connect Tiruvannamalai cluster to the state network
    ("Tiruvannamalai", "Villupuram", 2, 1, 1, "Moderate", "Medium", "Main Road", "Clear"),
    ("Gingee", "Villupuram", 2, 1, 1, "Moderate", "Medium", "Main Road", "Clear"),
    ("Gingee", "Tindivanam", 2, 1, 1, "Good", "High", "Highway", "Clear"),
    ("Gingee", "Melmalayanur", 2, 1, 0, "Moderate", "Low", "Main Road", "Clear"),
    ("Gingee", "Tirukkoyilur", 2, 1, 0, "Moderate", "Medium", "Main Road", "Clear"),
    ("Tirukkoyilur", "Kallakurichi", 2, 1, 1, "Moderate", "Medium", "Main Road", "Clear"),
    ("Kallakurichi", "Ulundurpet", 2, 1, 1, "Good", "High", "Highway", "Clear"),
    ("Ulundurpet", "Villupuram", 2, 1, 1, "Good", "High", "Highway", "Clear"),
    ("Tindivanam", "Pondicherry", 2, 1, 1, "Good", "High", "Highway", "Clear"),
    ("Vandavasi", "Kanchipuram", 2, 1, 1, "Moderate", "Medium", "Main Road", "Clear"),
    ("Arni", "Vellore", 2, 1, 1, "Moderate", "Medium", "Main Road", "Clear"),
    # NH44 corridor: Chennai - Vellore - Krishnagiri - Hosur (towards Bangalore)
    ("Chennai", "Kanchipuram", 3, 2, 1, "Good", "High", "Highway", "Clear"),
    ("Kanchipuram", "Vellore", 2, 1, 1, "Good", "High", "Highway", "Clear"),
    ("Vellore", "Krishnagiri", 2, 1, 1, "Good", "High", "Highway", "Clear"),
    ("Krishnagiri", "Hosur", 1, 0, 1, "Good", "High", "Highway", "Clear"),
    ("Krishnagiri", "Salem", 2, 1, 1, "Good", "High", "Highway", "Clear"),
    # Salem - Namakkal - Erode - Tiruppur - Coimbatore corridor
    ("Salem", "Namakkal", 2, 1, 1, "Good", "Medium", "Highway", "Clear"),
    ("Namakkal", "Karur", 2, 1, 0, "Moderate", "Medium", "Main Road", "Clear"),
    ("Salem", "Erode", 2, 1, 1, "Good", "High", "Highway", "Clear"),
    ("Erode", "Tiruppur", 1, 0, 1, "Good", "High", "Highway", "Clear"),
    ("Tiruppur", "Coimbatore", 2, 1, 1, "Good", "High", "Highway", "Clear"),
    ("Coimbatore", "Pollachi", 2, 1, 1, "Moderate", "Medium", "Main Road", "Clear"),
    ("Coimbatore", "Udhagamandalam (Ooty)", 5, 1, 0, "Poor", "Low", "Isolated Lane", "Foggy"),  # hill ghat road - genuinely riskiest
    # Karur - Trichy - Thanjavur - Kumbakonam - Nagapattinam
    ("Karur", "Tiruchirappalli (Trichy)", 2, 1, 1, "Good", "Medium", "Main Road", "Clear"),
    ("Tiruchirappalli (Trichy)", "Thanjavur", 1, 0, 1, "Good", "High", "Highway", "Clear"),
    ("Thanjavur", "Kumbakonam", 1, 0, 1, "Good", "Medium", "Main Road", "Clear"),
    ("Kumbakonam", "Nagapattinam", 2, 1, 0, "Moderate", "Low", "Main Road", "Clear"),
    ("Cuddalore", "Nagapattinam", 3, 1, 0, "Moderate", "Medium", "Main Road", "Rainy"),
    ("Cuddalore", "Chidambaram", 2, 1, 1, "Good", "Medium", "Main Road", "Clear"),
    ("Chidambaram", "Mayiladuthurai", 2, 1, 1, "Good", "Medium", "Main Road", "Clear"),
    ("Mayiladuthurai", "Nagapattinam", 2, 1, 1, "Good", "Medium", "Main Road", "Clear"),
    # Trichy - Pudukkottai - Sivaganga - Madurai
    ("Tiruchirappalli (Trichy)", "Pudukkottai", 2, 1, 0, "Moderate", "Medium", "Main Road", "Clear"),
    ("Pudukkottai", "Sivaganga", 2, 1, 0, "Moderate", "Low", "Main Road", "Clear"),
    ("Sivaganga", "Madurai", 2, 1, 1, "Moderate", "Medium", "Main Road", "Clear"),
    ("Tiruchirappalli (Trichy)", "Dindigul", 2, 1, 1, "Good", "High", "Highway", "Clear"),
    ("Tiruchirappalli (Trichy)", "Perambalur", 2, 1, 1, "Good", "High", "Highway", "Clear"),
    ("Perambalur", "Kallakurichi", 2, 1, 1, "Moderate", "Medium", "Main Road", "Clear"),
    ("Dindigul", "Madurai", 1, 0, 1, "Good", "High", "Highway", "Clear"),
    ("Dindigul", "Palani", 2, 1, 1, "Good", "Medium", "Main Road", "Clear"),
    ("Palani", "Coimbatore", 2, 1, 1, "Good", "Medium", "Main Road", "Clear"),
    ("Dindigul", "Karur", 2, 1, 0, "Moderate", "Medium", "Main Road", "Clear"),
    # Madurai region
    ("Madurai", "Theni", 2, 1, 1, "Moderate", "Medium", "Main Road", "Clear"),
    ("Madurai", "Virudhunagar", 1, 0, 1, "Good", "High", "Highway", "Clear"),
    ("Madurai", "Aruppukkottai", 2, 1, 1, "Good", "Medium", "Main Road", "Clear"),
    ("Aruppukkottai", "Paramakudi", 2, 1, 0, "Moderate", "Medium", "Main Road", "Clear"),
    ("Paramakudi", "Ramanathapuram", 2, 1, 1, "Good", "Medium", "Main Road", "Clear"),
    ("Virudhunagar", "Rajapalayam", 2, 1, 0, "Moderate", "Medium", "Main Road", "Clear"),
    ("Virudhunagar", "Sivakasi", 1, 0, 1, "Good", "Medium", "Main Road", "Clear"),
    ("Virudhunagar", "Ramanathapuram", 3, 1, 0, "Poor", "Low", "Main Road", "Clear"),
    ("Madurai", "Tirunelveli", 2, 1, 1, "Good", "High", "Highway", "Clear"),
    ("Tirunelveli", "Thoothukudi", 2, 1, 1, "Good", "Medium", "Main Road", "Clear"),
    ("Tirunelveli", "Nagercoil", 2, 1, 1, "Good", "Medium", "Main Road", "Clear"),
    # Villupuram - Cuddalore, Villupuram - Trichy
    ("Chennai", "Villupuram", 3, 2, 1, "Good", "High", "Highway", "Clear"),
    ("Villupuram", "Cuddalore", 2, 1, 1, "Good", "Medium", "Main Road", "Clear"),
    ("Villupuram", "Tiruchirappalli (Trichy)", 2, 1, 1, "Good", "High", "Highway", "Clear"),
    # Pondicherry connections: Chennai coastal corridor, Villupuram, and Cuddalore
    ("Chennai", "Pondicherry", 3, 2, 1, "Good", "High", "Highway", "Clear"),
    ("Villupuram", "Pondicherry", 2, 1, 1, "Good", "Medium", "Main Road", "Clear"),
    ("Cuddalore", "Pondicherry", 2, 1, 1, "Good", "Medium", "Main Road", "Clear"),
]

def haversine_km(p1, p2):
    from math import radians, sin, cos, sqrt, atan2
    lat1, lon1 = p1; lat2, lon2 = p2
    R = 6371
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))

def export_network(time_of_day="Evening", out_path="tn_network.json"):
    nodes = {name: {"lat": lat, "lon": lon} for name, (lat, lon) in LOCATIONS.items()}
    edges = []
    for u, v, acc, crime, patrol, light, traffic, rtype, weather in ROADS:
        safety = predict_edge_safety(acc, crime, patrol, light, traffic, rtype, weather, time_of_day)
        dist = round(haversine_km(LOCATIONS[u], LOCATIONS[v]) * 1.25, 1)
        edges.append({"from": u, "to": v, "safety": safety, "risk": round(100-safety,1),
                      "distance": dist, "road_type": rtype, "lighting": light})
    data = {"time_of_day": time_of_day, "nodes": nodes, "edges": edges}
    with open(out_path, "w") as f:
        json.dump(data, f, indent=1)
    return data

if __name__ == "__main__":
    for tod in ["Morning", "Afternoon", "Evening", "Night"]:
        data = export_network(tod, f"tn_network_{tod.lower()}.json")
        print(tod, "-> nodes:", len(data["nodes"]), "edges:", len(data["edges"]))
