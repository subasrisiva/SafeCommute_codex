"""SafeCommute realtime backend.

This service keeps short-lived trip state in memory, receives browser GPS
updates, detects route deviation, and prepares SMS-only women-safety alerts.
Configure Twilio through environment variables to send SMS; credentials are
never stored in the source code.
"""

from __future__ import annotations

import heapq
import math
import os
import re
import secrets
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request
from flask_cors import CORS


BASE = Path(__file__).resolve().parent
TRIP_TTL_SECONDS = 24 * 60 * 60
DEVIATION_ALERT_KM = 1.0
LOW_SAFETY_ALERT = 45.0
VALID_TIMES = {"Morning", "Afternoon", "Evening", "Night", "Any time"}
TIME_ALIASES = {
    "morning": "Morning",
    "afternoon": "Afternoon",
    "evening": "Evening",
    "night": "Night",
    "any time": "Any time",
    "anytime": "Any time",
    "all day": "Any time",
}

app = Flask(__name__)
CORS(app)

TRIPS: dict[str, dict[str, Any]] = {}
TRIPS_LOCK = threading.Lock()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def normalize_time(value: Any) -> str:
    key = " ".join(str(value or "").strip().lower().split())
    try:
        return TIME_ALIASES[key]
    except KeyError as error:
        raise ValueError(f"time must be one of: {', '.join(sorted(VALID_TIMES))}") from error


def normalize_location(value: Any, nodes: dict[str, Any]) -> str:
    raw = " ".join(str(value or "").strip().lower().split())
    aliases = {
        "pondicherry": "Pondicherry",
        "pondicherry city": "Pondicherry",
        "puducherry": "Pondicherry",
        "puducherry city": "Pondicherry",
        "gingee town": "Gingee",
        "senji": "Gingee",
        "senji town": "Gingee",
    }
    if raw in aliases:
        return aliases[raw]
    for name in nodes:
        if " ".join(name.lower().split()) == raw:
            return name
    return str(value or "").strip()


def load_network(time_of_day: str) -> dict[str, Any]:
    time_of_day = normalize_time(time_of_day)
    if time_of_day == "Any time":
        periods = [load_network(period) for period in ("Morning", "Afternoon", "Evening", "Night")]
        edge_map: dict[tuple[str, str], dict[str, Any]] = {}
        for network in periods:
            for raw_edge in network.get("edges", []):
                key = tuple(sorted((raw_edge["from"], raw_edge["to"])))
                current = edge_map.get(key)
                if current is None or float(raw_edge["risk"]) > float(current["risk"]):
                    edge_map[key] = {**raw_edge}
        return {
            "time_of_day": "Any time",
            "nodes": periods[0]["nodes"],
            "edges": list(edge_map.values()),
        }
    path = BASE / f"tn_network_{time_of_day.lower()}.json"
    if not path.exists():
        raise FileNotFoundError(f"Network file not found: {path.name}. Run run_all.py first.")
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def women_edge_profile(edge: dict[str, Any], time_of_day: str) -> dict[str, Any]:
    women_risk = float(edge["risk"])
    reasons: list[str] = []
    if time_of_day in {"Night", "Any time"}:
        women_risk *= 1.5
        reasons.append("night travel")
    if edge.get("lighting") == "Poor":
        women_risk += 15
        reasons.append("poor lighting")
    if edge.get("road_type") == "Isolated Lane":
        women_risk += 20
        reasons.append("isolated road")
    women_risk = min(100.0, women_risk)
    return {
        "women_risk": round(women_risk, 1),
        "women_safety": round(max(0.0, 100.0 - women_risk), 1),
        "women_reason": ", ".join(reasons),
    }


def build_graph(network: dict[str, Any], time_of_day: str) -> dict[str, list[dict[str, Any]]]:
    graph: dict[str, list[dict[str, Any]]] = {
        name: [] for name in network.get("nodes", {})
    }
    for raw in network.get("edges", []):
        profile = women_edge_profile(raw, time_of_day)
        edge = {**raw, **profile}
        edge["from"] = raw["from"]
        edge["to"] = raw["to"]
        reverse = {**edge, "from": raw["to"], "to": raw["from"]}
        graph[raw["from"]].append(edge)
        graph[raw["to"]].append(reverse)
    return graph


def dijkstra(graph: dict[str, list[dict[str, Any]]], start: str, end: str, weight: str) -> dict[str, Any] | None:
    if start not in graph or end not in graph:
        return None
    distances = {node: math.inf for node in graph}
    previous: dict[str, tuple[str, dict[str, Any]]] = {}
    distances[start] = 0.0
    queue: list[tuple[float, str]] = [(0.0, start)]

    while queue:
        current_distance, current = heapq.heappop(queue)
        if current_distance != distances[current]:
            continue
        if current == end:
            break
        for edge in graph[current]:
            candidate = current_distance + float(edge[weight])
            if candidate < distances[edge["to"]]:
                distances[edge["to"]] = candidate
                previous[edge["to"]] = (current, edge)
                heapq.heappush(queue, (candidate, edge["to"]))

    if math.isinf(distances[end]):
        return None
    path = [end]
    edges: list[dict[str, Any]] = []
    current = end
    while current != start:
        parent, edge = previous[current]
        edges.insert(0, edge)
        path.insert(0, parent)
        current = parent
    return {"path": path, "edges": edges, "total": distances[end]}


def summarize(route: dict[str, Any], women_mode: bool) -> dict[str, Any]:
    safety_key = "women_safety" if women_mode else "safety"
    distance = sum(float(edge["distance"]) for edge in route["edges"])
    average_safety = sum(float(edge[safety_key]) for edge in route["edges"]) / len(route["edges"])
    weakest = min(route["edges"], key=lambda edge: float(edge[safety_key]))
    reasons = list(dict.fromkeys(edge["women_reason"] for edge in route["edges"] if edge.get("women_reason")))
    return {
        "path": route["path"],
        "edges": route["edges"],
        "distance_km": round(distance, 1),
        "average_safety": round(average_safety, 1),
        "risk_rate": round(100.0 - average_safety, 1),
        "weakest_leg": {
            "from": weakest["from"],
            "to": weakest["to"],
            "safety": round(float(weakest[safety_key]), 1),
            "road_type": weakest.get("road_type"),
        },
        "reasons": reasons,
    }


def compute_route(start: str, end: str, time_of_day: str, women_mode: bool) -> dict[str, Any]:
    time_of_day = normalize_time(time_of_day)
    network = load_network(time_of_day)
    start = normalize_location(start, network["nodes"])
    end = normalize_location(end, network["nodes"])
    if start == end:
        raise ValueError("start and end must be different cities")
    graph = build_graph(network, time_of_day)
    safest = dijkstra(graph, start, end, "women_risk" if women_mode else "risk")
    shortest = dijkstra(graph, start, end, "distance")
    if not safest:
        raise ValueError("No route found between the requested cities")
    return {
        "start": start,
        "end": end,
        "time_of_day": time_of_day,
        "women_mode": women_mode,
        "safest": summarize(safest, women_mode),
        "shortest": summarize(shortest, women_mode) if shortest else None,
    }


def point_to_segment_km(lat: float, lon: float, a: dict[str, float], b: dict[str, float]) -> float:
    """Approximate distance from a GPS point to a lat/lon segment."""
    mean_lat = math.radians((a["lat"] + b["lat"]) / 2)
    scale_x = 111.320 * math.cos(mean_lat)
    scale_y = 110.574
    px, py = lon * scale_x, lat * scale_y
    ax, ay = a["lon"] * scale_x, a["lat"] * scale_y
    bx, by = b["lon"] * scale_x, b["lat"] * scale_y
    dx, dy = bx - ax, by - ay
    denominator = dx * dx + dy * dy
    fraction = 0.0 if denominator == 0 else ((px - ax) * dx + (py - ay) * dy) / denominator
    fraction = max(0.0, min(1.0, fraction))
    closest_x, closest_y = ax + fraction * dx, ay + fraction * dy
    return math.hypot(px - closest_x, py - closest_y)


def nearest_route_edge(trip: dict[str, Any], lat: float, lon: float) -> tuple[float, dict[str, Any] | None]:
    nodes = trip["network_nodes"]
    best_distance = math.inf
    best_edge = None
    for edge in trip["route"]["edges"]:
        a = nodes[edge["from"]]
        b = nodes[edge["to"]]
        distance = point_to_segment_km(lat, lon, a, b)
        if distance < best_distance:
            best_distance = distance
            best_edge = edge
    return best_distance, best_edge


def cleanup_old_trips() -> None:
    cutoff = time.time() - TRIP_TTL_SECONDS
    with TRIPS_LOCK:
        expired = [trip_id for trip_id, trip in TRIPS.items() if trip["created_epoch"] < cutoff]
        for trip_id in expired:
            del TRIPS[trip_id]


def public_trip(trip: dict[str, Any]) -> dict[str, Any]:
    return {
        "trip_id": trip["trip_id"],
        "user_name": trip["user_name"],
        "status": trip["status"],
        "created_at": trip["created_at"],
        "last_update": trip["last_update"],
        "women_mode": trip["women_mode"],
        "safety_contact_configured": bool(trip.get("safety_contact")),
        "route": trip["route"],
        "position": trip["position"],
        "alerts": trip["alerts"][-20:],
    }


@app.get("/api/health")
def health():
    return jsonify({"ok": True, "service": "SafeCommute realtime backend", "time": now_iso()})


@app.get("/api/route")
def route_api():
    try:
        start = request.args.get("start", "").strip()
        end = request.args.get("end", "").strip()
        time_of_day = normalize_time(request.args.get("time", "Evening"))
        women_mode = as_bool(request.args.get("women_mode", "false"))
        return jsonify(compute_route(start, end, time_of_day, women_mode))
    except (ValueError, FileNotFoundError) as error:
        return jsonify({"error": str(error)}), 400


@app.post("/api/trips")
def create_trip():
    cleanup_old_trips()
    body = request.get_json(silent=True) or {}
    try:
        start = str(body.get("start", "")).strip()
        end = str(body.get("end", "")).strip()
        time_of_day = normalize_time(body.get("time_of_day", "Evening"))
        women_mode = as_bool(body.get("women_mode", False))
        safety_contact = normalize_phone(body.get("safety_contact", body.get("contact", "")))
        route_data = compute_route(start, end, time_of_day, women_mode)
        network = load_network(time_of_day)
    except (TypeError, ValueError, FileNotFoundError) as error:
        return jsonify({"error": str(error)}), 400

    trip_id = secrets.token_urlsafe(10)
    trip = {
        "trip_id": trip_id,
        "user_name": str(body.get("user_name", "SafeCommute user")).strip() or "SafeCommute user",
        "status": "active",
        "created_at": now_iso(),
        "created_epoch": time.time(),
        "last_update": None,
        "route": route_data["safest"],
        "women_mode": women_mode,
        "safety_contact": safety_contact,
        "network_nodes": network["nodes"],
        "position": None,
        "alerts": [],
        "last_sms_alert_epoch": 0.0,
        "last_sms_alert_key": None,
    }
    with TRIPS_LOCK:
        TRIPS[trip_id] = trip
    return jsonify(public_trip(trip)), 201


@app.get("/api/trips/<trip_id>")
def get_trip(trip_id: str):
    cleanup_old_trips()
    with TRIPS_LOCK:
        trip = TRIPS.get(trip_id)
        if not trip:
            return jsonify({"error": "trip not found or expired"}), 404
        return jsonify(public_trip(trip))


@app.post("/api/trips/<trip_id>/position")
def update_position(trip_id: str):
    body = request.get_json(silent=True) or {}
    try:
        lat = float(body["lat"])
        lon = float(body["lon"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "lat and lon are required numeric values"}), 400
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return jsonify({"error": "coordinates are outside valid bounds"}), 400

    sms_alert: dict[str, str] | None = None
    response_data: dict[str, Any]
    with TRIPS_LOCK:
        trip = TRIPS.get(trip_id)
        if not trip:
            return jsonify({"error": "trip not found or expired"}), 404
        deviation_km, edge = nearest_route_edge(trip, lat, lon)
        safety = None
        if edge:
            safety = edge.get("women_safety") if trip["women_mode"] else edge.get("safety")
        alerts = []
        if deviation_km > DEVIATION_ALERT_KM:
            alerts.append({
                "type": "route_deviation",
                "message": f"You are approximately {deviation_km:.2f} km from the planned route.",
                "at": now_iso(),
            })
        if safety is not None and float(safety) < LOW_SAFETY_ALERT:
            alerts.append({
                "type": "low_safety",
                "message": f"Low-safety segment near {edge['from']} → {edge['to']} ({float(safety):.1f}/100).",
                "at": now_iso(),
            })
        if alerts and trip.get("women_mode") and trip.get("safety_contact"):
            alert_key = "|".join(sorted(
                f"low_safety:{edge['from']}:{edge['to']}" if alert["type"] == "low_safety" and edge else alert["type"]
                for alert in alerts
            ))
            now_epoch = time.time()
            if (
                alert_key != trip.get("last_sms_alert_key")
                or now_epoch - float(trip.get("last_sms_alert_epoch", 0)) >= 15 * 60
            ):
                maps_url = f"https://www.google.com/maps?q={lat:.6f},{lon:.6f}"
                details = " ".join(alert["message"] for alert in alerts)
                sms_alert = {
                    "contact": trip["safety_contact"],
                    "message": f"SafeCommute safety alert for {trip['user_name']}: {details} Location: {maps_url}",
                }
                trip["last_sms_alert_epoch"] = now_epoch
                trip["last_sms_alert_key"] = alert_key
        timestamp = str(body.get("timestamp", now_iso()))
        trip["position"] = {"lat": lat, "lon": lon, "timestamp": timestamp}
        trip["last_update"] = timestamp
        trip["alerts"].extend(alerts)
        trip["alerts"] = trip["alerts"][-50:]
        response_data = {
            "trip_id": trip_id,
            "position": trip["position"],
            "deviation_km": round(deviation_km, 3),
            "segment_safety": round(float(safety), 1) if safety is not None else None,
            "alerts": alerts,
            "status": trip["status"],
        }
    if sms_alert:
        delivery = deliver_sms_message(sms_alert["message"], [sms_alert["contact"]])
        response_data["sms_alert"] = {
            "status": delivery["status"],
            "message": delivery.get("message", sms_alert["message"]),
            "sent_to": delivery.get("sent_to", []),
            "delivery_note": delivery.get("delivery_note"),
        }
    return jsonify(response_data)


def normalize_contacts(raw: Any) -> list[str]:
    if isinstance(raw, str):
        raw = raw.split(",")
    if not isinstance(raw, list):
        return []
    contacts = []
    for value in raw:
        if isinstance(value, dict):
            value = value.get("phone")
        value = str(value or "").strip()
        if value and value not in contacts:
            contacts.append(value)
    return contacts[:3]


def normalize_phone(raw: Any) -> str:
    """Validate one trusted contact number without storing or dialing it."""
    value = str(raw or "").strip()
    if not value:
        return ""
    if not re.fullmatch(r"\+?[0-9()\s.-]{7,24}", value):
        raise ValueError("trusted contact must be one valid mobile number")
    digits = re.sub(r"\D", "", value)
    if not 7 <= len(digits) <= 15:
        raise ValueError("trusted contact must contain 7 to 15 digits")
    return value


def deliver_sms_message(message: str, contacts: list[str]) -> dict[str, Any]:
    """Send SMS only when Twilio is configured; never initiate a phone call."""
    result: dict[str, Any] = {
        "status": "prepared",
        "message": message,
        "contacts": contacts,
        "sent_to": [],
        "sms_only": True,
    }
    sid = os.getenv("TWILIO_ACCOUNT_SID")
    token = os.getenv("TWILIO_AUTH_TOKEN")
    from_number = os.getenv("TWILIO_FROM")
    if sid and token and from_number and contacts:
        try:
            from twilio.rest import Client

            client = Client(sid, token)
            for phone in contacts:
                client.messages.create(body=message, from_=from_number, to=phone)
            result["status"] = "sent"
            result["sent_to"] = contacts
        except Exception as error:  # keep the alert usable if SMS setup is incomplete
            result["delivery_error"] = f"SMS delivery unavailable: {type(error).__name__}"
    elif not contacts:
        result["delivery_note"] = "Add one trusted contact mobile number to enable SMS delivery."
    else:
        result["delivery_note"] = "Twilio is not configured; use the prepared message or open the phone SMS app."
    return result


@app.post("/api/risk-alert")
def risk_alert_api():
    """Prepare or send one SMS-only alert for a risky women-safety event."""
    body = request.get_json(silent=True) or {}
    try:
        contact = normalize_phone(body.get("safety_contact", body.get("contact", "")))
    except (TypeError, ValueError) as error:
        return jsonify({"error": str(error)}), 400
    if not contact:
        return jsonify({"error": "trusted contact mobile number is required"}), 400

    name = str(body.get("user_name", "SafeCommute user")).strip() or "SafeCommute user"
    message = str(body.get("message", "")).strip()
    if not message:
        route = str(body.get("route", "the planned route")).strip() or "the planned route"
        risk = body.get("risk")
        risk_text = f" Estimated risk: {float(risk):.1f}%." if risk is not None else ""
        message = f"SafeCommute safety alert for {name}: risk detected on {route}.{risk_text}"
    result = deliver_sms_message(message[:640], [contact])
    result["alert_type"] = str(body.get("alert_type", "risk"))
    return jsonify(result)


@app.post("/api/sos")
def sos_api():
    body = request.get_json(silent=True) or {}
    try:
        lat = float(body["lat"])
        lon = float(body["lon"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "lat and lon are required numeric values"}), 400
    name = str(body.get("user_name", "SafeCommute user")).strip() or "SafeCommute user"
    contacts = normalize_contacts(body.get("contacts", []))
    maps_url = f"https://www.google.com/maps?q={lat:.6f},{lon:.6f}"
    message = f"EMERGENCY: {name} needs help. Live location: {maps_url}"
    result = deliver_sms_message(message, contacts)
    result["location"] = {"lat": lat, "lon": lon}

    trip_id = str(body.get("trip_id", "")).strip()
    if trip_id:
        with TRIPS_LOCK:
            trip = TRIPS.get(trip_id)
            if trip:
                trip["alerts"].append({"type": "sos", "message": message, "at": now_iso()})
                trip["status"] = "sos_triggered"
    return jsonify(result)


if __name__ == "__main__":
    app.run(
        host=os.getenv("SAFECOMMUTE_HOST", "127.0.0.1"),
        port=int(os.getenv("SAFECOMMUTE_PORT", "5000")),
        debug=False,
    )
