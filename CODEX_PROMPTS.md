# SafeCommute — VSCode + Codex Setup and Prompts (with Women's Safety Alerts)

## PART 1: Set up VSCode (one time, ~15 minutes)

1. Install **VSCode** (code.visualstudio.com) and **Python 3.10+**. Tick "Add Python to PATH" during install.
2. Install VSCode extensions: **Python**, **Live Server** (to run the HTML app), and your AI assistant (**OpenAI Codex / ChatGPT extension**, or **GitHub Copilot Chat** if that is what you have).
3. Create a folder `SafeCommute` and copy ALL project files into it (all `phase*.py`, `tn_network.py`, `tn_safecommute_app.html`, `tn_combined.json`, `.pkl` model files, dataset CSV).
4. Open it: `File > Open Folder > SafeCommute`.
5. Open the terminal (`Ctrl + ~`) and run:
   ```
   python -m venv venv
   venv\Scripts\activate
   pip install pandas numpy scikit-learn folium networkx joblib flask flask-cors
   ```
   (Mac/Linux: `source venv/bin/activate`)
6. Test it works: `python phase6_7_model.py` should train and save the models.
7. Right-click `tn_safecommute_app.html` > **Open with Live Server** to run the app.

> Tip: always tell Codex to **explain a change before applying it**, and commit to git
> (`git init`, `git add .`, `git commit -m "working baseline"`) before every big prompt so you can undo.

---

## PART 2: Prompts to paste into Codex, IN THIS ORDER

### Prompt 1 — Context (paste first, no changes yet)
```
I'm building "SafeCommute", an AI-driven route safety analyzer for Tamil Nadu.
Stack: Python (scikit-learn RandomForest classifier + regressor, networkx),
and a single-file HTML/JS frontend (Leaflet map, client-side Dijkstra routing).

Files:
- phase1_2_dataset.py        synthetic road-segment data
- phase3_5_eda_features.py   cleaning, EDA, feature encoding
- phase6_7_model.py          trains + saves rf_classifier.pkl, rf_regressor.pkl, scaler.pkl
- phase8_route_predictor.py  segment/route safety prediction
- phase10_route_suggestion.py, phase12_real_locations.py   safest vs shortest route
- tn_network.py              40-city Tamil Nadu graph, scores edges with the model,
                             exports tn_combined.json
- tn_safecommute_app.html    main app (city dropdowns, Dijkstra in JS, Leaflet map,
                             simulated live tracker)

Read every file and summarize how data flows between them. Do NOT edit anything yet.
```

### Prompt 2 — Fix bugs and make it run on my machine
```
Make the project run cleanly on my Windows machine in VSCode:
1. Replace every absolute path like /home/claude/... with relative paths using
   os.path / pathlib so scripts work from the project folder.
2. Add a requirements.txt.
3. Add a run_all.py that runs phase1_2 -> phase3_5 -> phase6_7 -> tn_network.py
   in order and stops with a clear message if any step fails.
4. Check tn_safecommute_app.html for JavaScript console errors and fix them.
5. List every change you made at the end.
```

### Prompt 3 — Backend API (needed for real alerts)
```
Add a Flask backend (app.py) with flask-cors:
- GET  /api/route?start=...&end=...&time=Night&mode=women  -> returns safest route,
  risk rate, per-leg scores (reuse the logic in tn_network.py and the saved models)
- POST /api/sos  -> receives {user_name, lat, lon, timestamp, contacts[]} and
  triggers the alert flow (Prompt 5)
- GET  /api/health
Keep tn_safecommute_app.html working; make it call this API with fetch() and fall
back to the existing in-browser Dijkstra if the API is unreachable.
```

---

## PART 3: Women's safety features (prompts)

### What to add (and why it's realistic for an MCA project)
| Feature | What it does |
|---|---|
| **Women-safe routing mode** | Weights night-time, poor lighting, isolated roads, and no-patrol segments more heavily, so the "safest" route for a woman at night avoids isolated lanes |
| **SOS button** | One tap sends the user's live location to emergency contacts |
| **Live location share link** | A link contacts can open to watch the trip on a map |
| **Route-deviation alert** | If the user leaves the planned route or stops too long in a risky segment, the app asks "Are you OK?" and auto-alerts if there is no reply |
| **Safe check-in timer** | User sets an expected arrival time; if they do not tap "I'm safe" in time, contacts are alerted |
| **Nearby help** | Shows nearby police stations and hospitals on the map |
| **Fake call / quick helpline buttons** | Quick access to 112 (national emergency), 1091 and 181 (women helplines) |

### Prompt 4 — Women-safe routing mode
```
In tn_network.py and tn_safecommute_app.html add a "Women Safety Mode" toggle.
When on, recompute edge risk with extra penalties: Night time x1.5 weight,
street_lighting == Poor +15 risk, road_type == Isolated Lane +20 risk,
police_patrol_nearby == 0 +10 risk, crime_reports_count x1.5. Keep the original
mode available. Show both routes on the map and a short plain-language reason,
e.g. "Avoids Chengam-Jamunamarathur: isolated, poorly lit at night".
```

### Prompt 5 — SOS button + emergency contacts
```
Add an SOS feature to tn_safecommute_app.html and app.py:
1. A large red floating SOS button on the map screen.
2. A settings panel where the user saves their name and up to 3 emergency
   contacts (name + phone). Store in localStorage.
3. On SOS tap: use navigator.geolocation to get current coordinates (fallback to
   the simulated vehicle position if permission is denied), then POST to /api/sos.
4. In app.py, /api/sos builds a message: "EMERGENCY: <name> needs help. Live
   location: https://www.google.com/maps?q=<lat>,<lon>" and sends it via Twilio
   SMS (read TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM from environment
   variables, NEVER hardcode them). If Twilio is not configured, return the
   prepared message so the frontend can open an sms: link as a fallback.
5. Show a confirmation screen and a one-tap button to call 112.
6. Add a 5-second cancel countdown after pressing SOS to prevent accidental alerts.
```

### Prompt 6 — Route-deviation and check-in timer alerts
```
Add two safety monitors to tn_safecommute_app.html:
1. Deviation alert: during live tracking, if the position is more than 1 km from
   the planned route polyline, or stationary for more than 3 minutes in a segment
   with safety score below 45, show a full-screen "Are you OK?" prompt with a
   30-second countdown. If not dismissed, trigger the same flow as the SOS button.
2. Check-in timer: user sets an expected arrival time when starting a trip. If they
   do not tap "I'm safe" within 10 minutes after that time, trigger the SOS flow.
Make all thresholds constants at the top of the file so they are easy to change.
```

### Prompt 7 — Live location share link + realtime database
```
Integrate Firebase Realtime Database (config from my Firebase console will be
pasted below). When a trip starts, create /trips/{tripId} with the route and write
the current position to /trips/{tripId}/position every 5 seconds. Generate a
shareable page track.html?trip={tripId} that listens with onValue() and shows the
live marker on a Leaflet map, with the route risk rate and status. Include these
Firebase security rules: anyone with the tripId can read that trip, only the
creating client can write, and trips auto-expire after 24 hours.
My firebaseConfig: <paste here>
```

### Prompt 8 — Nearby police stations / hospitals
```
Add a "Nearby Help" layer. Create help_points.json with real police stations and
hospitals for the main towns in my network (Tiruvannamalai, Chennai, Coimbatore,
Madurai, Trichy, Salem, etc.), using only locations I can verify; mark any
placeholder clearly with "verify". Show them as map markers, and on SOS include the
nearest one in the alert message (haversine distance).
```

### Prompt 9 — Tests and cleanup
```
Write pytest tests for: route computation, women-mode scoring giving a higher risk
than normal mode on isolated night roads, /api/sos message formatting, and
deviation detection logic. Run them and fix failures. Then update README.md with
setup steps, the architecture diagram in text, and a clear "Limitations" section.
```

---

## PART 4: Things to be honest about (important for viva)

- **SMS cannot be sent from a browser alone.** It needs a backend plus a service such as Twilio (paid, small trial credit) or a free fallback: opening an `sms:` link on a phone. Say this clearly.
- **Your road risk data is simulated/estimated**, not official crime or accident records. Describe the women-safety scoring as a *decision-support estimate*, not a guarantee of safety.
- **The app does not replace emergency services.** Keep the 112 / 1091 / 181 buttons prominent.
- **Privacy:** location is sensitive. Share it only after the user presses SOS or starts a trip share, store the minimum, and let trips expire.
- **Do not commit secrets.** Put API keys in a `.env` file and add `.env` to `.gitignore`.

## Suggested build order
Prompt 1 > 2 > 3 > 4 > 5 > 6 > 7 > 8 > 9. Test after every prompt and commit to git each time it works.
