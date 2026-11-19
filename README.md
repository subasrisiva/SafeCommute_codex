<<<<<<< HEAD
# SafeCommute (VSCode + Codex)

The route network covers Tamil Nadu plus Pondicherry/Puducherry. Routes can be
scored for Morning, Afternoon, Evening, Night, or the conservative Any time mode.
It includes Gingee (Senji), Tindivanam, Melmalayanur, Tirukkoyilur, Kallakurichi,
Ulundurpet, Chidambaram, Mayiladuthurai, Perambalur, Palani, Aruppukkottai, and
Paramakudi alongside the larger towns.
1. `python -m venv venv` then `venv\Scripts\activate`
2. `pip install -r requirements.txt`
3. `python run_all.py` (builds the dataset, trains models, and exports the network)
4. Start the realtime backend in a second terminal: `python app.py` (optional for
   local map display, needed for backend trip alerts).
5. Run the HTML through VS Code Live Server: click **Go Live** in the VS Code status
   bar. Keep the page in VS Code's integrated browser tab if you do not want an
   external browser.
6. In the app, choose a **From** place and a **To** place from the Tamil Nadu +
   Pondicherry lists, then click **Find Safest Route**.
7. If live GPS tracking is needed during the trip, click **Start Realtime Tracking**
   and allow location access.

The map uses Leaflet/OpenStreetMap when those files are reachable. If the page is
opened offline or the CDN is blocked, it automatically switches to an embedded
SVG route map so the towns, route lines, and live simulation remain visible.

## Realtime safety alerts

The browser sends GPS positions to Flask while a trip is active. The backend keeps
trip state for 24 hours, detects positions more than 1 km from the planned route,
and returns low-safety-segment alerts. Women Safety Mode can store one trusted
mobile number for SMS-only risk alerts through `/api/risk-alert`.

For real SMS delivery, configure Twilio in the terminal before starting the backend;
never put these values in HTML or source code:

```powershell
$env:TWILIO_ACCOUNT_SID = "your-account-sid"
$env:TWILIO_AUTH_TOKEN = "your-auth-token"
$env:TWILIO_FROM = "+15551234567"
python app.py
```

Without Twilio, the safety screen opens the phone's SMS composer for the saved trusted
number and includes a Google Maps location link. A browser cannot silently send an
SMS by itself, so the user must press Send in the phone's SMS app. The app does not
place calls. Location and route data are stored only in the backend process and expire
when the trip reaches 24 hours or the server restarts.
=======
# SafeCommute_codex
AI-Driven Route Safety Analyzer
>>>>>>> b39ba30a5c9364f15592de8df636cd0dd264d13b
