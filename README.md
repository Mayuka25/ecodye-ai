# EcoDye AI — Working Software Prototype

AI-powered textile wastewater pollution prediction & sustainable management platform.
This is a **fully working software prototype** — simulated sensor data flowing through
real trained ML models into a live dashboard. No physical hardware required to run or demo it.

## What's included

| File | Purpose |
|---|---|
| `data_simulator.py` | Generates realistic effluent readings (pH, BOD, COD, TDS, color, turbidity, temperature, flow) with injected pollution events, based on TNPCB/CPCB-style discharge limits |
| `train_models.py` | Trains 3 ML models: COD forecaster (RandomForestRegressor), risk classifier (RandomForestClassifier: Safe/Needs Treatment/Hazardous), anomaly detector (IsolationForest) |
| `recommendation_engine.py` | Rule-based treatment recommendation logic (explainable — good for judges/inspectors) |
| `app.py` | FastAPI backend: simulates a live stream, runs it through all 3 models, computes a risk score, streams results over WebSocket, exposes REST endpoints |
| `static/index.html` | Live dashboard: real-time charts, risk score, predictive alerts, anomaly log, recommendations panel, session summary |
| `data/effluent_history.csv` | Pre-generated training dataset (20,000 readings) |
| `models/*.joblib` | Pre-trained models — ready to use immediately, no retraining needed |


## API endpoints

- `GET /` — the dashboard (HTML)
- `WS /ws/live` — live data stream (JSON messages: reading + risk label + risk score + anomaly flag + forecast + recommendations)
- `GET /api/history?limit=200` — recent readings history
- `GET /api/compliance-report` — summary stats (safe/needs-treatment/hazardous counts, anomaly count, compliance %)

## How the AI layer works

1. **Prediction** — a RandomForestRegressor looks at the last 5 readings (2.5 min of
   history) and forecasts COD 2.5 minutes ahead. If the forecast crosses the legal
   limit (250 mg/L) while the current reading is still safe, a predictive alert fires.
2. **Classification** — a RandomForestClassifier labels each reading Safe / Needs
   Treatment / Hazardous based on all 8 parameters together (test accuracy: ~99-100%
   on held-out simulated data).
3. **Anomaly detection** — an IsolationForest flags readings that don't fit the
   learned "normal" pattern — catches sudden dumping events even if no single
   parameter has crossed its legal limit yet.
4. **Recommendation** — a transparent rule-based engine maps out-of-range parameters
   to concrete corrective actions (e.g. "High COD → increase aeration").

## Deploying it live (for the hackathon demo)

This runs anywhere Python/FastAPI runs. Free options:
- **Render** or **Railway** — push this folder, they auto-detect and run `uvicorn app:app`
- Add a `Procfile` with `web: uvicorn app:app --host 0.0.0.0 --port $PORT` if the platform needs one

## Future scope (not in this prototype)

- Real ESP32 + pH/TDS/turbidity sensor integration (replace `next_live_reading()` in
  `app.py` with real sensor input over MQTT/serial)
- Multi-factory cluster view
- PDF compliance report export
- SMS/email alert integration
