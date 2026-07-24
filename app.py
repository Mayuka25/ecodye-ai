"""
EcoDye AI - Backend API
-------------------------
FastAPI server that:
  - Simulates a live effluent data stream for a cluster of factories
    (Tiruppur, Erode, Karur), reusing data_simulator logic
  - Runs each reading through the 3 trained ML models
  - Computes a combined pollution risk score
  - Generates treatment recommendations
  - Simulates Email/SMS alert notifications
  - Streams everything to the dashboard over WebSocket
  - Exposes REST endpoints for history / compliance report / notifications

Run:  uvicorn app:app --host 0.0.0.0 --port 8000
Then open http://localhost:8000 in a browser.
"""

import asyncio
import json
from collections import deque
from datetime import datetime

import joblib
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from data_simulator import LIMITS
from recommendation_engine import get_recommendations
from report_generator import build_compliance_pdf

app = FastAPI(title="EcoDye AI API")
app.mount("/static", StaticFiles(directory="static"), name="static")

FEATURES = ["pH", "bod", "cod", "tds", "color_admi", "turbidity_ntu", "temperature_c", "flow_lpm"]
WINDOW = 5
FORECAST_HORIZON = 5

# ---------------- Load trained models ----------------
predictor = joblib.load("models/predictor.joblib")
classifier = joblib.load("models/classifier.joblib")
label_encoder = joblib.load("models/label_encoder.joblib")
anomaly_model = joblib.load("models/anomaly.joblib")

# ---------------- Factory cluster configuration ----------------
FACTORIES = [
    {"id": "tiruppur", "name": "Tiruppur Unit", "location": "Tiruppur, TN", "seed_offset": 0},
    {"id": "erode", "name": "Erode Unit", "location": "Erode, TN", "seed_offset": 5000},
    {"id": "karur", "name": "Karur Unit", "location": "Karur, TN", "seed_offset": 9000},
]
FACTORY_IDS = [f["id"] for f in FACTORIES]

FACTORY_MANAGER_EMAIL = "factory.manager@ecodye.local"
FACTORY_MANAGER_PHONE = "+91-98XXXXXX21"

# ---------------- Per-factory in-memory live state ----------------
HISTORY = {fid: deque(maxlen=500) for fid in FACTORY_IDS}
RECENT_WINDOW = {fid: deque(maxlen=WINDOW) for fid in FACTORY_IDS}
NOTIFY_STATE = {fid: {"last_label": None, "last_notified_at": None} for fid in FACTORY_IDS}
NOTIFICATION_LOG = {fid: deque(maxlen=100) for fid in FACTORY_IDS}


def maybe_generate_notification(fid, risk_label, anomaly_flag, reading):
    """Simulate sending an Email/SMS alert to the factory manager.
    Throttled so it doesn't spam on every reading - only fires on a
    transition into Hazardous, or on a fresh anomaly (30s cooldown)."""
    now = datetime.now()
    state = NOTIFY_STATE[fid]
    should_notify, reason, channel = False, None, None

    if risk_label == "Hazardous" and state["last_label"] != "Hazardous":
        should_notify, reason, channel = True, "risk level reached Hazardous", "Email"
    elif anomaly_flag and (
        state["last_notified_at"] is None
        or (now - state["last_notified_at"]).total_seconds() > 30
    ):
        should_notify, reason, channel = True, "anomaly detected in effluent stream", "SMS"

    notif = None
    if should_notify:
        recipient = FACTORY_MANAGER_EMAIL if channel == "Email" else FACTORY_MANAGER_PHONE
        message = (
            f"[EcoDye AI Alert] {reason.capitalize()} at {now.strftime('%H:%M:%S')} - "
            f"COD {reading['cod']} mg/L, risk: {risk_label}. Immediate review recommended."
        )
        notif = {
            "channel": channel,
            "recipient": recipient,
            "message": message,
            "timestamp": now.isoformat(),
        }
        NOTIFICATION_LOG[fid].append(notif)
        state["last_notified_at"] = now

    state["last_label"] = risk_label
    return notif


def risk_score_from_label(label, anomaly_flag):
    base = {"Safe": 15, "Needs Treatment": 55, "Hazardous": 90}.get(label, 50)
    if anomaly_flag:
        base = min(100, base + 20)
    return base


def next_live_reading(fid: str, step: int):
    """Generate one realistic live reading for a given factory. Each factory
    uses a different seed offset so their streams diverge and feel distinct."""
    offset = next(f["seed_offset"] for f in FACTORIES if f["id"] == fid)
    rng = np.random.default_rng(1000 + step + offset)

    def val(mean, std, event_prob=0.02, spike_mult=(1.5, 3.0), lo=None):
        v = mean + rng.normal(0, std)
        is_event = rng.random() < event_prob
        if is_event:
            v *= rng.uniform(*spike_mult)
        if lo is not None:
            v = max(v, lo)
        return round(float(v), 2), is_event

    pH, _ = val(7.2, 0.35, event_prob=0.01, spike_mult=(0.6, 0.8), lo=3.5)
    bod, e1 = val(18, 3.5, lo=2)
    cod, e2 = val(160, 25, lo=20)
    tds, e3 = val(1500, 150, lo=200)
    color, e4 = val(180, 30, lo=10)
    turbidity, e5 = val(28, 6, lo=1)
    temperature, _ = val(32, 2.0, event_prob=0.005, spike_mult=(1.2, 1.4), lo=20)
    flow, _ = val(120, 15, event_prob=0.0, lo=10)

    return {
        "pH": pH, "bod": bod, "cod": cod, "tds": tds,
        "color_admi": color, "turbidity_ntu": turbidity,
        "temperature_c": temperature, "flow_lpm": flow,
    }, any([e1, e2, e3, e4, e5])


def process_reading(fid: str, reading: dict, injected_event: bool):
    feat_vec = np.array([[reading[f] for f in FEATURES]])

    # classification
    class_idx = classifier.predict(feat_vec)[0]
    risk_label = label_encoder.inverse_transform([class_idx])[0]

    # anomaly detection
    anomaly_flag = anomaly_model.predict(feat_vec)[0] == -1

    # forecast (needs WINDOW readings of history)
    window = RECENT_WINDOW[fid]
    window.append([reading[f] for f in FEATURES])
    forecast_cod = None
    if len(window) == WINDOW:
        window_vec = np.array(window).flatten().reshape(1, -1)
        forecast_cod = round(float(predictor.predict(window_vec)[0]), 1)

    score = risk_score_from_label(risk_label, anomaly_flag)
    recs = get_recommendations(reading)

    alert = None
    if forecast_cod is not None and forecast_cod > LIMITS["cod"] and reading["cod"] <= LIMITS["cod"]:
        alert = f"COD predicted to exceed the {LIMITS['cod']} mg/L limit within ~2.5 min (forecast: {forecast_cod} mg/L)"

    combined_anomaly = bool(anomaly_flag) or injected_event
    notification = maybe_generate_notification(fid, risk_label, combined_anomaly, reading)

    result = {
        "factory_id": fid,
        "timestamp": datetime.now().isoformat(),
        "reading": reading,
        "risk_label": risk_label,
        "risk_score": score,
        "anomaly": combined_anomaly,
        "forecast_cod": forecast_cod,
        "predictive_alert": alert,
        "recommendations": recs,
        "limits": LIMITS,
        "notification": notification,
    }
    HISTORY[fid].append(result)
    return result


@app.get("/", response_class=HTMLResponse)
async def index():
    with open("static/index.html", encoding="utf-8") as f:
        return f.read()


@app.get("/api/factories")
async def get_factories():
    return JSONResponse(FACTORIES)


@app.get("/api/history")
async def get_history(factory: str = "tiruppur", limit: int = 200):
    if factory not in HISTORY:
        return JSONResponse({"error": "unknown factory id"}, status_code=404)
    return JSONResponse(list(HISTORY[factory])[-limit:])


def _build_compliance_report(factory: str):
    """Shared logic for JSON and PDF compliance report endpoints."""
    data = list(HISTORY[factory])
    if not data:
        return {"message": "No data yet"}
    total = len(data)
    hazardous = sum(1 for d in data if d["risk_label"] == "Hazardous")
    needs_treatment = sum(1 for d in data if d["risk_label"] == "Needs Treatment")
    safe = total - hazardous - needs_treatment
    anomalies = sum(1 for d in data if d["anomaly"])
    return {
        "factory": factory,
        "generated_at": datetime.now().isoformat(),
        "total_readings": total,
        "safe": safe,
        "needs_treatment": needs_treatment,
        "hazardous": hazardous,
        "anomaly_events": anomalies,
        "compliance_rate_pct": round(safe / total * 100, 1),
        "legal_limits": LIMITS,
    }


@app.get("/api/compliance-report")
async def compliance_report(factory: str = "tiruppur"):
    """Simple compliance summary over stored history for one factory."""
    if factory not in HISTORY:
        return JSONResponse({"error": "unknown factory id"}, status_code=404)
    return JSONResponse(_build_compliance_report(factory))


@app.get("/api/compliance-report/pdf")
async def compliance_report_pdf(factory: str = "tiruppur"):
    """Downloadable PDF version of the compliance report."""
    if factory not in HISTORY:
        return JSONResponse({"error": "unknown factory id"}, status_code=404)
    report = _build_compliance_report(factory)
    meta = next(f for f in FACTORIES if f["id"] == factory)
    notifications = list(NOTIFICATION_LOG[factory])[::-1]  # most recent first
    pdf_bytes = build_compliance_pdf(meta["name"], meta["location"], report, notifications)
    filename = f"EcoDye_AI_Compliance_Report_{factory}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/notifications")
async def get_notifications(factory: str = "tiruppur", limit: int = 50):
    if factory not in NOTIFICATION_LOG:
        return JSONResponse({"error": "unknown factory id"}, status_code=404)
    return JSONResponse(list(NOTIFICATION_LOG[factory])[-limit:])


@app.websocket("/ws/live")
async def live_stream(websocket: WebSocket):
    """Streams one combined message per tick containing the latest reading
    for every factory in the cluster: { factories: { <id>: <result>, ... } }"""
    await websocket.accept()
    step = max(len(HISTORY[fid]) for fid in FACTORY_IDS)
    try:
        while True:
            factories_payload = {}
            for f in FACTORIES:
                fid = f["id"]
                reading, injected_event = next_live_reading(fid, step)
                result = process_reading(fid, reading, injected_event)
                factories_payload[fid] = result
            await websocket.send_text(json.dumps({
                "timestamp": datetime.now().isoformat(),
                "factories": factories_payload,
            }))
            step += 1
            await asyncio.sleep(1.5)
    except WebSocketDisconnect:
        pass
