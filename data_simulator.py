"""
EcoDye AI - Wastewater Data Simulator
--------------------------------------
Generates realistic textile dyeing effluent sensor readings.
Ranges are based on typical TNPCB/CPCB discharge-standard bands for
textile dyeing units (used here for prototype/demo purposes).

Parameters simulated:
  pH            : 5.5 - 9.0 safe band   (legal limit: 5.5 - 9.0)
  BOD  (mg/L)   : biological oxygen demand   (legal limit: <= 30)
  COD  (mg/L)   : chemical oxygen demand     (legal limit: <= 250)
  TDS  (mg/L)   : total dissolved solids     (legal limit: <= 2100)
  color_admi    : color intensity (ADMI units)  (legal limit: <= 300)
  turbidity_ntu : turbidity                  (legal limit: <= 50)
  temperature_c : effluent temperature       (legal limit: <= 40)
  flow_lpm      : flow rate, liters/minute (operational, not a legal limit)

The simulator produces a smooth baseline with noise, slow drift, and
randomly injected "pollution events" (spikes) so downstream ML models
have both normal and violation examples to learn from.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ---- Legal / safe limits (used for labeling + dashboard thresholds) ----
LIMITS = {
    "pH_low": 5.5, "pH_high": 9.0,
    "bod": 30.0,
    "cod": 250.0,
    "tds": 2100.0,
    "color_admi": 300.0,
    "turbidity_ntu": 50.0,
    "temperature_c": 40.0,
}

RNG = np.random.default_rng(42)


def _baseline_series(n, mean, std, drift_scale=0.0):
    """Smooth-ish baseline: slow random walk + white noise."""
    drift = np.cumsum(RNG.normal(0, drift_scale, n))
    noise = RNG.normal(0, std, n)
    return mean + drift + noise


def _inject_events(series, n, event_prob, spike_mult_range, min_len=3, max_len=12):
    """Randomly inject sustained spike 'pollution events' into a series."""
    series = series.copy()
    event_mask = np.zeros(n, dtype=bool)
    i = 0
    while i < n:
        if RNG.random() < event_prob:
            length = RNG.integers(min_len, max_len)
            end = min(i + length, n)
            mult = RNG.uniform(*spike_mult_range)
            series[i:end] *= mult
            event_mask[i:end] = True
            i = end
        else:
            i += 1
    return series, event_mask


def generate_dataset(n_points=20000, freq_seconds=30, start=None, seed=42):
    """Generate a full historical dataset for training + demo playback."""
    global RNG
    RNG = np.random.default_rng(seed)

    start = start or (datetime.now() - timedelta(seconds=n_points * freq_seconds))
    timestamps = [start + timedelta(seconds=i * freq_seconds) for i in range(n_points)]

    pH = _baseline_series(n_points, mean=7.2, std=0.35, drift_scale=0.01)
    pH = np.clip(pH, 3.5, 12.5)

    bod = _baseline_series(n_points, mean=18, std=3.5, drift_scale=0.05)
    bod, bod_events = _inject_events(bod, n_points, 0.004, (1.8, 4.0))
    bod = np.clip(bod, 2, None)

    cod = _baseline_series(n_points, mean=160, std=25, drift_scale=0.4)
    cod, cod_events = _inject_events(cod, n_points, 0.004, (1.6, 3.2))
    cod = np.clip(cod, 20, None)

    tds = _baseline_series(n_points, mean=1500, std=150, drift_scale=2.0)
    tds, tds_events = _inject_events(tds, n_points, 0.003, (1.3, 1.9))
    tds = np.clip(tds, 200, None)

    color = _baseline_series(n_points, mean=180, std=30, drift_scale=0.5)
    color, color_events = _inject_events(color, n_points, 0.005, (1.5, 3.0))
    color = np.clip(color, 10, None)

    turbidity = _baseline_series(n_points, mean=28, std=6, drift_scale=0.1)
    turbidity, turb_events = _inject_events(turbidity, n_points, 0.004, (1.6, 2.8))
    turbidity = np.clip(turbidity, 1, None)

    temperature = _baseline_series(n_points, mean=32, std=2.0, drift_scale=0.02)
    temperature = np.clip(temperature, 20, 55)

    flow = np.clip(_baseline_series(n_points, mean=120, std=15, drift_scale=0.3), 10, None)

    any_event = bod_events | cod_events | tds_events | color_events | turb_events

    df = pd.DataFrame({
        "timestamp": timestamps,
        "pH": pH.round(2),
        "bod": bod.round(1),
        "cod": cod.round(1),
        "tds": tds.round(1),
        "color_admi": color.round(1),
        "turbidity_ntu": turbidity.round(1),
        "temperature_c": temperature.round(1),
        "flow_lpm": flow.round(1),
        "is_event": any_event,
    })

    # ---- Risk label: Safe / Needs Treatment / Hazardous ----
    def classify(row):
        violations = 0
        if not (LIMITS["pH_low"] <= row.pH <= LIMITS["pH_high"]):
            violations += 1
        if row.bod > LIMITS["bod"]:
            violations += 1
        if row.cod > LIMITS["cod"]:
            violations += 1
        if row.tds > LIMITS["tds"]:
            violations += 1
        if row.color_admi > LIMITS["color_admi"]:
            violations += 1
        if row.turbidity_ntu > LIMITS["turbidity_ntu"]:
            violations += 1
        if violations == 0:
            return "Safe"
        elif violations == 1:
            return "Needs Treatment"
        else:
            return "Hazardous"

    df["risk_label"] = df.apply(classify, axis=1)
    return df


if __name__ == "__main__":
    df = generate_dataset(n_points=20000, freq_seconds=30)
    df.to_csv("/home/claude/ecodye_prototype/data/effluent_history.csv", index=False)
    print(f"Generated {len(df)} rows")
    print(df["risk_label"].value_counts())
    print(df.head())
