"""
EcoDye AI - Treatment Recommendation Engine
---------------------------------------------
Rule-based engine: given a reading (and which parameters are out of range),
suggests concrete corrective actions. Deliberately rule-based rather than
ML-based - transparent, explainable, and fast to justify to judges/inspectors.
"""

from data_simulator import LIMITS


def get_recommendations(reading: dict):
    recs = []

    if not (LIMITS["pH_low"] <= reading["pH"] <= LIMITS["pH_high"]):
        if reading["pH"] < LIMITS["pH_low"]:
            recs.append({
                "parameter": "pH",
                "issue": f"pH too low ({reading['pH']})",
                "action": "Dose with alkali (lime/soda ash) to neutralize acidity before discharge.",
            })
        else:
            recs.append({
                "parameter": "pH",
                "issue": f"pH too high ({reading['pH']})",
                "action": "Dose with acid (dilute H2SO4) to bring pH back within 5.5-9.0.",
            })

    if reading["bod"] > LIMITS["bod"]:
        recs.append({
            "parameter": "BOD",
            "issue": f"BOD elevated ({reading['bod']} mg/L, limit {LIMITS['bod']})",
            "action": "Increase aeration time in the biological treatment stage; check aerator load.",
        })

    if reading["cod"] > LIMITS["cod"]:
        recs.append({
            "parameter": "COD",
            "issue": f"COD elevated ({reading['cod']} mg/L, limit {LIMITS['cod']})",
            "action": "Increase aeration / oxidation dosing; verify pre-treatment chemical dosing rate.",
        })

    if reading["tds"] > LIMITS["tds"]:
        recs.append({
            "parameter": "TDS",
            "issue": f"TDS elevated ({reading['tds']} mg/L, limit {LIMITS['tds']})",
            "action": "Route through RO (reverse osmosis) unit; check salt-heavy dye batches upstream.",
        })

    if reading["color_admi"] > LIMITS["color_admi"]:
        recs.append({
            "parameter": "Color (ADMI)",
            "issue": f"Color intensity high ({reading['color_admi']} ADMI, limit {LIMITS['color_admi']})",
            "action": "Apply advanced oxidation (ozonation) or activated-carbon polishing stage.",
        })

    if reading["turbidity_ntu"] > LIMITS["turbidity_ntu"]:
        recs.append({
            "parameter": "Turbidity",
            "issue": f"Turbidity high ({reading['turbidity_ntu']} NTU, limit {LIMITS['turbidity_ntu']})",
            "action": "Increase settling/coagulation time; check flocculant dosing.",
        })

    if reading["temperature_c"] > LIMITS["temperature_c"]:
        recs.append({
            "parameter": "Temperature",
            "issue": f"Temperature high ({reading['temperature_c']} C, limit {LIMITS['temperature_c']})",
            "action": "Route through cooling tower/tank before discharge or reuse.",
        })

    if not recs:
        recs.append({
            "parameter": "All",
            "issue": "All parameters within safe limits",
            "action": "No corrective action needed - continue routine monitoring.",
        })

    return recs
