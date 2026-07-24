"""
EcoDye AI - PDF Compliance Report Generator
----------------------------------------------
Builds a branded compliance report PDF from a factory's live session data,
summarizing readings against TNPCB/CPCB-style discharge limits.
"""

import io
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
)

DEEP = colors.HexColor("#065A82")
NAVY = colors.HexColor("#21295C")
MINT = colors.HexColor("#02C39A")
WARN = colors.HexColor("#F4A100")
DANGER = colors.HexColor("#E23E3E")
MUTED = colors.HexColor("#5A6B7A")
CARD = colors.HexColor("#EDF4F7")


def build_compliance_pdf(factory_name: str, factory_location: str, report: dict, notifications: list) -> bytes:
    """report: output of the /api/compliance-report logic (dict)
    notifications: list of recent notification dicts"""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        topMargin=18 * mm, bottomMargin=16 * mm, leftMargin=18 * mm, rightMargin=18 * mm,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleC", parent=styles["Title"], textColor=NAVY, fontSize=20, spaceAfter=2)
    subtitle_style = ParagraphStyle("SubtitleC", parent=styles["Normal"], textColor=MUTED, fontSize=10.5, spaceAfter=12)
    h2_style = ParagraphStyle("H2C", parent=styles["Heading2"], textColor=DEEP, fontSize=13, spaceBefore=14, spaceAfter=6)
    body_style = ParagraphStyle("BodyC", parent=styles["Normal"], fontSize=10, leading=14, textColor=colors.HexColor("#13202B"))
    small_style = ParagraphStyle("SmallC", parent=styles["Normal"], fontSize=8.5, textColor=MUTED)

    story = []

    story.append(Paragraph("EcoDye AI — Compliance Report", title_style))
    story.append(Paragraph(
        f"{factory_name} &middot; {factory_location}<br/>"
        f"Generated: {datetime.now().strftime('%d %B %Y, %H:%M')}",
        subtitle_style,
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=CARD, spaceAfter=10))

    if "message" in report:
        story.append(Paragraph(report["message"], body_style))
        doc.build(story)
        return buf.getvalue()

    # ---- Summary table ----
    story.append(Paragraph("Session Summary", h2_style))
    summary_rows = [
        ["Metric", "Value"],
        ["Total readings analyzed", str(report["total_readings"])],
        ["Safe", str(report["safe"])],
        ["Needs Treatment", str(report["needs_treatment"])],
        ["Hazardous", str(report["hazardous"])],
        ["Anomaly events detected", str(report["anomaly_events"])],
        ["Overall compliance rate", f"{report['compliance_rate_pct']}%"],
    ]
    t = Table(summary_rows, colWidths=[90 * mm, 60 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, CARD]),
        ("GRID", (0, 0), (-1, -1), 0.5, CARD),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(t)

    # Compliance rate color note
    rate = report["compliance_rate_pct"]
    rate_color = "green (good standing)" if rate >= 90 else "amber (monitor closely)" if rate >= 70 else "red (needs urgent attention)"
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"<i>Compliance status: {rate_color}</i>", small_style))

    # ---- Legal limits reference table ----
    story.append(Paragraph("Legal Discharge Limits Referenced (TNPCB/CPCB-style)", h2_style))
    limits = report["legal_limits"]
    limit_rows = [
        ["Parameter", "Limit"],
        ["pH", f"{limits['pH_low']} - {limits['pH_high']}"],
        ["BOD (mg/L)", f"<= {limits['bod']}"],
        ["COD (mg/L)", f"<= {limits['cod']}"],
        ["TDS (mg/L)", f"<= {limits['tds']}"],
        ["Color (ADMI)", f"<= {limits['color_admi']}"],
        ["Turbidity (NTU)", f"<= {limits['turbidity_ntu']}"],
        ["Temperature (deg C)", f"<= {limits['temperature_c']}"],
    ]
    t2 = Table(limit_rows, colWidths=[90 * mm, 60 * mm])
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), DEEP),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, CARD]),
        ("GRID", (0, 0), (-1, -1), 0.5, CARD),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(t2)

    # ---- Recent alerts ----
    story.append(Paragraph("Recent Alert Notifications", h2_style))
    if not notifications:
        story.append(Paragraph("No alerts were triggered during this session.", body_style))
    else:
        alert_rows = [["Time", "Channel", "Message"]]
        for n in notifications[:15]:
            t_str = datetime.fromisoformat(n["timestamp"]).strftime("%H:%M:%S")
            alert_rows.append([t_str, n["channel"], Paragraph(n["message"], small_style)])
        t3 = Table(alert_rows, colWidths=[22 * mm, 20 * mm, 108 * mm])
        t3.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, CARD]),
            ("GRID", (0, 0), (-1, -1), 0.5, CARD),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(t3)

    story.append(Spacer(1, 16))
    story.append(HRFlowable(width="100%", thickness=0.5, color=CARD))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "This report is generated from simulated/live effluent monitoring data by the EcoDye AI "
        "platform for demonstration purposes. Readings are compared against representative "
        "TNPCB/CPCB-style discharge limits for textile dyeing units.",
        small_style,
    ))

    doc.build(story)
    return buf.getvalue()
