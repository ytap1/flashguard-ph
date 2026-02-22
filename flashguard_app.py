"""
FlashGuard PH — Hackathon Demo Prototype (Streamlit + Gemini Tool Calling)

This version includes:
- Deterministic mock "sensor truth" for stable demos (works offline)
- Adversarial cross-check ("Nemesis") to block unsafe auto-dispatch
- Open‑Meteo live evidence (weather + flood proxy) as supporting context (best-effort)
- Clear UI explainability:
  * Status bar with Decision Clock label (seconds)
  * Nemesis badge (APPROVE / BLOCK)
  * KPI strip: Decision time, False alarm prevented, Human escalation
  * Signal vs Truth panel

Key design principle:
- The system can show live evidence, but actions are gated by strict rules.
"""

from __future__ import annotations

import json
import time
from typing import Optional, Dict, Any, Tuple

import requests
import streamlit as st
from google import genai
from google.genai import types


# ============================================================
# 0) DEMO DATA (MOCKED "OFFICIAL SENSOR TRUTH")
# ============================================================
# In production, these values would come from real APIs (river gauges, rainfall feeds, satellites).
# For demo reliability and offline operation, we hardcode them here.
MOCK_ENVIRONMENTAL_DATA: Dict[str, Dict[str, Any]] = {
    "Bulacan": {
        "timestamp": "2026-02-18T20:30:00Z",
        "river_basin": "Angat River System",
        "river_gauge_meters": 18.5,
        "critical_threshold_meters": 15.0,
        "rainfall_mm_per_hr": 45.2,
        "satellite_soil_saturation": "94%",
        "satellite_cloud_cover": "Heavy Nimbus",
        "status": "CRITICAL_SPILL_LEVEL",
        "data_source": "MOCK_OFFICIAL_SENSOR",
    },
    "Marikina": {
        "timestamp": "2026-02-18T20:30:00Z",
        "river_basin": "Marikina River Basin",
        "river_gauge_meters": 12.1,
        "critical_threshold_meters": 15.0,
        "rainfall_mm_per_hr": 10.5,
        "satellite_soil_saturation": "60%",
        "satellite_cloud_cover": "Moderate",
        "status": "NORMAL",
        "data_source": "MOCK_OFFICIAL_SENSOR",
    },
}


# ============================================================
# 0b) ADVERSARIAL DATA (MOCKED 2nd SOURCE)
# ============================================================
# Simulates an independent channel (e.g., radio + drone recon + rescue requests).
# Used by Nemesis cross-check to reduce false dispatches.
ADVERSARIAL_MOCK_DATA: Dict[str, Dict[str, Any]] = {
    "Bulacan": {
        "timestamp": "2026-02-18T20:31:00Z",
        "alt_source": "Citizen Radio + LGU Drone Recon",
        "alt_flood_status": "SEVERE_FLOOD_WARNING",
        "estimated_inundation_meters": 1.4,
        "rescue_requests_count": 7,
        "confidence": "HIGH",
        "data_source": "MOCK_ADVERSARIAL",
    },
    "Marikina": {
        "timestamp": "2026-02-18T20:31:00Z",
        "alt_source": "Citizen Radio + LGU Drone Recon",
        "alt_flood_status": "LOW_RISK",
        "estimated_inundation_meters": 0.1,
        "rescue_requests_count": 0,
        "confidence": "MEDIUM",
        "data_source": "MOCK_ADVERSARIAL",
    },
}


# ============================================================
# 0c) COORDINATES FOR OPEN‑METEO (demo convenience)
# ============================================================
# Used for Open‑Meteo live evidence. In production, replace with geocoding or station coords.
LOCATION_COORDS: Dict[str, Tuple[float, float]] = {
    "Bulacan": (14.85, 120.81),
    "Marikina": (14.65, 121.10),
}


# ============================================================
# 1) HELPER UTILITIES
# ============================================================
def _standard_tool_response(ok: bool, payload: Dict[str, Any], message: str) -> str:
    """
    Tools return a consistent JSON envelope:
      { ok: bool, message: str, payload: {...} }
    This makes tool-calling and debugging more reliable.
    """
    return json.dumps({"ok": ok, "message": message, "payload": payload}, ensure_ascii=False)


def _infer_location_from_text(input_text: str) -> str:
    """
    Extract a known location name from free text.
    Example: "Check flood status in bulacan pls" -> "Bulacan"
    """
    if not input_text:
        return input_text

    text = input_text.lower()

    for known_location in MOCK_ENVIRONMENTAL_DATA.keys():
        if known_location.lower() in text:
            return known_location

    for known_location in ADVERSARIAL_MOCK_DATA.keys():
        if known_location.lower() in text:
            return known_location

    for known_location in LOCATION_COORDS.keys():
        if known_location.lower() in text:
            return known_location

    return input_text


def _is_sensor_critical(sensor_record: Dict[str, Any]) -> bool:
    """
    Deterministic safety gate:
    - Dispatch allowed ONLY if sensors indicate critical risk.
    """
    if not sensor_record:
        return False
    return (
        sensor_record.get("status") == "CRITICAL_SPILL_LEVEL"
        or float(sensor_record.get("river_gauge_meters", 0) or 0)
        >= float(sensor_record.get("critical_threshold_meters", 999999) or 999999)
    )


def _mock_social_reports(area_name: str) -> Dict[str, Any]:
    """
    Mock "citizen signal" channel.
    Demo rule: Marikina always has chatter to demonstrate suppression against false alarms.
    """
    if area_name and "marikina" in area_name.lower():
        return {
            "verified_reports_count": 3,
            "highlights": [
                "Waist-deep water reported on Main Street",
                "1 user requesting boat rescue",
            ],
            "confidence": "MEDIUM",
            "data_source": "MOCK_SOCIAL",
        }
    return {
        "verified_reports_count": 0,
        "highlights": [],
        "confidence": "LOW",
        "data_source": "MOCK_SOCIAL",
    }


# ============================================================
# 2) OPEN‑METEO LIVE EVIDENCE (best-effort, cached)
# ============================================================
@st.cache_data(ttl=600)
def _fetch_open_meteo_weather(lat: float, lon: float) -> Dict[str, Any]:
    """
    Open‑Meteo Forecast API: current precipitation + short hourly preview.
    (No API key required for typical non-commercial usage.)
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,precipitation",
        "hourly": "precipitation,precipitation_probability",
        "forecast_days": 2,
        "timezone": "auto",
    }
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()

    current = data.get("current", {}) or {}
    hourly = data.get("hourly", {}) or {}

    return {
        "provider": "open-meteo",
        "endpoint": "forecast",
        "lat_used": data.get("latitude", lat),
        "lon_used": data.get("longitude", lon),
        "current": {
            "time": current.get("time"),
            "temperature_2m": current.get("temperature_2m"),
            "precipitation": current.get("precipitation"),
        },
        "hourly_preview": {
            "time": (hourly.get("time") or [])[:8],
            "precipitation": (hourly.get("precipitation") or [])[:8],
            "precipitation_probability": (hourly.get("precipitation_probability") or [])[:8],
        },
        "data_source": "OPEN_METEO_LIVE",
    }


@st.cache_data(ttl=600)
def _fetch_open_meteo_flood(lat: float, lon: float) -> Dict[str, Any]:
    """
    Open‑Meteo Flood API: modeled river discharge (GloFAS proxy).
    Used as supporting evidence only.
    """
    url = "https://flood-api.open-meteo.com/v1/flood"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "river_discharge,river_discharge_max",
        "forecast_days": 7,
        "timeformat": "iso8601",
    }
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()

    daily = data.get("daily", {}) or {}
    return {
        "provider": "open-meteo",
        "endpoint": "flood",
        "lat_cell": data.get("latitude"),
        "lon_cell": data.get("longitude"),
        "timezone": data.get("timezone"),
        "daily_preview": {
            "time": (daily.get("time") or [])[:7],
            "river_discharge": (daily.get("river_discharge") or [])[:7],
            "river_discharge_max": (daily.get("river_discharge_max") or [])[:7],
        },
        "data_source": "OPEN_METEO_LIVE",
    }


def _get_open_meteo_bundle(location: str) -> Dict[str, Any]:
    """
    Bundle live evidence for a location (weather + flood proxy).
    If offline/unavailable, returns structured errors (demo continues with mock data).
    """
    coords = LOCATION_COORDS.get(location)
    if not coords:
        return {"ok": False, "reason": "No coordinates mapped for this demo location."}

    lat, lon = coords
    out: Dict[str, Any] = {"ok": True, "location": location, "lat": lat, "lon": lon}

    try:
        out["weather"] = _fetch_open_meteo_weather(lat, lon)
    except Exception as e:
        out["weather"] = {"ok": False, "error": str(e), "data_source": "OPEN_METEO_FAILED"}

    try:
        out["flood"] = _fetch_open_meteo_flood(lat, lon)
    except Exception as e:
        out["flood"] = {"ok": False, "error": str(e), "data_source": "OPEN_METEO_FAILED"}

    return out


# ============================================================
# 3) NEMESIS (adversarial cross-check)
# ============================================================
def nemesis_ai(location_name: str) -> str:
    """
    Adversarial cross-check:
    - APPROVE only if BOTH primary sensor truth and adversarial source indicate danger
    - BLOCK if conflict or both indicate low risk
    """
    st.toast(f"🛡️ Nemesis: Cross-checking {location_name}...")

    primary = MOCK_ENVIRONMENTAL_DATA.get(location_name)
    adversarial = ADVERSARIAL_MOCK_DATA.get(location_name)

    if not primary and not adversarial:
        return json.dumps(
            {
                "location": location_name,
                "decision": "BLOCK",
                "reason": "No reliable data in either source.",
                "used_sources": ["MOCK_ENVIRONMENTAL_DATA", "ADVERSARIAL_MOCK_DATA"],
            },
            ensure_ascii=False,
        )

    primary_is_critical = False
    if primary:
        primary_is_critical = (
            float(primary.get("river_gauge_meters", 0) or 0)
            >= float(primary.get("critical_threshold_meters", 999) or 999)
            or "CRITICAL" in str(primary.get("status", "")).upper()
        )

    adversarial_is_critical = False
    if adversarial:
        adversarial_is_critical = (
            str(adversarial.get("alt_flood_status", "")).upper() in {"SEVERE_FLOOD_WARNING", "CRITICAL"}
            or float(adversarial.get("estimated_inundation_meters", 0) or 0) >= 0.5
            or int(adversarial.get("rescue_requests_count", 0) or 0) >= 2
        )

    if primary_is_critical and adversarial_is_critical:
        decision = "APPROVE"
        reason = "Both sources indicate flood danger."
    elif primary_is_critical != adversarial_is_critical:
        decision = "BLOCK"
        reason = "Sources conflict. Escalate for verification before dispatch."
    else:
        decision = "BLOCK"
        reason = "Both sources indicate low risk for auto-dispatch."

    return json.dumps(
        {
            "location": location_name,
            "decision": decision,
            "reason": reason,
            "primary_is_critical": primary_is_critical,
            "adversarial_is_critical": adversarial_is_critical,
            "used_sources": ["MOCK_ENVIRONMENTAL_DATA", "ADVERSARIAL_MOCK_DATA"],
        },
        ensure_ascii=False,
    )


# ============================================================
# 4) KPI + BADGE STATE (generic)
# ============================================================
def _set_kpi_state(
    decision_seconds: float,
    false_alarm_prevented: bool,
    human_escalation: bool,
    nemesis_decision: Optional[str],
    nemesis_reason: Optional[str],
) -> None:
    """
    Store KPI values for display.
    """
    st.session_state.kpi_decision_seconds = decision_seconds
    st.session_state.kpi_false_alarm_prevented = false_alarm_prevented
    st.session_state.kpi_human_escalation = human_escalation
    st.session_state.kpi_nemesis_decision = nemesis_decision
    st.session_state.kpi_nemesis_reason = nemesis_reason


def _render_kpi_bar() -> None:
    """
    KPI strip under the status bar.
    """
    sec = st.session_state.get("kpi_decision_seconds", None)
    fa = st.session_state.get("kpi_false_alarm_prevented", None)
    he = st.session_state.get("kpi_human_escalation", None)

    if sec is None and fa is None and he is None:
        return

    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        st.metric("Decision time (sec)", f"{sec:.2f}" if isinstance(sec, (int, float)) else "—")
    with c2:
        st.metric("False alarm prevented", "Yes ✅" if fa else "No")
    with c3:
        st.metric("Human escalation", "Yes ⚠️" if he else "No")


def _render_nemesis_badge() -> None:
    """
    Always-visible Nemesis badge for transparency.
    """
    decision = st.session_state.get("kpi_nemesis_decision")
    reason = st.session_state.get("kpi_nemesis_reason")

    if not decision:
        st.info("🛡️ Nemesis: N/A (No dispatch path evaluated yet)")
        return

    if decision == "APPROVE":
        st.success("🛡️ Nemesis: APPROVE ✅")
    else:
        st.warning(f"🛡️ Nemesis: BLOCK ⛔ — {reason or 'Blocked by adversarial cross-check.'}")


# ============================================================
# 5) STATUS BAR WITH DECISION CLOCK LABEL
# ============================================================
def _render_status_bar(location: Optional[str]) -> None:
    """
    Status bar shows current state + decision clock label.
    - Decision clock uses last computed KPI decision time (if available).
    """
    decision_clock = st.session_state.get("kpi_decision_seconds")
    clock_label = f"⏱ Decision: {decision_clock:.2f}s" if isinstance(decision_clock, (int, float)) else "⏱ Decision: —"

    if not location or location not in MOCK_ENVIRONMENTAL_DATA:
        st.markdown(
            f"""
<div style="padding:0.85rem 1rem;border-radius:0.6rem;background:#1565C0;color:#fff;font-weight:800;margin-bottom:0.75rem;">
🟦 READY — awaiting location check <span style="font-weight:500;">| {clock_label}</span>
<div style="margin-top:0.25rem;font-weight:500;opacity:0.95;">
Use demo buttons (left) or type: "Check flood status in Bulacan" / "Marikina"
</div>
</div>
""",
            unsafe_allow_html=True,
        )
        return

    sensor = MOCK_ENVIRONMENTAL_DATA[location]
    critical = _is_sensor_critical(sensor)

    bg = "#B71C1C" if critical else "#1B5E20"
    icon = "🟥" if critical else "🟩"
    label = "CRITICAL (DISPATCH ELIGIBLE)" if critical else "NORMAL (NO ALERT)"

    basin = sensor.get("river_basin", "—")
    gauge = sensor.get("river_gauge_meters", "—")
    threshold = sensor.get("critical_threshold_meters", "—")
    rain = sensor.get("rainfall_mm_per_hr", "—")

    st.markdown(
        f"""
<div style="padding:0.85rem 1rem;border-radius:0.6rem;background:{bg};color:#fff;font-weight:800;margin-bottom:0.75rem;">
{icon} {label} — {location}
<span style="font-weight:400;"> | Basin: {basin} | River: {gauge}m / {threshold}m | Rain: {rain} mm/hr</span>
<span style="font-weight:500;"> | {clock_label}</span>
<div style="margin-top:0.25rem;font-weight:500;opacity:0.95;">
Decision is grounded in sensor truth; citizen reports are treated as signal.
</div>
</div>
""",
        unsafe_allow_html=True,
    )


# ============================================================
# 6) GEMINI TOOL FUNCTIONS (callable by the model)
# ============================================================
def check_pagasa_water_level(location_name: str) -> str:
    """
    Tool: returns sensor truth (mock) + adversarial data + Open-Meteo evidence.
    The AI uses this to produce grounded, explainable responses.
    """
    location_key = _infer_location_from_text(location_name)
    st.toast(f"📡 System: Checking sensor feeds for {location_key}...")

    primary = MOCK_ENVIRONMENTAL_DATA.get(location_key)
    adversarial = ADVERSARIAL_MOCK_DATA.get(location_key)

    st.toast(f"🌐 System: Pulling Open‑Meteo evidence for {location_key}...")
    open_meteo = _get_open_meteo_bundle(location_key)

    if primary:
        payload = dict(primary)
        payload["adversarial"] = adversarial
        payload["open_meteo"] = open_meteo
        payload["data_source"] = "MOCK_OFFICIAL_SENSOR + ADVERSARIAL(mock) + OPEN_METEO(best-effort)"
        return _standard_tool_response(True, payload, f"Found sensor record for {location_key}.")

    payload = {"location": location_key, "adversarial": adversarial, "open_meteo": open_meteo}
    return _standard_tool_response(False, payload, "No sensor record found in demo dataset.")


def check_social_media_reports(area_name: str) -> str:
    """
    Tool: returns deterministic mock citizen reports.
    """
    st.toast(f"📱 System: Scanning citizen reports for {area_name}...")
    reports = _mock_social_reports(area_name)
    if reports["verified_reports_count"] > 0:
        return _standard_tool_response(True, reports, "Citizen reports detected.")
    return _standard_tool_response(True, reports, "No recent citizen reports found.")


def dispatch_emergency_alert(area_name: str, action_plan: str) -> str:
    """
    Tool: dispatch with enforced safety gates.
    Gate 1: sensor truth must be CRITICAL
    Gate 2: Nemesis must APPROVE (blocks if conflict/low-risk)
    """
    location_key = _infer_location_from_text(area_name)
    st.toast(f"🚨 ACTION: Validating before dispatch to {location_key}...")

    sensor_record = MOCK_ENVIRONMENTAL_DATA.get(location_key, {})

    # Gate 1: sensor truth
    if not _is_sensor_critical(sensor_record):
        return _standard_tool_response(
            ok=False,
            payload={
                "area": location_key,
                "reason": "Dispatch blocked: sensors are not critical.",
                "sensor_status": sensor_record.get("status", "UNKNOWN"),
            },
            message="SAFETY BLOCK: No evacuation alert sent.",
        )

    # Gate 2: Nemesis cross-check
    adversarial_verdict = json.loads(nemesis_ai(location_key))
    if adversarial_verdict.get("decision") != "APPROVE":
        st.toast(f"🛑 BLOCKED: Nemesis requires verification for {location_key}.")
        return _standard_tool_response(
            ok=False,
            payload={
                "area": location_key,
                "reason": adversarial_verdict.get("reason", "Blocked by adversarial cross-check."),
                "nemesis_decision": adversarial_verdict.get("decision", "BLOCK"),
            },
            message="SAFETY BLOCK: Dispatch was blocked by Nemesis.",
        )

    # Dispatch proceeds
    st.toast(f"🚨 ACTION TRIGGERED: Dispatching units to {location_key}!")
    return _standard_tool_response(
        ok=True,
        payload={
            "area": location_key,
            "action_plan": action_plan,
            "dispatch": "Evacuation SMS broadcasted + resources coordinated (mock).",
            "nemesis_decision": adversarial_verdict.get("decision", "APPROVE"),
        },
        message="SUCCESS: Evacuation alert dispatched.",
    )


# ============================================================
# 7) STREAMLIT UI SETUP
# ============================================================
st.set_page_config(page_title="FlashGuard PH", page_icon="⛈️", layout="centered")
st.title("⛈️ FlashGuard PH")
st.subheader("Autonomous Crisis Management System")
st.caption("Runs in mock mode with optional live evidence (Open‑Meteo).")


# Sidebar: demo controls + optional panels
with st.sidebar:
    st.header("🎬 Demo Controls")
    st.caption("One-click prompts for two demo scenarios.")

    if st.button("Scenario A: Bulacan (Critical → Dispatch)"):
        st.session_state["_demo_prompt"] = "Check flood status in Bulacan."

    if st.button("Scenario B: Marikina (Normal → Suppress)"):
        st.session_state["_demo_prompt"] = "Check flood status in Marikina."

    if st.button("Reset"):
        st.session_state.pop("messages", None)
        st.session_state.pop("debug_log", None)
        st.session_state.pop("active_location", None)
        st.session_state.pop("kpi_decision_seconds", None)
        st.session_state.pop("kpi_false_alarm_prevented", None)
        st.session_state.pop("kpi_human_escalation", None)
        st.session_state.pop("kpi_nemesis_decision", None)
        st.session_state.pop("kpi_nemesis_reason", None)
        st.rerun()

    st.divider()
    st.subheader("⚙️ Optional Panels")
    show_signal_truth = st.checkbox("Show Signal vs Truth", value=True)
    show_open_meteo = st.checkbox("Show Open‑Meteo evidence", value=True)
    show_nemesis_panel = st.checkbox("Show full Nemesis output", value=False)
    show_raw = st.checkbox("Show raw demo datasets", value=False)


# ============================================================
# 8) INITIALIZE GEMINI CHAT AGENT (ONCE PER SESSION)
# ============================================================
# Streamlit secrets must include GEMINI_API_KEY.
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    st.error("Missing GEMINI_API_KEY in Streamlit secrets. Add it to .streamlit/secrets.toml")
    st.stop()

if "crisis_agent" not in st.session_state:
    client = genai.Client(api_key=api_key)

    directive = """
You are FlashGuard PH, an autonomous disaster response AI.

NON-NEGOTIABLE PROTOCOL:
1) When a user asks about a location or flood status, you MUST call:
   - check_pagasa_water_level(location)
   - check_social_media_reports(location)

2) Only if sensor truth indicates CRITICAL risk, you MUST call:
   - dispatch_emergency_alert(location, action_plan)
   NOTE: If sensors are NORMAL, you MUST NOT dispatch.

3) Every dispatch is adversarially cross-checked by Nemesis inside dispatch tool.

4) In your final response:
   - Summarize sensor findings and citizen reports separately
   - Provide a short bilingual SMS alert (English + Tagalog)
   - If no dispatch happened, explicitly say "No alert sent" and why

Keep responses factual and concise.
"""

    st.session_state.crisis_agent = client.chats.create(
        model="gemini-2.5-flash-lite",
        config=types.GenerateContentConfig(
            tools=[check_pagasa_water_level, check_social_media_reports, dispatch_emergency_alert],
            temperature=0.0,
            system_instruction=directive,
        ),
    )

if "messages" not in st.session_state:
    st.session_state.messages = []
if "debug_log" not in st.session_state:
    st.session_state.debug_log = []


# ============================================================
# 9) TOP STATUS + BADGE + KPI STRIP
# ============================================================
active_loc = st.session_state.get("active_location")
_render_status_bar(active_loc)
_render_nemesis_badge()
_render_kpi_bar()
st.divider()


# ============================================================
# 10) CHAT HISTORY
# ============================================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ============================================================
# 11) INPUT HANDLING
# ============================================================
prefill = st.session_state.pop("_demo_prompt", None)
user_prompt = st.chat_input("Enter a crisis report or check a location...")

if prefill and not user_prompt:
    user_prompt = prefill


# ============================================================
# 12) MAIN CHAT TURN
# ============================================================
if user_prompt:
    start_t = time.perf_counter()

    loc = _infer_location_from_text(user_prompt)
    st.session_state.active_location = loc if loc in MOCK_ENVIRONMENTAL_DATA else None

    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing risk data..."):
            response = st.session_state.crisis_agent.send_message(user_prompt)
            st.markdown(response.text)

    end_t = time.perf_counter()
    decision_seconds = end_t - start_t

    # Derived KPIs
    false_alarm_prevented = False
    human_escalation = False
    nemesis_decision = None
    nemesis_reason = None

    if loc in MOCK_ENVIRONMENTAL_DATA:
        sensor = MOCK_ENVIRONMENTAL_DATA[loc]
        social = _mock_social_reports(loc)
        sensor_is_critical = _is_sensor_critical(sensor)

        # False alarm prevented = social chatter exists but sensors are normal
        false_alarm_prevented = (not sensor_is_critical) and (social.get("verified_reports_count", 0) > 0)

        # Human escalation = sensors are critical but Nemesis blocks
        if sensor_is_critical:
            verdict = json.loads(nemesis_ai(loc))
            nemesis_decision = verdict.get("decision")
            nemesis_reason = verdict.get("reason")
            human_escalation = (nemesis_decision != "APPROVE")

    _set_kpi_state(
        decision_seconds=decision_seconds,
        false_alarm_prevented=false_alarm_prevented,
        human_escalation=human_escalation,
        nemesis_decision=nemesis_decision,
        nemesis_reason=nemesis_reason,
    )

    st.session_state.messages.append({"role": "assistant", "content": response.text})
    st.rerun()


# ============================================================
# 13) OPTIONAL PANELS (Explainability)
# ============================================================
active_loc = st.session_state.get("active_location")

if active_loc:
    if show_signal_truth:
        sensors = MOCK_ENVIRONMENTAL_DATA[active_loc]
        signal = _mock_social_reports(active_loc)
        critical = _is_sensor_critical(sensors)

        st.markdown("### 📊 Signal vs Truth")
        col_signal, col_truth = st.columns(2, gap="large")

        with col_signal:
            st.markdown("#### 📡 Signal (Citizen Reports)")
            st.markdown(f"**Verified reports:** `{signal['verified_reports_count']}`")
            st.markdown(f"**Confidence:** `{signal['confidence']}`")
            if signal["highlights"]:
                st.markdown("**Highlights:**")
                for h in signal["highlights"]:
                    st.markdown(f"- {h}")
            else:
                st.markdown("_No recent reports detected (mock)._")

        with col_truth:
            st.markdown("#### 🧭 Truth (Sensors)")
            st.markdown(f"**Status:** `{sensors.get('status', 'UNKNOWN')}`")
            st.markdown(
                f"**River gauge:** `{sensors.get('river_gauge_meters')}m` "
                f"(Critical: `{sensors.get('critical_threshold_meters')}m`)"
            )
            st.markdown(f"**Rainfall:** `{sensors.get('rainfall_mm_per_hr')} mm/hr`")
            st.markdown(f"**Soil saturation:** `{sensors.get('satellite_soil_saturation')}`")
            st.markdown(f"**Cloud cover:** `{sensors.get('satellite_cloud_cover')}`")

        if critical:
            st.error("🚨 Dispatch eligible (sensor truth is critical). Nemesis must approve.")
        else:
            st.success("✅ Suppressed (sensor truth is normal). No alert sent.")

    if show_open_meteo and active_loc in LOCATION_COORDS:
        st.markdown("### 🌐 Open‑Meteo Live Evidence (Best-effort)")
        bundle = _get_open_meteo_bundle(active_loc)

        if not bundle.get("ok"):
            st.warning(f"Open‑Meteo unavailable: {bundle.get('reason')}")
        else:
            weather = bundle.get("weather", {})
            flood = bundle.get("flood", {})

            c1, c2 = st.columns(2, gap="large")
            with c1:
                st.markdown("#### ☔ Weather")
                if weather.get("data_source") == "OPEN_METEO_LIVE":
                    st.markdown(f"**Current time:** `{weather['current'].get('time')}`")
                    st.markdown(f"**Temp (°C):** `{weather['current'].get('temperature_2m')}`")
                    st.markdown(f"**Precip now (mm):** `{weather['current'].get('precipitation')}`")
                    st.caption("Next hours (preview):")
                    st.json(weather.get("hourly_preview", {}))
                else:
                    st.warning(f"Weather fetch failed: {weather.get('error', 'unknown error')}")

            with c2:
                st.markdown("#### 🌊 Flood Proxy (River Discharge)")
                if flood.get("data_source") == "OPEN_METEO_LIVE":
                    st.caption("Next days (preview):")
                    st.json(flood.get("daily_preview", {}))
                else:
                    st.warning(f"Flood fetch failed: {flood.get('error', 'unknown error')}")

            st.caption(
                "Note: Flood proxy is modeled discharge. Actions remain gated by sensor truth + Nemesis cross-check."
            )

    if show_nemesis_panel:
        with st.expander("🛡️ Nemesis Output (Full JSON)"):
            st.json(json.loads(nemesis_ai(active_loc)))

    if show_raw:
        with st.expander("Raw demo datasets"):
            st.markdown("**MOCK_ENVIRONMENTAL_DATA**")
            st.json(MOCK_ENVIRONMENTAL_DATA)
            st.markdown("**ADVERSARIAL_MOCK_DATA**")
            st.json(ADVERSARIAL_MOCK_DATA)