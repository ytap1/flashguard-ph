"""
FlashGuard PH — Hackathon Demo Prototype (Streamlit + Gemini Tool Calling)

WHAT'S IN THIS VERSION (merged work):
✅ Improvements:
   - ADVERSARIAL_MOCK_DATA (independent second source)
   - Nemesis AI adversarial cross-check for every dispatch
   - Location inference from free-text inputs
   - Tool outputs standardized as JSON envelopes for reliability

✅ Latest enhancements:
   - Open-Meteo LIVE augmentation (Weather + Flood proxy) with graceful fallback
   - Projector-friendly status bar
   - Signal vs Truth side-by-side card
   - "Why no alert?" explanation (covers both sensor-block and Nemesis-block)

IMPORTANT DEMO NOTE:
- "Sensor Truth" is still driven by MOCK_ENVIRONMENTAL_DATA for deterministic, repeatable demo results.
- Open-Meteo is added as supporting evidence (best-effort, no API key). If offline, app still works.
"""

from __future__ import annotations

import json
from typing import Optional, Dict, Any, Tuple

import streamlit as st
import requests
from google import genai
from google.genai import types


# ============================================================
# 0) DEMO DATA (MOCKED) — deterministic "official sensor truth"
# ============================================================
# In production, these would come from real PAGASA river gauges/rainfall APIs and satellite feeds.
# For hackathon speed + offline reliability, we hardcode realistic demo values.
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
# 0b) ADVERSARIAL DATA (MOCKED) — "Nemesis AI" source
# ============================================================
# Think of this as an independent channel (citizen radio, drone recon, LGU bulletin).
# Nemesis AI uses this to cross-check and block dispatch when sources conflict.
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
# 0c) Coordinates (demo convenience) for Open-Meteo calls
# ============================================================
# In production, you'd geocode barangay/municipality or use official station coordinates.
LOCATION_COORDS: Dict[str, Tuple[float, float]] = {
    "Bulacan": (14.85, 120.81),
    "Marikina": (14.65, 121.10),
}


# ============================================================
# 1) Helper Logic — deterministic + demo-safe
# ============================================================
def _standard_tool_response(ok: bool, payload: Dict[str, Any], message: str) -> str:
    """All tools return a consistent JSON envelope for robust tool-calling."""
    return json.dumps({"ok": ok, "message": message, "payload": payload}, ensure_ascii=False)


def _infer_location_from_text(input_text: str) -> str:
    """
    Extract known location key from free text.
    (Improvement: makes tool calls robust to prompts like "check Bulacan please")
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
    QA Safety Gate (deterministic):
    - Only allow dispatch if status is CRITICAL_SPILL_LEVEL or gauge >= threshold.
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
    Deterministic social signal mock:
    - Marikina always has some chatter to demonstrate 'signal vs truth'.
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
# 2) Open‑Meteo Live Integration (best-effort + cached)
# ============================================================
@st.cache_data(ttl=600)
def _fetch_open_meteo_weather(lat: float, lon: float) -> Dict[str, Any]:
    """
    Open‑Meteo Forecast API: pulls current precipitation + short hourly preview.
    (No API key required for non-commercial usage.) 
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
    Open‑Meteo Flood API: modeled river discharge (GloFAS proxy) for flood risk context.
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
    Combined Open‑Meteo bundle (weather + flood proxy), best-effort.
    If offline/unavailable, returns structured errors (app still works with mock).
    """
    coords = LOCATION_COORDS.get(location)
    if not coords:
        return {"ok": False, "reason": "No coordinates mapped for this demo location."}

    lat, lon = coords
    bundle: Dict[str, Any] = {"ok": True, "location": location, "lat": lat, "lon": lon}

    try:
        bundle["weather"] = _fetch_open_meteo_weather(lat, lon)
    except Exception as e:
        bundle["weather"] = {"ok": False, "error": str(e), "data_source": "OPEN_METEO_FAILED"}

    try:
        bundle["flood"] = _fetch_open_meteo_flood(lat, lon)
    except Exception as e:
        bundle["flood"] = {"ok": False, "error": str(e), "data_source": "OPEN_METEO_FAILED"}

    return bundle


# ============================================================
# 3) Nemesis AI — adversarial cross-check (addition)
# ============================================================
def nemesis_ai(location_name: str) -> str:
    """
    "Nemesis AI" adversarial verifier:
    - checks primary sensor truth (mock official) vs adversarial alternate source (mock)
    - blocks dispatch if sources conflict or both indicate low risk
    - approves only when BOTH indicate danger
    """
    st.toast(f"🛡️ Nemesis AI: Running adversarial cross-check for {location_name}...")

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

    # Primary critical logic (style)
    primary_is_critical = False
    if primary:
        primary_is_critical = (
            float(primary.get("river_gauge_meters", 0) or 0)
            >= float(primary.get("critical_threshold_meters", 999) or 999)
            or "CRITICAL" in str(primary.get("status", "")).upper()
        )

    # Adversarial critical logic (style)
    adversarial_is_critical = False
    if adversarial:
        adversarial_is_critical = (
            str(adversarial.get("alt_flood_status", "")).upper() in {"SEVERE_FLOOD_WARNING", "CRITICAL"}
            or float(adversarial.get("estimated_inundation_meters", 0) or 0) >= 0.5
            or int(adversarial.get("rescue_requests_count", 0) or 0) >= 2
        )

    if primary_is_critical and adversarial_is_critical:
        decision = "APPROVE"
        reason = "Both primary and adversarial sources indicate flood danger."
    elif primary_is_critical != adversarial_is_critical:
        decision = "BLOCK"
        reason = "Sources conflict. Escalate for human verification before dispatch."
    else:
        decision = "BLOCK"
        reason = "Both sources indicate low risk for immediate auto-dispatch."

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
# 4) Status Bar (projector-friendly)
# ============================================================
def _update_status_context(location: Optional[str]) -> None:
    if not location:
        st.session_state.active_location = None
        st.session_state.active_decision = "READY"
        st.session_state.active_sensor = {}
        st.session_state.active_source = "—"
        return

    sensor = MOCK_ENVIRONMENTAL_DATA.get(location)
    if not sensor:
        st.session_state.active_location = location
        st.session_state.active_decision = "UNKNOWN"
        st.session_state.active_sensor = {}
        st.session_state.active_source = "UNKNOWN"
        return

    critical = _is_sensor_critical(sensor)
    st.session_state.active_location = location
    st.session_state.active_decision = "CRITICAL" if critical else "NORMAL"
    st.session_state.active_sensor = sensor
    st.session_state.active_source = sensor.get("data_source", "—")


def _render_status_bar() -> None:
    decision = st.session_state.get("active_decision", "READY")
    location = st.session_state.get("active_location", None)
    sensor = st.session_state.get("active_sensor", {}) or {}
    source = st.session_state.get("active_source", "—")

    palette = {
        "READY": {"bg": "#1565C0", "fg": "#FFFFFF", "icon": "🟦", "label": "READY"},
        "UNKNOWN": {"bg": "#455A64", "fg": "#FFFFFF", "icon": "⬛", "label": "UNKNOWN LOCATION"},
        "NORMAL": {"bg": "#1B5E20", "fg": "#FFFFFF", "icon": "🟩", "label": "NORMAL (NO ALERT)"},
        "CRITICAL": {"bg": "#B71C1C", "fg": "#FFFFFF", "icon": "🟥", "label": "CRITICAL (DISPATCH)"},
    }
    style = palette.get(decision, palette["READY"])

    if decision == "READY":
        subtitle = "Awaiting a location check… Use demo buttons (left) or type a location."
        location_text = "—"
        metrics = ""
    elif decision == "UNKNOWN":
        subtitle = "Location not in demo dataset."
        location_text = location or "—"
        metrics = ""
    else:
        location_text = location or "—"
        gauge = sensor.get("river_gauge_meters", "—")
        threshold = sensor.get("critical_threshold_meters", "—")
        rain = sensor.get("rainfall_mm_per_hr", "—")
        basin = sensor.get("river_basin", "—")
        metrics = f" | Source: {source} | Basin: {basin} | River: {gauge}m / {threshold}m | Rain: {rain} mm/hr"
        subtitle = "Decision is grounded in sensor truth; citizen reports are treated as signal."

    st.markdown(
        f"""
<div style="
    padding: 0.85rem 1rem;
    border-radius: 0.6rem;
    background: {style['bg']};
    color: {style['fg']};
    font-weight: 800;
    letter-spacing: 0.2px;
    margin-bottom: 0.75rem;
">
  <span style="font-size: 1.05rem;">{style['icon']} {style['label']}</span>
  <span style="font-weight: 600;"> — {location_text}</span>
  <span style="font-weight: 400;">{metrics}</span>
  <div style="margin-top: 0.25rem; font-weight: 500; opacity: 0.95;">
    {subtitle}
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


# ============================================================
# 5) GEMINI TOOL FUNCTIONS (callable by the model)
# ============================================================
def check_pagasa_water_level(location_name: str) -> str:
    """
    Tool: Primary "Truth" (sensor truth) + augmentation.
    - Deterministic sensor truth: MOCK_ENVIRONMENTAL_DATA
    - Added augmentation: Open-Meteo weather + flood proxy (best-effort)
    - Added transparency: includes adversarial record for context
    """
    st.toast(f"📡 System: Checking sensor feeds for {location_name}...")

    location_key = _infer_location_from_text(location_name)

    sensor_truth = MOCK_ENVIRONMENTAL_DATA.get(location_key)
    adversarial = ADVERSARIAL_MOCK_DATA.get(location_key)

    # Best-effort live evidence (does NOT override sensor gate)
    st.toast(f"🌐 System: Pulling Open‑Meteo live signals for {location_key}...")
    open_meteo_bundle = _get_open_meteo_bundle(location_key)

    if sensor_truth:
        payload = dict(sensor_truth)
        payload["adversarial"] = adversarial or {"note": "No adversarial record for this location."}
        payload["open_meteo"] = open_meteo_bundle
        payload["data_source"] = "MOCK_OFFICIAL_SENSOR + OPEN_METEO(best-effort) + ADVERSARIAL(mock)"
        return _standard_tool_response(True, payload, f"Loaded sensor truth for {location_key} (mock) + augmentation.")

    # If not in mock dataset, return whatever live/adversarial we have
    payload = {
        "location": location_key,
        "sensor_truth": None,
        "adversarial": adversarial,
        "open_meteo": open_meteo_bundle,
        "data_source": "OPEN_METEO(best-effort) + ADVERSARIAL(mock)",
    }
    ok = bool(adversarial) or bool(open_meteo_bundle.get("ok"))
    return _standard_tool_response(ok, payload, "No mock sensor truth found; returning available sources.")


def check_social_media_reports(area_name: str) -> str:
    """
    Tool: Citizen signal reports (mock).
    """
    st.toast(f"📱 System: Scanning citizen reports for {area_name}...")
    reports = _mock_social_reports(area_name)
    if reports["verified_reports_count"] > 0:
        return _standard_tool_response(True, reports, "Citizen reports detected (mock).")
    return _standard_tool_response(True, reports, "No recent citizen reports found (mock).")


def dispatch_emergency_alert(area_name: str, action_plan: str) -> str:
    """
    Tool: Dispatch (mocked).
    Safety gates:
    1) Sensor gate (must be critical)
    2) Nemesis AI gate (must be APPROVE; blocks if conflict/low-risk)
    """
    st.toast(f"🚨 ACTION: Validating before dispatch to {area_name}...")

    location_key = _infer_location_from_text(area_name)
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

    # Gate 2: adversarial cross-check (Nemesis AI)
    adversarial_verdict = json.loads(nemesis_ai(location_key))
    if adversarial_verdict.get("decision") != "APPROVE":
        st.toast(f"🛑 ALERT BLOCKED: Nemesis AI flagged {location_key} for manual verification.")
        return _standard_tool_response(
            ok=False,
            payload={
                "area": location_key,
                "reason": adversarial_verdict.get("reason", "Nemesis AI blocked dispatch."),
                "nemesis_decision": adversarial_verdict.get("decision", "BLOCK"),
            },
            message="SAFETY BLOCK: Nemesis AI blocked auto-dispatch.",
        )

    # Dispatch allowed
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
# 6) STREAMLIT UI SETUP
# ============================================================
st.set_page_config(page_title="FlashGuard PH", page_icon="⛈️", layout="centered")
st.title("⛈️ FlashGuard PH")
st.subheader("Autonomous Crisis Management System (Hackathon Demo)")
st.caption("Hybrid demo: deterministic sensor truth + adversarial cross-check + live Open‑Meteo evidence (best‑effort).")

# Init status state
if "active_decision" not in st.session_state:
    st.session_state.active_decision = "READY"
if "active_location" not in st.session_state:
    st.session_state.active_location = None
if "active_sensor" not in st.session_state:
    st.session_state.active_sensor = {}
if "active_source" not in st.session_state:
    st.session_state.active_source = "—"

_render_status_bar()


# ============================================================
# 7) SIDEBAR — demo controls + judge toggles
# ============================================================
with st.sidebar:
    st.header("🎬 Demo Controls")
    st.caption("One-click prompts for the two scripted scenarios.")

    if st.button("Scenario A: Bulacan (Critical → Dispatch)"):
        st.session_state["_demo_prompt"] = "Check flood status in Bulacan."
        _update_status_context("Bulacan")

    if st.button("Scenario B: Marikina (Normal → Suppress)"):
        st.session_state["_demo_prompt"] = "Check flood status in Marikina."
        _update_status_context("Marikina")

    if st.button("Reset Demo"):
        st.session_state.pop("messages", None)
        st.session_state.pop("debug_log", None)
        _update_status_context(None)
        st.rerun()

    st.divider()
    st.subheader("🧪 Debug Views")
    show_signal_truth = st.checkbox("Show 'Signal vs Truth' card", value=True)
    show_why_no_alert = st.checkbox("Show 'Why no alert?' explanation", value=True)
    show_open_meteo = st.checkbox("Show Open‑Meteo live evidence panel", value=True)
    show_nemesis_panel = st.checkbox("Show Nemesis AI verdict panel", value=True)
    show_raw = st.checkbox("Show raw mock JSON (demo only)", value=False)
    show_debug = st.checkbox("Show debug timeline", value=False)


# ============================================================
# 8) INITIALIZE GEMINI CHAT AGENT (ONCE PER SESSION)
# ============================================================
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    st.error("Missing GEMINI_API_KEY in Streamlit secrets. Add it to .streamlit/secrets.toml")
    st.stop()

if "crisis_agent" not in st.session_state:
    client = genai.Client(api_key=api_key)

    directive = """
You are FlashGuard PH, an autonomous disaster response AI for a hackathon demo.

NON-NEGOTIABLE PROTOCOL:
1) When a user asks about a location or flood status, you MUST call:
   - check_pagasa_water_level(location)
   - check_social_media_reports(location)

2) Only if the sensor truth indicates CRITICAL risk, you MUST call:
   - dispatch_emergency_alert(location, action_plan)
   NOTE: If sensors are NORMAL, you MUST NOT dispatch.

3) Every dispatch is adversarially cross-checked by Nemesis AI inside the dispatch tool.

4) In your final response:
   - Summarize sensor findings and citizen reports separately
   - Provide a short bilingual SMS alert (English + Tagalog)
   - If no dispatch happened, explicitly say "No alert sent" and why

Keep responses factual and concise. No hallucinations.
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
# 9) CHAT UI RENDER
# ============================================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ============================================================
# 10) INPUT HANDLING (manual input OR sidebar demo buttons)
# ============================================================
prefill = st.session_state.pop("_demo_prompt", None)
user_prompt = st.chat_input("Enter a crisis report or check a location...")

if prefill and not user_prompt:
    user_prompt = prefill


# ============================================================
# 11) MAIN CHAT TURN
# ============================================================
if user_prompt:
    # Store & show user prompt
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Update status bar based on inferred location
    loc = _infer_location_from_text(user_prompt)
    if loc in MOCK_ENVIRONMENTAL_DATA:
        _update_status_context(loc)
    else:
        _update_status_context(None)
    _render_status_bar()

    # Ask agent (tool-calling)
    with st.chat_message("assistant"):
        with st.spinner("Analyzing risk data..."):
            response = st.session_state.crisis_agent.send_message(user_prompt)
            st.markdown(response.text)

        # --- Signal vs Truth card (side-by-side) ---
        if show_signal_truth and loc in MOCK_ENVIRONMENTAL_DATA:
            sensors = MOCK_ENVIRONMENTAL_DATA[loc]
            signal = _mock_social_reports(loc)
            critical = _is_sensor_critical(sensors)

            st.markdown("### 📊 Signal vs Truth (How FlashGuard decides)")
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
                    st.markdown("_No recent citizen reports detected (mock)._")

            with col_truth:
                st.markdown("#### 🧭 Truth (Sensors — demo-stable)")
                st.markdown(f"**Status:** `{sensors.get('status', 'UNKNOWN')}`")
                st.markdown(
                    f"**River gauge:** `{sensors.get('river_gauge_meters')}m` "
                    f"(Critical: `{sensors.get('critical_threshold_meters')}m`)"
                )
                st.markdown(f"**Rainfall:** `{sensors.get('rainfall_mm_per_hr')} mm/hr`")
                st.markdown(f"**Soil saturation:** `{sensors.get('satellite_soil_saturation')}`")
                st.markdown(f"**Cloud cover:** `{sensors.get('satellite_cloud_cover')}`")

            if critical:
                st.error("🚨 DISPATCH ELIGIBLE: Sensor truth is critical (Nemesis AI still must approve).")
            else:
                st.success("✅ SUPPRESSED: Sensor truth is normal → no evacuation alert.")

        # --- Nemesis verdict panel (optional) ---
        if show_nemesis_panel and loc in MOCK_ENVIRONMENTAL_DATA:
            verdict = json.loads(nemesis_ai(loc))
            with st.expander("🛡️ Nemesis AI Verdict (Adversarial Cross-check)", expanded=False):
                st.json(verdict)

        # --- Why no alert explanation ---
        if show_why_no_alert and loc in MOCK_ENVIRONMENTAL_DATA:
            sensor = MOCK_ENVIRONMENTAL_DATA[loc]
            critical = _is_sensor_critical(sensor)

            if not critical:
                # Blocked at sensor gate
                gauge = sensor.get("river_gauge_meters")
                threshold = sensor.get("critical_threshold_meters")
                status = sensor.get("status", "UNKNOWN")
                st.info(
                    f"""
**Why no alert was sent (Sensor QA gate)**

- **Sensor status:** `{status}`
- **River gauge vs threshold:** `{gauge}m` vs `{threshold}m` → **below critical**
- **Rule:** Auto-dispatch only when sensors are **critical**

✅ Result: **No evacuation alert sent** — prevents false alarms.
""".strip()
                )
            else:
                # Sensor critical; may still be blocked by Nemesis
                verdict = json.loads(nemesis_ai(loc))
                if verdict.get("decision") != "APPROVE":
                    st.warning(
                        f"""
**Why dispatch was blocked (Nemesis AI safeguard)**

Sensor truth is **critical**, but **Nemesis AI blocked auto-dispatch**:
- **Reason:** {verdict.get('reason', 'Sources conflict or low risk from adversarial channel.')}
- **Decision:** `{verdict.get('decision')}`

✅ Result: **No auto-dispatch** — escalated for human verification.
""".strip()
                    )

        # --- Open‑Meteo live evidence panel (best-effort) ---
        if show_open_meteo and loc in LOCATION_COORDS:
            st.markdown("### 🌐 Live Evidence (Open‑Meteo, best‑effort)")
            bundle = _get_open_meteo_bundle(loc)

            if not bundle.get("ok"):
                st.warning(f"Open‑Meteo live fetch unavailable: {bundle.get('reason')}")
            else:
                weather = bundle.get("weather", {})
                flood = bundle.get("flood", {})

                c1, c2 = st.columns(2, gap="large")
                with c1:
                    st.markdown("#### ☔ Weather (Forecast API)")
                    if weather.get("data_source") == "OPEN_METEO_LIVE":
                        st.markdown(f"**Current time:** `{weather['current'].get('time')}`")
                        st.markdown(f"**Temp (°C):** `{weather['current'].get('temperature_2m')}`")
                        st.markdown(f"**Precip now (mm):** `{weather['current'].get('precipitation')}`")
                        st.caption("Next hours (preview):")
                        st.json(weather.get("hourly_preview", {}))
                    else:
                        st.warning(f"Weather fetch failed: {weather.get('error', 'unknown error')}")

                with c2:
                    st.markdown("#### 🌊 Flood Proxy (River Discharge, GloFAS model)")
                    if flood.get("data_source") == "OPEN_METEO_LIVE":
                        st.caption("Next days (preview):")
                        st.json(flood.get("daily_preview", {}))
                    else:
                        st.warning(f"Flood fetch failed: {flood.get('error', 'unknown error')}")

                st.caption(
                    "Note: Flood API values are modeled discharge estimates (not a local gauge). "
                    "Dispatch remains grounded in verified sensor truth + Nemesis cross-check."
                )

        # --- Debug / raw ---
        if show_debug:
            st.session_state.debug_log.append({"prompt": user_prompt, "response": response.text})
            with st.expander("Debug log"):
                st.json(st.session_state.debug_log)

        if show_raw:
            with st.expander("Raw mock datasets (demo only)"):
                st.markdown("**MOCK_ENVIRONMENTAL_DATA**")
                st.json(MOCK_ENVIRONMENTAL_DATA)
                st.markdown("**ADVERSARIAL_MOCK_DATA**")
                st.json(ADVERSARIAL_MOCK_DATA)

    # Save assistant response (no trailing comma bug)
    st.session_state.messages.append({"role": "assistant", "content": response.text})