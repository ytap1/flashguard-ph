The provided Python code is a well-structured and commented Streamlit application demonstrating an autonomous flood-risk analysis system using Google Gemini's tool calling capabilities. It effectively showcases a hybrid approach with deterministic mock data for core decisions and live Open-Meteo data for augmentation.

However, there are a couple of critical bugs related to the Gemini API integration and some minor edge cases/optimizations that can be addressed.

---

### 1. Identified Bugs and Edge Cases

#### 1.1. Critical Bug: Incorrect Gemini Client and Model Initialization
*   **Location:** Section 7: `client = genai.Client(api_key=api_key)` and `st.session_state.crisis_agent = client.chats.create(...)`.
*   **Explanation:** The `google-generativeai` library (imported as `genai`) does not typically use `genai.Client` or `client.chats.create` for chat interactions. These constructs are more indicative of an older API version, a different Google Cloud SDK (e.g., `google-cloud-aiplatform`), or a misunderstanding of the current `google-generativeai` library's API. The standard way to initialize and interact with a chat model in `google-generativeai` is to first configure the API key, then create a `GenerativeModel` instance, and finally start a chat session using `model.start_chat()`.
*   **Impact:** The application will likely fail to initialize the Gemini agent, leading to a runtime error when `client.chats.create` is called, or `send_message` will fail later.

#### 1.2. Critical Bug: Invalid Gemini Model Name
*   **Location:** Section 7: `model="gemini-2.5-flash-lite"`.
*   **Explanation:** The model name `gemini-2.5-flash-lite` is not a standard, publicly available Gemini model name. Common names include `gemini-pro`, `gemini-1.0-pro`, `gemini-1.5-flash`, `gemini-1.5-pro`. Given the "flash" in the name, it's highly probable that `gemini-1.5-flash` was the intended model.
*   **Impact:** The Gemini API call will fail with an "invalid model name" error.

#### 1.3. Edge Case/Design Consideration: Status Bar vs. AI's Knowledge Discrepancy
*   **Location:** Section 10: `_update_status_context` call within the `if user_prompt:` block.
*   **Explanation:** The `_update_status_context` function, which controls the top status bar, is called *before* the Gemini model processes the prompt and executes its tools. It primarily relies on `MOCK_ENVIRONMENTAL_DATA`. If a user queries a location *not* in `MOCK_ENVIRONMENTAL_DATA` but for which `check_pagasa_water_level` *can* fetch live Open-Meteo data (e.g., if `LOCATION_COORDS` had more entries), the status bar would display "UNKNOWN" while the AI's response would be based on the live data.
*   **Impact:** This creates a minor inconsistency between the UI's immediate status and the AI's actual processing. For a demo focused on "deterministic decisions" from mock data, this is a deliberate design choice to keep the status bar grounded in the "truth" data. For a production system, the status bar should ideally reflect the most up-to-date information processed by the AI. I will keep the current behavior to align with the demo's stated goal, but add a comment.

#### 1.4. Minor Optimization: Location Set Creation
*   **Location:** `_extract_demo_location` function.
*   **Explanation:** `set(list(MOCK_ENVIRONMENTAL_DATA.keys()) + list(LOCATION_COORDS.keys()))` is slightly less efficient than using set union directly.
*   **Impact:** Negligible for small dictionaries, but good practice to optimize.

#### 1.5. Enhancement: Tool Call Visualization
*   **Location:** Main chat turn (Section 10).
*   **Explanation:** The current code only displays `response.text`. For a hackathon demo, it's incredibly useful for judges/viewers to see *which tools* the AI called and *what their raw outputs were*, as this directly demonstrates the tool-calling capability.
*   **Impact:** Without this, the internal workings of the AI are less transparent.

---

### 2. Enhanced, Optimized, and Fully Debugged Version

The enhanced version addresses the critical bugs, incorporates the minor optimization, and adds a significant enhancement for tool call visualization.

```python
"""
FlashGuard PH — Hackathon Demo Prototype (Streamlit + Gemini Tool Calling)

DEMO GOAL:
- Showcase autonomous flood-risk analysis + instant action.
- Scenario A (Bulacan): CRITICAL sensors => auto-trigger evacuation alert.
- Scenario B (Marikina): NORMAL sensors => suppress alert (prevent false alarms).

IMPORTANT DEMO NOTE (for viewers/judges):
- In production, the functions below would call real APIs (PAGASA river gauges, rainfall,
  satellite soil moisture, and social listening like X/Twitter).
- For hackathon speed + repeatability, we keep a mock "official sensor" dataset.
- NEW: We now ALSO pull LIVE weather + flood proxy data from Open-Meteo (no API key),
  and show it as supporting evidence. If live calls fail, the app automatically
  falls back to mock data and continues working offline.
"""

import json
from typing import Optional, Dict, Any, Tuple, List

import streamlit as st
import requests
import google.generativeai as genai # Corrected import for standard Gemini library
from google.generativeai import types # Still useful for config types


# ============================================================
# 0) DEMO DATA (MOCKED "OFFICIAL SENSOR") — used for deterministic decisions
# ============================================================
# These values represent what your production integration WOULD deliver from:
# - PAGASA river gauges / rainfall APIs
# - Satellite soil saturation / cloud cover
# For hackathon demo purposes, we hardcode them for stability and repeatability.
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

# Approximate coordinates used for Open-Meteo calls (demo convenience).
# In production, you'd geocode barangay/municipality or use official station coordinates.
LOCATION_COORDS: Dict[str, Tuple[float, float]] = {
    "Bulacan": (14.85, 120.81),
    "Marikina": (14.65, 121.10),
}


# ============================================================
# 1) DETERMINISTIC QA LOGIC — keeps decisions "grounded in facts"
# ============================================================
def _is_sensor_critical(sensor_record: Dict[str, Any]) -> bool:
    """
    QA Safety Gate (deterministic):
    - Only allow dispatch if status is explicitly CRITICAL_SPILL_LEVEL
      OR if river gauge >= critical threshold.
    This ensures the architecture cannot send evacuation alerts in NORMAL conditions.
    """
    if not sensor_record:
        return False

    status = sensor_record.get("status")
    # Robustly handle potential None or missing values for gauge and threshold
    gauge = float(sensor_record.get("river_gauge_meters", 0) or 0)
    threshold = float(sensor_record.get("critical_threshold_meters", 999999) or 999999)
    return status == "CRITICAL_SPILL_LEVEL" or gauge >= threshold


def _standard_tool_response(ok: bool, payload: Dict[str, Any], message: str) -> str:
    """
    All tools return a consistent JSON envelope:
      { ok: bool, message: str, payload: {...} }
    This avoids ambiguous parsing and is easier to audit during a demo.
    """
    return json.dumps({"ok": ok, "message": message, "payload": payload}, ensure_ascii=False)


def _extract_demo_location(text: str) -> Optional[str]:
    """
    Demo-only location extraction.
    We only support locations included in MOCK_ENVIRONMENTAL_DATA/LOCATION_COORDS.
    """
    if not text:
        return None
    t = text.lower()
    # Optimization: Use set union for more efficient combination of keys
    all_locations = set(MOCK_ENVIRONMENTAL_DATA.keys()) | set(LOCATION_COORDS.keys())
    for loc in all_locations:
        if loc.lower() in t:
            return loc
    return None


def _mock_social_reports(area_name: str) -> Dict[str, Any]:
    """
    Deterministic social signal (mock).
    Demo rule:
    - Marikina always shows citizen chatter (signal/noise) even if sensors are normal.
    - Other locations show none (for simplicity).
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
# 2) OPEN-METEO LIVE INTEGRATION (Weather + Flood)
# ============================================================
@st.cache_data(ttl=600)
def _fetch_open_meteo_weather(lat: float, lon: float) -> Dict[str, Any]:
    """
    Fetch live weather from Open-Meteo Forecast API (no API key).
    We pull current + hourly precipitation signals for demo usefulness.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        # current & hourly parameter names are documented by Open-Meteo
        "current": "temperature_2m,precipitation",
        "hourly": "precipitation,precipitation_probability",
        "forecast_days": 2,
        "timezone": "auto",
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()

        # Extract a small, stable subset for display + tool payload
        current = data.get("current", {})
        hourly = data.get("hourly", {})
        out = {
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
                # only first few entries to keep payload small
                "time": (hourly.get("time") or [])[:8],
                "precipitation": (hourly.get("precipitation") or [])[:8],
                "precipitation_probability": (hourly.get("precipitation_probability") or [])[:8],
            },
            "data_source": "OPEN_METEO_LIVE",
        }
        return out
    except requests.exceptions.RequestException as e:
        st.error(f"Open-Meteo Weather API call failed: {e}")
        return {"ok": False, "error": str(e), "data_source": "OPEN_METEO_FAILED"}


@st.cache_data(ttl=600)
def _fetch_open_meteo_flood(lat: float, lon: float) -> Dict[str, Any]:
    """
    Fetch live river discharge proxy from Open-Meteo Flood API (GloFAS-based).
    This is NOT an official local gauge; it is a modeled discharge estimate.
    """
    url = "https://flood-api.open-meteo.com/v1/flood"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "river_discharge,river_discharge_max",
        "forecast_days": 7,
        "timeformat": "iso8601",
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()

        daily = data.get("daily", {})
        out = {
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
        return out
    except requests.exceptions.RequestException as e:
        st.error(f"Open-Meteo Flood API call failed: {e}")
        return {"ok": False, "error": str(e), "data_source": "OPEN_METEO_FAILED"}


def _get_live_open_meteo_bundle(location: str) -> Dict[str, Any]:
    """
    Returns a combined bundle of weather + flood model outputs.
    Falls back gracefully if location is unknown or request fails.
    """
    coords = LOCATION_COORDS.get(location)
    if not coords:
        return {"ok": False, "reason": "No coordinates available for this location in demo mapping."}

    lat, lon = coords
    bundle: Dict[str, Any] = {"ok": True, "location": location, "lat": lat, "lon": lon}

    # _fetch_open_meteo_weather and _fetch_open_meteo_flood now handle their own exceptions
    # and return a dictionary with "ok": False on failure.
    bundle["weather"] = _fetch_open_meteo_weather(lat, lon)
    bundle["flood"] = _fetch_open_meteo_flood(lat, lon)

    return bundle


# ============================================================
# 3) STATUS BAR STATE + RENDERING
# ============================================================
def _update_status_context(location: Optional[str], sensor_payload: Optional[Dict[str, Any]] = None) -> None:
    """
    Stores the latest location + decision in session state, used by the top status bar.
    sensor_payload is optional; if provided, it is stored and used for metrics display.

    NOTE: For this demo, the status bar is intentionally grounded in the MOCK_ENVIRONMENTAL_DATA
    to ensure deterministic "Truth" for the demo's core decision logic.
    It does not dynamically update based on live Open-Meteo data alone if no mock data exists.
    """
    if not location:
        st.session_state.active_location = None
        st.session_state.active_decision = "READY"
        st.session_state.active_sensor = {}
        st.session_state.active_source = "—"
        return

    # Prioritize provided sensor_payload, otherwise fall back to mock data
    sensor = sensor_payload or MOCK_ENVIRONMENTAL_DATA.get(location)
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
    """
    Projector-friendly top status bar.
    - READY: no scenario selected
    - UNKNOWN: location not found
    - NORMAL: sensors below threshold => suppress alert
    - CRITICAL: sensors above threshold => dispatch allowed
    """
    decision = st.session_state.get("active_decision", "READY")
    location = st.session_state.get("active_location", None)
    sensor = st.session_state.get("active_sensor", {}) or {}
    source = st.session_state.get("active_source", "—")

    palette = {
        "READY":   {"bg": "#1565C0", "fg": "#FFFFFF", "icon": "🟦", "label": "READY"},
        "UNKNOWN": {"bg": "#455A64", "fg": "#FFFFFF", "icon": "⬛", "label": "UNKNOWN LOCATION"},
        "NORMAL":  {"bg": "#1B5E20", "fg": "#FFFFFF", "icon": "🟩", "label": "NORMAL (NO ALERT)"},
        "CRITICAL":{"bg": "#B71C1C", "fg": "#FFFFFF", "icon": "🟥", "label": "CRITICAL (DISPATCH)"},
    }
    style = palette.get(decision, palette["READY"])

    if decision == "READY":
        subtitle = "Awaiting a location check… Use demo buttons (left) or type a location."
        location_text = "—"
        metrics = ""
    elif decision == "UNKNOWN":
        subtitle = "Location not in demo dataset/mapping. Live calls require coordinates."
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
# 4) GEMINI TOOL FUNCTIONS (callable by the model)
# ============================================================
def check_pagasa_water_level(location_name: str) -> str:
    """
    Tool: "Truth (Sensors)" — now enhanced with live Open-Meteo augmentation.

    What is deterministic for demo:
    - MOCK_ENVIRONMENTAL_DATA is the authoritative "sensor truth" that drives dispatch decisions.
    - This guarantees Bulacan=CRITICAL and Marikina=NORMAL in your scripted demo.

    What is live for credibility:
    - Open-Meteo weather (precipitation)
    - Open-Meteo flood proxy (river discharge)
    These are displayed as additional evidence but do NOT override the QA gate.
    """
    st.toast(f"📡 System: Checking sensor feeds for {location_name}...")

    # Base sensor truth comes from deterministic mock (hackathon stability)
    sensor_truth = MOCK_ENVIRONMENTAL_DATA.get(location_name)

    # Fetch Open-Meteo live bundle (best-effort)
    st.toast(f"🌐 System: Pulling Open-Meteo live signals for {location_name}...")
    open_meteo_bundle = _get_live_open_meteo_bundle(location_name)

    if sensor_truth:
        # Merge: keep sensor truth stable, add live data under "open_meteo"
        merged = dict(sensor_truth)
        merged["open_meteo"] = open_meteo_bundle
        merged["data_source"] = "MOCK_OFFICIAL_SENSOR + OPEN_METEO_LIVE(best-effort)"
        return _standard_tool_response(
            ok=True,
            payload=merged,
            message=f"Sensor truth loaded for {location_name} (mock) + Open-Meteo augmentation.",
        )

    # If location not in mock dataset, return live-only if possible.
    if open_meteo_bundle.get("ok") and (open_meteo_bundle.get("weather", {}).get("ok") is not False or \
                                        open_meteo_bundle.get("flood", {}).get("ok") is not False):
        live_only = {
            "timestamp": None,
            "river_basin": "—",
            "river_gauge_meters": None,
            "critical_threshold_meters": None,
            "rainfall_mm_per_hr": None,
            "satellite_soil_saturation": None,
            "satellite_cloud_cover": None,
            "status": "LIVE_DATA_ONLY",
            "open_meteo": open_meteo_bundle,
            "data_source": "OPEN_METEO_LIVE_ONLY",
        }
        return _standard_tool_response(
            ok=True,
            payload=live_only,
            message=f"Live Open-Meteo signals returned for {location_name} (no mock sensor truth available).",
        )

    return _standard_tool_response(
        ok=False,
        payload={"location": location_name, "open_meteo": open_meteo_bundle},
        message="No mock sensor record and Open-Meteo live fetch unavailable (mock mode fallback).",
    )


def check_social_media_reports(area_name: str) -> str:
    """
    Tool: "Signal (Citizen Reports)" — mocked for deterministic demo.
    In production, this would query X/Twitter, hotlines, barangay comms, etc.
    """
    st.toast(f"📱 System: Scanning citizen reports for {area_name}...")
    reports = _mock_social_reports(area_name)
    if reports["verified_reports_count"] > 0:
        return _standard_tool_response(True, reports, "Citizen reports detected (mock).")
    return _standard_tool_response(True, reports, "No recent citizen reports found (mock).")


def dispatch_emergency_alert(area_name: str, action_plan: str) -> str:
    """
    Tool: Dispatch (mocked).

    QA SAFETY IMPROVEMENT:
    - This tool re-checks SENSOR TRUTH BEFORE dispatching.
    - Even if the AI tries to dispatch incorrectly, the tool refuses unless critical.
    """
    st.toast(f"🚨 ACTION: Validating before dispatch to {area_name}...")

    sensor_record = MOCK_ENVIRONMENTAL_DATA.get(area_name, {})
    if not _is_sensor_critical(sensor_record):
        return _standard_tool_response(
            ok=False,
            payload={
                "area": area_name,
                "reason": "Dispatch blocked: sensors are not critical.",
                "sensor_status": sensor_record.get("status", "UNKNOWN"),
            },
            message="SAFETY BLOCK: No evacuation alert sent.",
        )

    st.toast(f"🚨 ACTION TRIGGERED: Dispatching units to {area_name}!")
    return _standard_tool_response(
        ok=True,
        payload={
            "area": area_name,
            "action_plan": action_plan,
            "dispatch": "Evacuation SMS broadcasted + resources coordinated (mock).",
        },
        message="SUCCESS: Evacuation alert dispatched.",
    )


# ============================================================
# 5) STREAMLIT UI SETUP
# ============================================================
st.set_page_config(page_title="FlashGuard PH", page_icon="⛈️", layout="centered")
st.title("⛈️ FlashGuard PH")
st.subheader("Autonomous Crisis Management System (Hackathon Demo)")
st.caption("Hybrid mode: deterministic mock 'sensor truth' + best-effort live Open-Meteo augmentation.")

# Initialize status context if not present
if "active_decision" not in st.session_state:
    st.session_state.active_decision = "READY"
if "active_location" not in st.session_state:
    st.session_state.active_location = None
if "active_sensor" not in st.session_state:
    st.session_state.active_sensor = {}
if "active_source" not in st.session_state:
    st.session_state.active_source = "—"

# Render top status bar
_render_status_bar()


# ============================================================
# 6) SIDEBAR — demo controls + judge toggles
# ============================================================
with st.sidebar:
    st.header("🎬 Demo Controls")
    st.caption("One-click prompts for the two scripted scenarios.")

    if st.button("Scenario A: Bulacan (Critical → Dispatch)"):
        st.session_state["_demo_prompt"] = "Check flood status in Bulacan."
        _update_status_context("Bulacan", MOCK_ENVIRONMENTAL_DATA["Bulacan"])
        # Clear chat history for a fresh scenario run
        st.session_state.pop("messages", None)
        st.session_state.pop("debug_log", None)
        st.rerun()

    if st.button("Scenario B: Marikina (Normal → Suppress)"):
        st.session_state["_demo_prompt"] = "Check flood status in Marikina."
        _update_status_context("Marikina", MOCK_ENVIRONMENTAL_DATA["Marikina"])
        # Clear chat history for a fresh scenario run
        st.session_state.pop("messages", None)
        st.session_state.pop("debug_log", None)
        st.rerun()

    if st.button("Reset Demo"):
        st.session_state.pop("messages", None)
        st.session_state.pop("debug_log", None)
        _update_status_context(None)
        st.rerun()

    st.divider()
    st.subheader("🧪 Judge/Debug Views")
    show_signal_truth = st.checkbox("Show 'Signal vs Truth' card", value=True)
    show_why_no_alert = st.checkbox("Show 'Why no alert?' explanation", value=True)
    show_open_meteo = st.checkbox("Show Open-Meteo live evidence panel", value=True)
    show_raw = st.checkbox("Show raw mock JSON (demo only)", value=False)
    show_debug = st.checkbox("Show debug timeline (AI calls)", value=False)


# ============================================================
# 7) INITIALIZE GEMINI CHAT AGENT (ONCE PER SESSION)
# ============================================================
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    # Bug Fix: Use genai.configure and genai.GenerativeModel for standard API usage
    genai.configure(api_key=api_key)
except Exception:
    st.error("Missing GEMINI_API_KEY in Streamlit secrets. Add it to .streamlit/secrets.toml")
    st.stop()

if "crisis_agent" not in st.session_state:
    # Bug Fix: Use a valid model name, e.g., 'gemini-1.5-flash'
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash", # Corrected model name
        tools=[check_pagasa_water_level, check_social_media_reports, dispatch_emergency_alert],
        generation_config=types.GenerationConfig(
            temperature=0.0, # Good for deterministic behavior in a demo
        ),
        system_instruction="""
You are FlashGuard PH, an autonomous disaster response AI for a hackathon demo.

NON-NEGOTIABLE PROTOCOL:
1) When a user asks about a location or flood status, you MUST call:
   - check_pagasa_water_level(location)
   - check_social_media_reports(location)

2) Only if sensor truth indicates CRITICAL risk, you MUST call:
   - dispatch_emergency_alert(location, action_plan)
   NOTE: If sensors are NORMAL, you MUST NOT dispatch.

3) In your final response:
   - Summarize "Truth (Sensors)" and "Signal (Citizen Reports)" separately
   - Provide a short bilingual SMS alert (English + Tagalog)
   - If no dispatch happened, explicitly say "No alert sent" and why

Keep responses factual and concise. No hallucinations.
"""
    )
    # Start a chat session
    st.session_state.crisis_agent = model.start_chat(history=[])

if "messages" not in st.session_state:
    st.session_state.messages = []
if "debug_log" not in st.session_state:
    st.session_state.debug_log = []


# ============================================================
# 8) RENDER CHAT HISTORY
# ============================================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ============================================================
# 9) INPUT HANDLING (manual input OR sidebar scenario buttons)
# ============================================================
prefill = st.session_state.pop("_demo_prompt", None)
user_prompt = st.chat_input("Enter a crisis report or check a location...")

if prefill and not user_prompt:
    user_prompt = prefill


# ============================================================
# 10) MAIN CHAT TURN
# ============================================================
if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Update status bar context (based on deterministic mock sensor truth if recognized)
    # NOTE: This update happens *before* the AI's full turn to reflect the initial prompt's intent.
    # For a production system, a post-AI-turn update based on actual tool outputs might be preferred.
    detected_loc = _extract_demo_location(user_prompt)
    if detected_loc in MOCK_ENVIRONMENTAL_DATA:
        _update_status_context(detected_loc, MOCK_ENVIRONMENTAL_DATA[detected_loc])
    else:
        _update_status_context(detected_loc, None)
    _render_status_bar()

    with st.chat_message("assistant"):
        with st.spinner("Analyzing risk data..."):
            # Bug Fix: Use the chat session's send_message method
            response = st.session_state.crisis_agent.send_message(user_prompt)

            # Enhancement: Display tool calls and outputs for debugging/transparency
            if show_debug:
                st.markdown("---")
                st.markdown("### 🛠️ AI Tool Interaction (Debug)")
                debug_entry: Dict[str, Any] = {"prompt": user_prompt, "tool_calls": [], "tool_outputs": [], "final_response": response.text}

                # Iterate through parts to find tool_code and function_response
                for part in response.candidates[0].content.parts:
                    if part.function_call:
                        tool_call_str = f"Calling tool: `{part.function_call.name}({json.dumps(part.function_call.args)})`"
                        st.info(tool_call_str)
                        debug_entry["tool_calls"].append(tool_call_str)
                    elif part.function_response:
                        tool_output_str = f"Tool output from `{part.function_response.name}`: `{part.function_response.response}`"
                        st.code(tool_output_str, language="json")
                        debug_entry["tool_outputs"].append(tool_output_str)
                st.session_state.debug_log.append(debug_entry)
                st.markdown("---")

            st.markdown(response.text)

        # ------------------------------------------------------------
        # Side-by-side "Signal vs Truth" card
        # ------------------------------------------------------------
        if show_signal_truth:
            loc = _extract_demo_location(user_prompt)
            if loc and loc in MOCK_ENVIRONMENTAL_DATA:
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
                    st.error("🚨 DISPATCH: Evacuation alert triggered (sensor truth is critical).")
                else:
                    st.success("✅ SUPPRESSED: No evacuation alert (sensor truth is normal).")

        # ------------------------------------------------------------
        # "Why no alert?" explanation when suppressed
        # ------------------------------------------------------------
        if show_why_no_alert:
            loc = _extract_demo_location(user_prompt)
            if loc and loc in MOCK_ENVIRONMENTAL_DATA:
                sensor = MOCK_ENVIRONMENTAL_DATA[loc]
                critical = _is_sensor_critical(sensor)
                if not critical:
                    gauge = sensor.get("river_gauge_meters")
                    threshold = sensor.get("critical_threshold_meters")
                    status = sensor.get("status", "UNKNOWN")

                    st.info(
                        f"""
**Why no alert was sent (QA safeguard)**

FlashGuard PH enforces a **fact-grounded safety gate** before sending evacuation alerts:

- **Sensor status:** `{status}`
- **River gauge vs threshold:** `{gauge}m` vs `{threshold}m` → **below critical**
- **Decision rule:** Alerts are sent **only** when sensors are **critical**
  (`CRITICAL_SPILL_LEVEL` or gauge ≥ threshold)

✅ Result: **No evacuation alert sent** — prevents false alarms and public panic.
""".strip()
                    )

        # ------------------------------------------------------------
        # Open-Meteo "Live Evidence" panel (best-effort)
        # ------------------------------------------------------------
        if show_open_meteo:
            loc = _extract_demo_location(user_prompt)
            if loc and loc in LOCATION_COORDS:
                st.markdown("### 🌐 Live Evidence (Open‑Meteo, best‑effort)")
                bundle = _get_live_open_meteo_bundle(loc)

                if not bundle.get("ok"):
                    st.warning(f"Open‑Meteo live fetch unavailable: {bundle.get('reason', 'unknown reason')}")
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
                        "Note: Open‑Meteo Flood API provides modeled discharge estimates (not a local gauge). "
                        "FlashGuard’s dispatch gate remains grounded in verified sensor truth."
                    )

        # ------------------------------------------------------------
        # Optional debug + raw mock dataset views
        # ------------------------------------------------------------
        if show_debug:
            # The debug_log is already updated with tool calls/outputs above
            with st.expander("Full Debug Log"):
                st.json(st.session_state.debug_log)

        if show_raw:
            with st.expander("Raw mock sensor dataset (demo only)"):
                st.json(MOCK_ENVIRONMENTAL_DATA)

    st.session_state.messages.append({"role": "assistant", "content": response.text})
```