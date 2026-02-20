"""
FlashGuard PH — Hackathon Demo Prototype (Streamlit + Gemini Tool Calling)

GOAL:
- Showcase "autonomous" flood-risk decisions in a controlled demo.
- Scenario A (Bulacan): critical sensors => immediate evacuation alert dispatch.
- Scenario B (Marikina): "NORMAL" sensors => suppress alert (prevent false alarms).

IMPORTANT DEMO NOTE:
- In production, the functions below would call real APIs (PAGASA, satellite feeds, X/Twitter).
- For hackathon speed + offline reliability, we HARD-CODE mock data instead of calling live services.
"""

import json
import streamlit as st
from google import genai
from google.genai import types


# ============================================================
# 0) DEMO DATA (MOCKED) — in production this would come from APIs
# ============================================================
# Example of what would normally happen:
#   data = requests.get("https://pagasa.gov.ph/api/river-gauges?...").json()
# But for demo purposes, we hardcode realistic-looking samples
# to save time and ensure repeatable demo behavior.
MOCK_ENVIRONMENTAL_DATA = {
    "Bulacan": {
        "timestamp": "2026-02-18T20:30:00Z",
        "river_basin": "Angat River System",
        "river_gauge_meters": 18.5,
        "critical_threshold_meters": 15.0,
        "rainfall_mm_per_hr": 45.2,
        "satellite_soil_saturation": "94%",
        "satellite_cloud_cover": "Heavy Nimbus",
        "status": "CRITICAL_SPILL_LEVEL",
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
    },
}


# ============================================================
# 1) HELPER LOGIC (DETERMINISTIC) — keeps the demo "QA grounded"
# ============================================================
def _is_sensor_critical(sensor_record: dict) -> bool:
    """
    Hard rule for demo safety:
    - Only allow dispatch if status is explicitly CRITICAL_SPILL_LEVEL
      OR if river gauge is above critical threshold.
    """
    if not sensor_record:
        return False
    return (
        sensor_record.get("status") == "CRITICAL_SPILL_LEVEL"
        or sensor_record.get("river_gauge_meters", 0) >= sensor_record.get("critical_threshold_meters", 999999)
    )


def _standard_tool_response(ok: bool, payload: dict, message: str) -> str:
    """
    All tools return a *consistent JSON string*.
    This makes the AI's job easier and avoids parsing ambiguity.
    """
    return json.dumps(
        {
            "ok": ok,
            "message": message,
            "payload": payload,
        },
        ensure_ascii=False,
    )


# ============================================================
# 2) GEMINI TOOL FUNCTIONS (these are callable by the model)
# ============================================================
def check_pagasa_water_level(location_name: str) -> str:
    """
    Tool: Simulated "official" data lookup (PAGASA + satellite).

    In production:
      - Call real river gauge API, rainfall, forecast, satellite soil moisture feeds.
    In demo:
      - Use MOCK_ENVIRONMENTAL_DATA for fast + reliable scenarios.
    """
    st.toast(f"📡 System: Checking sensor feeds for {location_name}...")

    record = MOCK_ENVIRONMENTAL_DATA.get(location_name)
    if record:
        return _standard_tool_response(
            ok=True,
            payload=record,
            message=f"Found sensor record for {location_name}.",
        )

    # Unknown location => still return JSON, not plain text
    return _standard_tool_response(
        ok=False,
        payload={"location": location_name},
        message="No sensor record found in demo dataset. (Mock mode)",
    )


def check_social_media_reports(area_name: str) -> str:
    """
    Tool: Simulated social listening (X/Twitter citizen reports).

    In production:
      - Query X/Twitter, news, hotlines, barangay channels, etc.
      - Verify with geotags, media evidence, and rate-limit protection.
    In demo:
      - Return deterministic results to illustrate 'signal vs truth' contrast.
    """
    st.toast(f"📱 System: Scanning citizen reports for {area_name}...")

    # Demo rule: Marikina always has some citizen chatter, even if sensors are normal.
    if "Marikina" in area_name:
        reports = {
            "verified_reports_count": 3,
            "highlights": [
                "Waist-deep water reported on Main Street",
                "1 user requesting boat rescue",
            ],
            "confidence": "MEDIUM",
        }
        return _standard_tool_response(True, reports, "Citizen reports detected (mock).")

    # Everything else: no reports (mock)
    reports = {"verified_reports_count": 0, "highlights": [], "confidence": "LOW"}
    return _standard_tool_response(True, reports, "No recent citizen reports found (mock).")


def dispatch_emergency_alert(area_name: str, action_plan: str) -> str:
    """
    Tool: Simulated dispatch of evacuation SMS + responder coordination.

    QA SAFETY IMPROVEMENT:
    - This tool re-checks sensors BEFORE dispatching.
    - Even if the AI tries to dispatch incorrectly, the tool will refuse
      unless sensor data is critical.
    """
    st.toast(f"🚨 ACTION: Validating before dispatch to {area_name}...")

    sensor_record = MOCK_ENVIRONMENTAL_DATA.get(area_name, {})
    if not _is_sensor_critical(sensor_record):
        # HARD BLOCK false alarms — this is what you can call "QA-tested reliability"
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
# 3) STREAMLIT UI SETUP
# ============================================================
st.set_page_config(page_title="FlashGuard PH", page_icon="⛈️", layout="centered")
st.title("⛈️ FlashGuard PH")
st.subheader("Autonomous Crisis Management System (Hackathon Demo)")
st.info("System Status: Online (Mock mode). Monitoring sensors + citizen comms.")

# Sidebar helpers for judges/demo flow
with st.sidebar:
    st.header("🎬 Demo Controls")
    st.caption("One-click prompts for the two scenarios.")
    if st.button("Scenario A: Bulacan (Critical → Dispatch)"):
        st.session_state["_demo_prompt"] = "Check flood status in Bulacan."
    if st.button("Scenario B: Marikina (Normal → Suppress)"):
        st.session_state["_demo_prompt"] = "Check flood status in Marikina."

    st.divider()
    show_raw = st.checkbox("Show raw tool JSON (for judges)", value=True)
    show_debug = st.checkbox("Show debug timeline", value=False)


# ============================================================
# 4) INITIALIZE GEMINI CHAT AGENT (ONCE PER SESSION)
# ============================================================
# Note: Streamlit secrets must include GEMINI_API_KEY.
# Example: .streamlit/secrets.toml
#   GEMINI_API_KEY="YOUR_KEY_HERE"
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    st.error("Missing GEMINI_API_KEY in Streamlit secrets. Add it to .streamlit/secrets.toml")
    st.stop()

if "crisis_agent" not in st.session_state:
    client = genai.Client(api_key=api_key)

    # System instruction: forces tool usage and multilingual SMS drafting
    directive = """
You are FlashGuard PH, an autonomous disaster response AI for a hackathon demo.

NON-NEGOTIABLE PROTOCOL:
1) When a user asks about a location or flood status, you MUST call:
   - check_pagasa_water_level(location)
   - check_social_media_reports(location)
2) Only if the sensor result indicates CRITICAL risk, you MUST call:
   - dispatch_emergency_alert(location, action_plan)
   NOTE: If sensors are NORMAL, you MUST NOT dispatch.
3) In your final response:
   - Summarize sensor findings and citizen reports separately
   - Provide a short bilingual SMS alert (English + Tagalog)
   - If no dispatch happened, explicitly say "No alert sent" and why
Keep responses factual and concise. No hallucinations.
"""

    st.session_state.crisis_agent = client.chats.create(
        model="gemini-2.5-flash-lite",
        config=types.GenerateContentConfig(
            tools=[check_pagasa_water_level, check_social_media_reports, dispatch_emergency_alert],
            temperature=0.0,  # Strictly factual for demo
            system_instruction=directive,
        ),
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

if "debug_log" not in st.session_state:
    st.session_state.debug_log = []


# ============================================================
# 5) CHAT UI RENDER
# ============================================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ============================================================
# 6) INPUT HANDLING (manual input OR sidebar demo buttons)
# ============================================================
prefill = st.session_state.pop("_demo_prompt", None)
user_prompt = st.chat_input("Enter a crisis report or check a location...")

if prefill and not user_prompt:
    user_prompt = prefill

if user_prompt:
    # Save + display user message
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Ask Gemini agent (which will call tools as needed)
    with st.chat_message("assistant"):
        with st.spinner("Analyzing risk data..."):
            response = st.session_state.crisis_agent.send_message(user_prompt)
            st.markdown(response.text)

        # Optional: show debug panel (for judges)
        if show_debug:
            st.caption("Debug timeline (demo)")
            st.session_state.debug_log.append({"prompt": user_prompt, "response": response.text})
            with st.expander("Debug log"):
                st.json(st.session_state.debug_log)

        # Optional: show raw mock datasets used by tools
        if show_raw:
            with st.expander("Raw mock sensor dataset (demo only)"):
                st.json(MOCK_ENVIRONMENTAL_DATA)

    # Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": response.text})