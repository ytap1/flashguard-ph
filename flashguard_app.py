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

ADVERSARIAL_MOCK_DATA = {
    "Bulacan": {
        "timestamp": "2026-02-18T20:31:00Z",
        "alt_source": "Citizen Radio + LGU Drone Recon",
        "alt_flood_status": "SEVERE_FLOOD_WARNING",
        "estimated_inundation_meters": 1.4,
        "rescue_requests_count": 7,
        "confidence": "HIGH",
    },
    "Marikina": {
        "timestamp": "2026-02-18T20:31:00Z",
        "alt_source": "Citizen Radio + LGU Drone Recon",
        "alt_flood_status": "LOW_RISK",
        "estimated_inundation_meters": 0.1,
        "rescue_requests_count": 0,
        "confidence": "MEDIUM",
    },
}


# ============================================================
# 1) HELPER LOGIC (DETERMINISTIC)
# ============================================================
def _infer_location_from_text(input_text: str) -> str:
    text = input_text.lower()
    for known_location in MOCK_ENVIRONMENTAL_DATA.keys():
        if known_location.lower() in text:
            return known_location
    for known_location in ADVERSARIAL_MOCK_DATA.keys():
        if known_location.lower() in text:
            return known_location
    return input_text


def _is_sensor_critical(sensor_record: dict) -> bool:
    if not sensor_record:
        return False
    return (
        sensor_record.get("status") == "CRITICAL_SPILL_LEVEL"
        or sensor_record.get("river_gauge_meters", 0) >= sensor_record.get("critical_threshold_meters", 999999)
    )


def _standard_tool_response(ok: bool, payload: dict, message: str) -> str:
    return json.dumps({"ok": ok, "message": message, "payload": payload}, ensure_ascii=False)


def nemesis_ai(location_name: str) -> str:
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
            }
        )

    primary_is_critical = False
    if primary:
        primary_is_critical = (
            primary.get("river_gauge_meters", 0) >= primary.get("critical_threshold_meters", 999)
            or "CRITICAL" in str(primary.get("status", "")).upper()
        )

    adversarial_is_critical = False
    if adversarial:
        adversarial_is_critical = (
            str(adversarial.get("alt_flood_status", "")).upper() in {"SEVERE_FLOOD_WARNING", "CRITICAL"}
            or adversarial.get("estimated_inundation_meters", 0) >= 0.5
            or adversarial.get("rescue_requests_count", 0) >= 2
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
        }
    )


# ============================================================
# 2) GEMINI TOOL FUNCTIONS
# ============================================================
def check_pagasa_water_level(location_name: str) -> str:
    st.toast(f"📡 System: Checking sensor feeds for {location_name}...")

    record = MOCK_ENVIRONMENTAL_DATA.get(location_name)
    if record:
        return _standard_tool_response(
            ok=True,
            payload=record,
            message=f"Found sensor record for {location_name}.",
        )

    return _standard_tool_response(
        ok=False,
        payload={"location": location_name},
        message="No sensor record found in demo dataset. (Mock mode)",
    )


def check_social_media_reports(area_name: str) -> str:
    st.toast(f"📱 System: Scanning citizen reports for {area_name}...")

    if "marikina" in area_name.lower():
        reports = {
            "verified_reports_count": 3,
            "highlights": [
                "Waist-deep water reported on Main Street",
                "1 user requesting boat rescue",
            ],
            "confidence": "MEDIUM",
        }
        return _standard_tool_response(True, reports, "Citizen reports detected (mock).")

    reports = {"verified_reports_count": 0, "highlights": [], "confidence": "LOW"}
    return _standard_tool_response(True, reports, "No recent citizen reports found (mock).")


def dispatch_emergency_alert(area_name: str, action_plan: str) -> str:
    st.toast(f"🚨 ACTION: Validating before dispatch to {area_name}...")

    location_key = _infer_location_from_text(area_name)
    sensor_record = MOCK_ENVIRONMENTAL_DATA.get(location_key, {})

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
# 3) STREAMLIT UI SETUP
# ============================================================
st.set_page_config(page_title="FlashGuard PH", page_icon="⛈️", layout="centered")
st.title("⛈️ FlashGuard PH")
st.subheader("Autonomous Crisis Management System (Hackathon Demo)")
st.info("System Status: Online (Mock mode). Monitoring sensors + citizen comms.")

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
2) Only if the sensor result indicates CRITICAL risk, you MUST call:
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
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing risk data..."):
            response = st.session_state.crisis_agent.send_message(user_prompt)
            st.markdown(response.text)

        if show_debug:
            st.caption("Debug timeline (demo)")
            st.session_state.debug_log.append({"prompt": user_prompt, "response": response.text})
            with st.expander("Debug log"):
                st.json(st.session_state.debug_log)

        if show_raw:
            with st.expander("Raw mock sensor dataset (demo only)"):
                st.json(MOCK_ENVIRONMENTAL_DATA)

    st.session_state.messages.append({"role": "assistant", "content": response.text})
