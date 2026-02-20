import streamlit as st
from google import genai
from google.genai import types
import os
import json

# ==========================================
# MOCK API DATA: PAGASA + SATELLITE INSIGHTS
# ==========================================
mock_environmental_data = {
    "Bulacan": {
        "timestamp": "2026-02-18T20:30:00Z",
        "river_basin": "Angat River System",
        "river_gauge_meters": 18.5,
        "critical_threshold_meters": 15.0,
        "rainfall_mm_per_hr": 45.2,
        "satellite_soil_saturation": "94%", 
        "satellite_cloud_cover": "Heavy Nimbus",
        "status": "CRITICAL_SPILL_LEVEL"
    },
    "Marikina": {
        "timestamp": "2026-02-18T20:30:00Z",
        "river_basin": "Marikina River Basin",
        "river_gauge_meters": 12.1,
        "critical_threshold_meters": 15.0,
        "rainfall_mm_per_hr": 10.5,
        "satellite_soil_saturation": "60%",
        "satellite_cloud_cover": "Moderate",
        "status": "NORMAL"
    }
}

# ==========================================
# ADVERSARIAL AI MOCK DATA (SECOND OPINION)
# ==========================================
adversarial_mock_data = {
    "Bulacan": {
        "timestamp": "2026-02-18T20:31:00Z",
        "alt_source": "Citizen Radio + LGU Drone Recon",
        "alt_flood_status": "SEVERE_FLOOD_WARNING",
        "estimated_inundation_meters": 1.4,
        "rescue_requests_count": 7,
        "confidence": "HIGH"
    },
    "Marikina": {
        "timestamp": "2026-02-18T20:31:00Z",
        "alt_source": "Citizen Radio + LGU Drone Recon",
        "alt_flood_status": "LOW_RISK",
        "estimated_inundation_meters": 0.1,
        "rescue_requests_count": 0,
        "confidence": "MEDIUM"
    }
}

# ==========================================
# 1. DEFINE CRISIS TOOLS 
# ==========================================    
def nemesis_ai(location_name: str) -> str:
    """Adversarial AI checker that cross-validates dispatch safety using both primary and adversarial sources."""
    st.toast(f"🛡️ Nemesis AI: Running adversarial cross-check for {location_name}...")

    primary = mock_environmental_data.get(location_name)
    adversarial = adversarial_mock_data.get(location_name)

    if not primary and not adversarial:
        return json.dumps({
            "location": location_name,
            "decision": "BLOCK",
            "reason": "No reliable data in either source.",
            "used_sources": ["mock_environmental_data", "adversarial_mock_data"]
        })

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

    return json.dumps({
        "location": location_name,
        "decision": decision,
        "reason": reason,
        "primary_is_critical": primary_is_critical,
        "adversarial_is_critical": adversarial_is_critical,
        "used_sources": ["mock_environmental_data", "adversarial_mock_data"]
    })


def _infer_location_from_text(input_text: str) -> str:
    """Maps free-text barangay/location mentions to known mock datasets."""
    text = input_text.lower()
    for known_location in mock_environmental_data.keys():
        if known_location.lower() in text:
            return known_location
    for known_location in adversarial_mock_data.keys():
        if known_location.lower() in text:
            return known_location
    return input_text


def check_pagasa_water_level(location_name: str) -> str:
    """Checks official river gauges, rainfall, and satellite soil saturation for a location."""
    st.toast(f"📡 System: Pinging environmental APIs for {location_name}...")
    
    # Check if the requested location is in our new mock JSON database
    data = mock_environmental_data.get(location_name)
    
    if data:
        # Return the rich JSON data to the AI
        return json.dumps(data)
    else:
        return f"SAFE: No critical sensor data found for {location_name}." 

def check_social_media_reports(barangay_name: str) -> str:
    """Scrapes Twitter/X for geotagged citizen distress reports in the area."""
    st.toast(f"📱 System: Scanning social media for {barangay_name}...")
    
    if "Barangay X" in barangay_name or "Marikina" in barangay_name:
        return "3 verified reports of waist-deep water on Main Street. 1 user requesting boat rescue."
    return "0 recent flood reports found."

def dispatch_emergency_alert(barangay_name: str, action_plan: str) -> str:
    """Dispatches official SMS evacuation alerts and coordinates physical rescue resources."""
    location_key = _infer_location_from_text(barangay_name)
    adversarial_verdict = json.loads(nemesis_ai(location_key))

    if adversarial_verdict["decision"] != "APPROVE":
        st.toast(f"🛑 ALERT BLOCKED: Nemesis AI flagged {barangay_name} for manual verification.")
        return (
            f"HOLD: Nemesis AI blocked automatic dispatch for {barangay_name}. "
            f"Reason: {adversarial_verdict['reason']}"
        )

    st.toast(f"🚨 ACTION TRIGGERED: Dispatching units to {barangay_name}!")
    return (
        f"SUCCESS: Evacuation SMS broadcasted to all residents in {barangay_name}. "
        f"Resources coordinated: {action_plan}. "
        f"Nemesis AI verdict: {adversarial_verdict['decision']}"
    )

# ==========================================
# 2. STREAMLIT UI SETUP
# ==========================================
st.set_page_config(page_title="FlashGuard PH", page_icon="⛈️", layout="centered")
st.title("⛈️ FlashGuard PH")
st.subheader("Autonomous Crisis Management System")

# A little visual flair for the judges
st.info("System Status: Online. Monitoring PAGASA APIs and Citizen Comms.")

# ==========================================
# 3. INITIALIZE THE MASTER AGENT
# ==========================================
api_key = st.secrets["GEMINI_API_KEY"]

if "crisis_agent" not in st.session_state:
    print("API key: ", api_key)
    client = genai.Client(api_key=api_key)
    
    # Strict rules to prevent hallucination 
    directive = """
    You are FlashGuard PH, an elite, autonomous disaster response AI. 
    Your job is to evaluate flood risks and dispatch help.
    
    CRITICAL PROTOCOL:
    1. If a user asks about a location, you MUST use 'check_pagasa_water_level' AND 'check_social_media_reports' first.
    2. If BOTH sources indicate a flood, you MUST use the 'dispatch_emergency_alert' tool to send help.
    3. Draft a short, multilingual (English/Tagalog) SMS alert in your final response.
    4. Never dispatch resources if the sensors say the area is safe.
    5. Every dispatch is adversarially checked by Nemesis AI before alerts are sent.
    """
    
    st.session_state.crisis_agent = client.chats.create(
        model="gemini-2.5-flash-lite",
        config=types.GenerateContentConfig(
            tools=[check_pagasa_water_level, check_social_media_reports, dispatch_emergency_alert],
            temperature=0.0, # 0.0 means ZERO creativity/hallucination. Strictly factual.
            system_instruction=directive
        )
    )

# ==========================================
# 4. CHAT INTERFACE
# ==========================================
# Initialize chat history for the UI
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input box at the bottom of the screen
if user_prompt := st.chat_input("Enter a crisis report or check a location..."):
    
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Show AI thinking and tools running
    with st.chat_message("assistant"):
        with st.spinner("Analyzing risk data..."):
            response = st.session_state.crisis_agent.send_message(user_prompt)
            st.markdown(response.text)
            
    # Save AI response to history
    st.session_state.messages.append({"role": "assistant", "content": response.text})