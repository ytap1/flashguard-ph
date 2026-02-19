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
# 1. DEFINE CRISIS TOOLS 
# ==========================================    
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
    st.toast(f"🚨 ACTION TRIGGERED: Dispatching units to {barangay_name}!")
    return f"SUCCESS: Evacuation SMS broadcasted to all residents in {barangay_name}. Resources coordinated: {action_plan}"

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