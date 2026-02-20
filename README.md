# FlashGuard PH

FlashGuard PH is a Streamlit-based autonomous crisis management prototype for flood-response triage. It combines environmental sensor checks, citizen report checks, and AI-guided emergency escalation in a single conversational interface.

## Core Capabilities

1. **Location-Based Flood Risk Assessment**
	- Accepts a user location/crisis prompt via chat.
	- Queries mocked environmental intelligence (`check_pagasa_water_level`) including:
	  - river gauge level
	  - critical threshold
	  - rainfall intensity
	  - satellite soil saturation
	  - cloud cover

2. **Citizen Signal Verification**
	- Checks social media-style distress signals (`check_social_media_reports`) for the target area.
	- Adds human-grounded context to sensor-based flood indicators.

3. **Autonomous Emergency Dispatch Trigger**
	- Uses a dispatch tool (`dispatch_emergency_alert`) to simulate evacuation SMS broadcasting and resource coordination.
	- Designed with a strict protocol in the agent system instruction:
	  - check environmental + social sources first
	  - dispatch only when flood evidence supports escalation
	  - apply Nemesis AI adversarial validation before final alert send

4. **Adversarial Counter-Check (Nemesis AI)**
	- Runs `nemesis_ai` before any alert dispatch is finalized.
	- Uses two internal datasets:
	  - `mock_environmental_data` (primary source)
	  - `adversarial_mock_data` (independent adversarial source)
	- Enforces a decision gate:
	  - `APPROVE`: both sources indicate critical flood risk
	  - `BLOCK`: conflicting signals or low-risk signals
	- Prevents automatic SMS dispatch when confidence is insufficient.

5. **AI Crisis Assistant (Gemini + Function Tools)**
	- Uses `google-genai` chat with tool-calling enabled.
	- Configured for deterministic behavior (`temperature=0.0`) to reduce hallucination risk in crisis scenarios.

6. **Operational Chat UI for Incident Handling**
	- Real-time Streamlit chat interface with:
	  - persistent chat history in session state
	  - inline status toasts/spinner during checks
	  - structured assistant responses

## Current Data Model

The app currently uses mock environmental data for sample areas (e.g., Bulacan, Marikina). This enables end-to-end demo of assessment and dispatch behavior without live API integration.

- `mock_environmental_data`: primary flood, rainfall, and river metrics
- `adversarial_mock_data`: adversarial second-opinion risk intelligence for counter-validation

## Nemesis AI Cross-Check Flow

1. User submits location or incident report.
2. FlashGuard evaluates environmental + social evidence.
3. Before dispatch, `dispatch_emergency_alert(...)` invokes `nemesis_ai(...)`.
4. `nemesis_ai` evaluates both `mock_environmental_data` and `adversarial_mock_data`.
5. Dispatch proceeds only when Nemesis AI returns `APPROVE`; otherwise alert is held for manual verification.

## Demo Validation Matrix

Use these prompts to quickly validate expected behavior in the current mock setup:

| Scenario | Example Prompt | Expected Nemesis Decision | Expected Dispatch Outcome |
|---|---|---|---|
| High-risk in Bulacan | "Flood alert in Bulacan, send help now." | `APPROVE` | Dispatch allowed (`SUCCESS`) |
| Low-risk in Marikina | "Check Marikina flood status and dispatch." | `BLOCK` | Dispatch held (`HOLD`) |
| Unknown location | "Dispatch to Barangay Z now." | `BLOCK` | Dispatch held (`HOLD`) |

## Architecture Summary

- **Frontend:** Streamlit chat app
- **AI Engine:** Gemini (`gemini-2.5-flash-lite`) via `google-genai`
- **Tool Functions:**
  - `check_pagasa_water_level(location_name)`
  - `check_social_media_reports(barangay_name)`
	- `nemesis_ai(location_name)`
  - `dispatch_emergency_alert(barangay_name, action_plan)`

## Setup

1. Install dependencies:

	```bash
	pip install -r requirements.txt
	```

2. Configure secrets for Streamlit:

	Create `.streamlit/secrets.toml` and add:

	```toml
	GEMINI_API_KEY = "your_api_key_here"
	```

## Run

```bash
streamlit run flashguard_app.py
```

## Notes and Limitations

- External integrations are currently mocked (environmental and social feeds).
- Dispatch is a simulated action for prototype/demo use, now protected by adversarial pre-dispatch validation.
- Production deployment should add:
  - live PAGASA/hydrology and satellite data connectors
  - verified social media ingestion pipeline
  - secure alerting integrations (SMS gateways, LGU dispatch systems)
  - observability, audit logs, and fail-safe controls
