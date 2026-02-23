@#  FlashGuard PH  Autonomous Flood Decision System (Hackathon MVP)

FlashGuard PH is an AInative autonomous crisis management prototype designed to show how flood warnings can be issued faster and more safely using autonomous decision logic with built-in safeguards against false alarms.

---

##  Pitch (Wait, what is this?)

**FlashGuard PH is an autonomous flood-response AI that acts fast when danger is realand proves when its safe not to act.**

The Philippines is highly disaster-prone, but many systems today are data-heavy but decision-light. FlashGuard PH focuses on turning signals into decisions, ensuring they are fact-grounded and cross-verified.

---

##  Demo Case Studies

The system demonstrates two contrasting scenarios:

 **Scenario A  Bulacan (Critical  Auto-Dispatch)**
- Official sensor readings indicate critical spill levels.
- Independent adversarial sources (Nemesis AI) confirm severe flooding.
- **Outcome:** System automatically triggers an evacuation alert.

 **Scenario B  Marikina (Normal  Suppress)**
- Rainfall and citizen reports exist (noise / signal).
- Official sensors remain below critical thresholds.
- **Outcome:** System suppresses evacuation alerts, preventing false alarms and public panic.

---

##  Core Capabilities & Logic

### 1. Sensor Truth (Primary Gate)
Uses authoritative flood indicators (mocked for demo):
- River gauge vs. critical threshold
- Basin status (NORMAL vs CRITICAL)
- Rainfall intensity & soil saturation

**Rule:** No alert is possible unless sensors hit critical thresholds. This ensures the system is grounded in factual measurements, not social noise.

### 2. Nemesis AI  Adversarial Cross-Check (Safety Gate)
Before any alert is dispatched, Nemesis AI runs an adversarial validation. It compares primary data against independent sources:
- Citizen radio
- LGU drone reconnaissance
- Rescue request density

**Rule:** Dispatch is allowed only if BOTH sources agree risk is critical. Conflicting signals result in a **BLOCK** for manual human verification.

### 3. Live Weather & Flood Context (Open-Meteo)
FlashGuard PH augments its decisions with live data from OpenMeteo:
-  Rainfall & precipitation forecasts
-  Modeled river discharge (GloFAS flood proxy)

Live data is used as supporting evidence, not as the final authority, ensuring demo reliability even if live APIs are noisy.

### 4. AI Crisis Assistant (Gemini + Tool Calling)
Uses `gemini-2.5-flash-lite` for intelligent orchestration. Deterministic behavior (`temperature=0.0`) ensures consistent, trustworthy decisions during a crisis.

---

##  Setup & Run

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure secrets:**
   Create `.streamlit/secrets.toml` and add:
   ```toml
   GEMINI_API_KEY = "your_api_key_here"
   ```

3. **Run the app:**
   ```bash
   streamlit run flashguard_app.py
   ```

---

##  Architecture Summary

- **Frontend:** Streamlit chat app
- **AI Engine:** Gemini via `google-genai`
- **Integrations:**
  - Mock Environmental Truth (deterministic)
  - Nemesis AI Adversarial Logic (safeguard)
  - OpenMeteo Live APIs (augmentation)

---

##  Notes & Limitations

- External integrations are currently hybrid (mocked sensors + live Open-Meteo).
- Dispatch is simulated for prototype/demo use.
- Production versions should add live national hydrology feeds (PAGASA, Project NOAH) and secure SMS/LGU dispatch gateways.
@
