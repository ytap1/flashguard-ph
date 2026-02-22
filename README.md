🛑 FlashGuard PH — Autonomous Flood Decision System (Hackathon MVP)
FlashGuard PH is an AI‑native crisis management prototype designed to demonstrate how flood warnings can be issued faster and more safely using autonomous decision logic with built‑in safeguards against false alarms.
The Philippines is highly disaster‑prone, yet many systems today are data‑heavy but decision‑light. FlashGuard PH focuses on turning signals into decisions, while ensuring those decisions are fact‑grounded and verified.

🎯 What This Demo Shows
The MVP intentionally demonstrates two contrasting scenarios:
✅ Scenario A — Bulacan (Critical → Auto‑Dispatch)

Official sensor readings indicate critical spill levels
Independent adversarial sources confirm severe flooding
✅ The system automatically triggers an evacuation alert
No human approval required

✅ Scenario B — Marikina (Normal → Suppress)

Rainfall and citizen reports exist (noise / signal)
Official sensors remain below critical thresholds
✅ The system suppresses evacuation alerts
Prevents false alarms and public panic

These two cases prove that FlashGuard PH can be fast when needed, and cautious when required.

🧠 How FlashGuard PH Decides (High‑Level)
FlashGuard PH uses a multi‑layered decision pipeline:
1️⃣ Sensor Truth (Primary Gate)

Uses authoritative flood indicators (mocked for demo):

River gauge vs. critical threshold
Basin status (NORMAL vs CRITICAL)


No alert is possible unless sensors are critical

This ensures the system is grounded in factual measurements, not social noise.

2️⃣ Nemesis AI — Adversarial Cross‑Check (Safety Gate)
Every potential dispatch is verified by Nemesis AI, an adversarial validation layer that simulates an independent source such as:

Citizen radio
LGU drone reconnaissance
Rescue request density

Rules:

✅ Dispatch allowed only if BOTH sources agree it’s critical
❌ Dispatch blocked if:

Sources conflict
Both indicate low risk



This introduces a “two‑source agreement rule”, similar to safety‑critical systems in aviation and healthcare.

3️⃣ Live Weather & Flood Context (Open‑Meteo)
FlashGuard PH augments its decisions with live, open data from Open‑Meteo:

☔ Rainfall & precipitation forecasts
🌊 Modeled river discharge (GloFAS flood proxy)


Live data is used as supporting evidence, not as the final authority, ensuring:

✅ Demo reliability (works offline)
✅ No false dispatches due to noisy models



✅ Why Judges Can Trust This System

Deterministic safety gates prevent hallucinated actions
Adversarial AI actively blocks unsafe automation
Transparent explanations show why alerts were sent or suppressed
Offline‑safe architecture (mock truth + live augmentation)
No black‑box decisions

FlashGuard PH does not just warn — it decides responsibly.

🚀 Architecture Highlights

Streamlit UI for fast, explainable demos
Gemini Tool‑Calling for structured AI actions
Open‑Meteo (free, no API key) for live weather & flood context
Nemesis AI for adversarial validation
QA‑enforced dispatch gate (cannot be bypassed by the LLM)


🧪 Hackathon Scope Disclaimer
This MVP uses mocked sensor data for:

Repeatable demos
Offline reliability
Deterministic behavior

In production, FlashGuard PH is designed to integrate directly with:

PAGASA river gauges & rainfall feeds
Project NOAH hazard layers
LGU and national emergency systems


✅ In One Sentence (Pitch‑Ready)

FlashGuard PH is an autonomous flood‑response AI that acts fast when danger is real—and proves when it’s safe not to act.