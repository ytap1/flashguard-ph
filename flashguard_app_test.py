import json
import unittest
from unittest.mock import MagicMock, patch
import sys

# Mock streamlit before importing the app
mock_st = MagicMock()
def mock_decorator(*args, **kwargs):
    return lambda f: f
mock_st.cache_data = mock_decorator
mock_st.cache_resource = mock_decorator
sys.modules["streamlit"] = mock_st

# Import functions to test
from flashguard_app import (
    _infer_location_from_text,
    _is_sensor_critical,
    nemesis_ai,
    check_pagasa_water_level,
    check_social_media_reports,
    dispatch_emergency_alert,
    MOCK_ENVIRONMENTAL_DATA,
    ADVERSARIAL_MOCK_DATA
)

class TestFlashGuard(unittest.TestCase):
    def setUp(self):
        # Reset mock_st for each test
        mock_st.reset_mock()

    def test_infer_location(self):
        self.assertEqual(_infer_location_from_text("Check Bulacan"), "Bulacan")
        self.assertEqual(_infer_location_from_text("check manila"), "Manila")
        self.assertEqual(_infer_location_from_text("what is the status in San Lorenzo?"), "San Lorenzo")
        self.assertEqual(_infer_location_from_text("random"), "random")

    def test_is_sensor_critical(self):
        self.assertTrue(_is_sensor_critical(MOCK_ENVIRONMENTAL_DATA["Bulacan"]))
        self.assertFalse(_is_sensor_critical(MOCK_ENVIRONMENTAL_DATA["Marikina"]))
        self.assertTrue(_is_sensor_critical(MOCK_ENVIRONMENTAL_DATA["Manila"]))
        self.assertFalse(_is_sensor_critical(MOCK_ENVIRONMENTAL_DATA["San Lorenzo"]))

    def test_nemesis_ai_approval(self):
        # Bulacan has critical in both primary and adversarial
        result = json.loads(nemesis_ai("Bulacan"))
        self.assertEqual(result["decision"], "APPROVE")
        self.assertEqual(result["priority"], 1)

    def test_nemesis_ai_block_conflict(self):
        # Marikina has NORMAL in primary but LOW_RISK in adversarial (actually same, but let's check logic)
        # Marikina is NORMAL (primary_is_critical=False) and LOW_RISK (adversarial_is_critical=False)
        # So decision should be BLOCK (both low risk)
        result = json.loads(nemesis_ai("Marikina"))
        self.assertEqual(result["decision"], "BLOCK")
        self.assertEqual(result["reason"], "Both sources indicate low risk for immediate auto-dispatch.")

    def test_check_pagasa_tool(self):
        # Should return JSON string with ok=True
        resp_str = check_pagasa_water_level("Bulacan")
        resp = json.loads(resp_str)
        self.assertTrue(resp["ok"])
        self.assertIn("payload", resp)
        self.assertEqual(resp["payload"]["river_basin"], "Angat River System")

    def test_check_pagasa_tool_missing_location(self):
        # Location not in mock should still return ok=True (per my fix)
        resp_str = check_pagasa_water_level("UnknownCity")
        resp = json.loads(resp_str)
        self.assertTrue(resp["ok"])
        self.assertIsNone(resp["payload"]["sensor_truth"])

    def test_check_social_reports(self):
        resp_str = check_social_media_reports("Marikina")
        resp = json.loads(resp_str)
        self.assertTrue(resp["ok"])
        self.assertEqual(resp["payload"]["verified_reports_count"], 3)

    def test_dispatch_blocked_by_sensor(self):
        # San Lorenzo is Normal
        resp_str = dispatch_emergency_alert("San Lorenzo", "Evacuate")
        resp = json.loads(resp_str)
        self.assertFalse(resp["ok"])
        self.assertIn("Dispatch blocked", resp["payload"]["reason"])

    def test_dispatch_approved(self):
        # Manila is Critical in both
        resp_str = dispatch_emergency_alert("Manila", "Evacuate")
        resp = json.loads(resp_str)
        self.assertTrue(resp["ok"])
        self.assertEqual(resp["payload"]["priority"], 1)
        self.assertIn("🚨 HIGH PRIORITY", resp["payload"]["dispatch"])

if __name__ == "__main__":
    unittest.main()
