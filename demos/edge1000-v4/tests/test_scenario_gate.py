"""The deployed model must pass the team's nine scenarios (external to the training generator)."""

import unittest

from edge.model import ROOT
from lab.scenario_gate import DEFAULT_DIR, SCENARIOS, compare, run_all


@unittest.skipUnless(DEFAULT_DIR.exists(), "scenario files not present")
class ScenarioGateTest(unittest.TestCase):
    def test_deployed_model_passes_all_nine(self):
        report = run_all(ROOT / "models/forest.json")
        self.assertEqual(report["total"], len(SCENARIOS))
        self.assertEqual(report["failed"], [], report["results"])

    def test_same_model_is_not_a_regression(self):
        model = ROOT / "models/forest.json"
        self.assertTrue(compare(model, model)["approved"])


if __name__ == "__main__":
    unittest.main()
