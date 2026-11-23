"""Retraining trigger, gate-controlled promotion and rollback (training is injected)."""

import json
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from cloud.app import AlertStore, BrokerStatus, TOPIC, build_handler
from cloud.retrain import Retrainer
from tests.test_readings import alert, batch, reading


def model_file(path, version):
    Path(path).write_text(json.dumps({"version": version}), encoding="utf-8")


class RetrainTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.store = AlertStore(root / "c.sqlite")
        self.model = root / "forest.json"
        model_file(self.model, "rf-base")
        self.approved = True
        self.promoted = []
        self.counter = 0

        def train(extra, output):
            self.counter += 1
            model_file(output, f"rf-new{self.counter}")

        def gate(candidate, baseline, report):
            return {"approved": self.approved, "same_version": False, "reasons": [] if self.approved else ["macro_f1_regression"],
                    "baseline_model": "rf-base"}

        def promote(src, target):
            self.promoted.append(Path(src).name)
            Path(target).write_bytes(Path(src).read_bytes())

        self.rt = Retrainer(self.store, root / "work", self.model, trigger_labels=2,
                            train_fn=train, gate_fn=gate, promote_fn=promote,
                            scenario_fn=lambda c, b: self.scenarios)
        self.scenarios = {"approved": True, "candidate": {"passed": 9, "total": 9, "failed": []}}
        self.store.on_labels = self.rt.maybe_trigger
        # one labelled incident with readings around the alarm
        self.store.ingest(TOPIC, alert("a", "node_0496", 0))
        base = 1795201200000
        self.store.ingest_readings(batch([
            {**reading(496, s, 0, pm25=60.0), "ts": base + s * 10_000} for s in range(1, 40)
        ]))
        self.incident = self.store.incidents()["incidents"][0]["incident_id"]

    def tearDown(self):
        self.temp.cleanup()

    def test_initial_registry_and_no_data(self):
        state = self.rt.state()
        self.assertEqual([(m["version"], m["status"]) for m in state["models"]], [("rf-base", "active")])
        self.assertEqual(self.rt.run("manual")["result"], "no_data")

    def test_trigger_needs_enough_labels(self):
        self.store.set_label(self.incident, "incorrect", "normal")
        self.assertEqual(self.rt.pending_labels(), 1)
        self.assertFalse(self.rt.maybe_trigger())

    def test_candidate_promote_and_rollback(self):
        self.store.set_label(self.incident, "incorrect", "normal")
        out = self.rt.run("manual")
        self.assertEqual((out["result"], out["model_version"]), ("candidate", "rf-new1"))
        self.assertEqual(self.rt.pending_labels(), 0)
        self.assertEqual(self.rt.promote("rf-new1"), "rf-new1")
        status = {m["version"]: m["status"] for m in self.rt.state()["models"]}
        self.assertEqual(status, {"rf-base": "previous", "rf-new1": "active"})
        self.assertEqual(json.loads(self.model.read_text())["version"], "rf-new1")
        self.assertEqual(self.rt.rollback(), "rf-base")
        status = {m["version"]: m["status"] for m in self.rt.state()["models"]}
        self.assertEqual(status, {"rf-base": "active", "rf-new1": "candidate"})

    def test_rejected_candidate_cannot_be_promoted(self):
        self.store.set_label(self.incident, "incorrect", "normal")
        self.approved = False
        out = self.rt.run("manual")
        self.assertEqual((out["result"], out["model_version"]), ("rejected", "rf-new1"))
        with self.assertRaises(ValueError):
            self.rt.promote("rf-new1")
        with self.assertRaises(ValueError):
            self.rt.rollback()
        self.assertEqual(json.loads(self.model.read_text())["version"], "rf-base")

    def test_scenario_regression_rejects_candidate(self):
        self.store.set_label(self.incident, "incorrect", "normal")
        self.scenarios = {"approved": False, "regressions": ["niebla"],
                          "candidate": {"passed": 8, "total": 9, "failed": ["niebla"]}}
        out = self.rt.run("manual")
        self.assertEqual(out["result"], "rejected")
        metrics = {m["version"]: m for m in self.rt.state()["models"]}["rf-new1"]["metrics"]
        self.assertIn("scenario_regression:niebla", metrics["reasons"])
        self.assertEqual(metrics["scenarios"]["failed"], ["niebla"])
        with self.assertRaises(ValueError):
            self.rt.promote("rf-new1")

    def test_training_failure_is_recorded_not_fatal(self):
        self.store.set_label(self.incident, "incorrect", "normal")
        self.rt.train_fn = lambda extra, output: (_ for _ in ()).throw(RuntimeError("boom"))
        self.assertTrue(self.rt.run("manual")["result"].startswith("error"))
        self.assertEqual(self.rt.state()["runs"][0]["result"], "error: RuntimeError")

    def test_http_models_and_actions(self):
        srv = ThreadingHTTPServer(("127.0.0.1", 0), build_handler(self.store, BrokerStatus(), self.rt))
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.server_close)
        self.addCleanup(srv.shutdown)
        base = f"http://127.0.0.1:{srv.server_port}"
        with urlopen(base + "/api/models") as r:
            self.assertEqual(json.load(r)["models"][0]["version"], "rf-base")
        self.store.set_label(self.incident, "incorrect", "normal")
        self.rt.run("manual")

        def post(path, body):
            return urlopen(Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"}))

        with post("/api/models/promote", {"version": "rf-new1"}) as r:
            self.assertEqual(json.load(r)["active"], "rf-new1")
        with post("/api/models/rollback", {"x": 1}) as r:
            self.assertEqual(json.load(r)["active"], "rf-base")
        with self.assertRaises(HTTPError) as err:
            post("/api/models/promote", {"version": "nope"})
        self.assertEqual(err.exception.code, 400)


if __name__ == "__main__":
    unittest.main()
