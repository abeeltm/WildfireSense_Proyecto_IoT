"""Reading batches: validation, idempotency, bounded window, propagation timeline."""

import json
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

from cloud import readings as rd
from cloud.app import AlertStore, TOPIC, BrokerStatus, build_handler

T0 = 1784124000000


def reading(node, seq, t_s, pm25=10.0, boot="b1"):
    return {
        "id": node,
        "ts": T0 + t_s * 1000,
        "seq": seq,
        "boot_id": boot,
        "kind": "periodic",
        "values": {"PM2.5": pm25, "PM10": pm25 * 1.6, "temperature": 20.0, "humidity": 50.0},
    }


def batch(items, ident="b-1", run="r1"):
    return json.dumps({"batch_id": ident, "run_id": run, "readings": items}).encode()


def alert(ident, node, minute, run="r1"):
    return json.dumps(
        {
            "event_id": ident,
            "node_id": node,
            "type": "environmental_alert",
            "decision": "suspected_fire",
            "run_id": run,
            "priority": "high",
            "fire_score": 0.9,
            "generated_at": "2026-11-20T19:00:00+00:00",
            "reading_timestamp": f"2026-11-20T19:{minute:02d}:00+00:00",
        }
    ).encode()


class ReadingsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = AlertStore(Path(self.temp.name) / "cloud.sqlite")

    def tearDown(self):
        self.temp.cleanup()

    def test_node_names_and_validation(self):
        self.assertEqual(rd.node_name(496), "node_0496")
        self.assertEqual(rd.node_name("node_0496"), "node_0496")
        self.assertIsNone(rd.node_name(0))
        self.assertIsNone(rd.node_name(True))
        bad = [reading(0, 1, 0), {**reading(1, 1, 0), "values": {}}, {**reading(1, 1, 0), "seq": -1}]
        for item in bad:
            self.assertEqual(self.store.ingest_readings(batch([item]))[0], "invalid")
        self.assertEqual(self.store.ingest_readings(b"not json")[0], "invalid")
        self.assertEqual(self.store.ingest_readings(b"")[0], "invalid")
        self.assertEqual(self.store.stats()["stored_readings"], 0)

    def test_batch_is_idempotent_and_deduplicates_readings(self):
        items = [reading(496, s, s * 10) for s in range(5)]
        self.assertEqual(self.store.ingest_readings(batch(items)), ("inserted", "b-1", 5, 0))
        self.assertEqual(self.store.ingest_readings(batch(items)), ("duplicate", "b-1", 0, 5))
        overlap = batch(items[3:] + [reading(496, 5, 50)], "b-2")
        self.assertEqual(self.store.ingest_readings(overlap), ("inserted", "b-2", 1, 2))
        self.assertEqual(self.store.stats()["stored_readings"], 6)
        self.assertEqual(self.store.stats()["reading_nodes"], 1)

    def test_same_seq_after_reboot_is_new_reading(self):
        self.store.ingest_readings(batch([reading(7, 1, 0, boot="a")]))
        self.store.ingest_readings(batch([reading(7, 1, 10, boot="b")], "b-2"))
        self.assertEqual(self.store.stats()["stored_readings"], 2)

    def test_window_purges_old_readings_per_run(self):
        self.store.ingest_readings(batch([reading(1, 1, 0)], "a"), window_s=60)
        self.store.ingest_readings(batch([reading(1, 1, 0)], "c", run="other"), window_s=60)
        self.store.ingest_readings(batch([reading(1, 2, 120)], "b"), window_s=60)
        rows = self.store.readings(run_id="r1")
        self.assertEqual([r["seq"] for r in rows], [2])
        self.assertEqual(len(self.store.readings(run_id="other")), 1)

    def test_query_filters(self):
        items = [reading(n, s, s * 10) for n in (1, 2) for s in range(4)]
        self.store.ingest_readings(batch(items))
        only = self.store.readings(run_id="r1", node_id="node_0002", since_ms=T0 + 10_000)
        self.assertEqual(len(only), 3)
        self.assertTrue(all(r["node_id"] == "node_0002" for r in only))
        self.assertEqual(len(self.store.readings(limit=2)), 2)

    def test_incident_timeline_and_peak(self):
        for ident, node, minute in (("a", "node_0496", 0), ("b", "node_0497", 4), ("c", "node_0529", 9)):
            self.assertEqual(self.store.ingest(TOPIC, alert(ident, node, minute)), "inserted")
        base = 1795201200000  # 2026-11-20T19:00:00Z
        items = [
            {**reading(496, 1, 0, pm25=80.0), "ts": base + 2 * 60_000},
            {**reading(497, 1, 0, pm25=140.0), "ts": base + 5 * 60_000},
            {**reading(1000, 1, 0, pm25=999.0), "ts": base + 5 * 60_000},
        ]
        self.store.ingest_readings(batch(items))
        incident = self.store.incidents()["incidents"][0]
        self.assertEqual(incident["first_node"], "node_0496")
        self.assertEqual([s["node"] for s in incident["timeline"]], ["node_0496", "node_0497", "node_0529"])
        self.assertEqual([s["offset_s"] for s in incident["timeline"]], [0.0, 240.0, 540.0])
        self.assertEqual(incident["duration_s"], 540.0)
        self.assertEqual(incident["peak_pm25"], 140.0)

    def test_http_readings_endpoint(self):
        self.store.ingest_readings(batch([reading(496, 1, 0), reading(497, 1, 0)]))
        server = ThreadingHTTPServer(("127.0.0.1", 0), build_handler(self.store, BrokerStatus()))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        base = f"http://127.0.0.1:{server.server_port}/api/readings"
        with urlopen(base + "?node_id=496") as response:
            data = json.load(response)
        self.assertEqual([r["node_id"] for r in data["readings"]], ["node_0496"])
        with self.assertRaises(HTTPError) as caught:
            urlopen(base + "?since=abc")
        self.assertEqual(caught.exception.code, 400)


if __name__ == "__main__":
    unittest.main()
