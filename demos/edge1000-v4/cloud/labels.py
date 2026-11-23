"""Verified labels for incidents and export for retraining (lab/train.py --extra).

Only two sources are accepted: the scenario truth and a manual correction.
Edge output is never a label. Manual labels override scenario labels.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from . import readings as rd

FIRE_CLASSES = ("slow_fire", "fast_fire")
NON_FIRE_CLASSES = ("normal", "dust", "fog", "direct_sun")
BASE_MS = 1784073600000  # 2026-07-15T00:00:00Z, clock of the scenario files
PRE_MS = 600_000  # clean "normal" context kept this long before the first alarm
MARGIN_MS = 120_000
SCENARIO_CLASS = {
    "fuego_lento": ("correct", "slow_fire"),
    "fuego_rapido": ("correct", "fast_fire"),
    "polvo_tractor": ("incorrect", "dust"),
    "niebla": ("incorrect", "fog"),
    "sol_directo": ("incorrect", "direct_sun"),
}


def _now():
    return datetime.now(timezone.utc).isoformat()


def check_label(label, true_class):
    if label == "correct":
        return true_class if true_class in FIRE_CLASSES else None
    if label == "incorrect":
        return true_class or "normal" if (true_class or "normal") in NON_FIRE_CLASSES else None
    return None


def set_label(conn, incident_id, label, true_class, source):
    cls = check_label(label, true_class)
    if (
        cls is None
        or source not in ("scenario", "manual")
        or not isinstance(incident_id, str)
        or not 0 < len(incident_id) <= 64
    ):
        raise ValueError("invalid label")
    conn.execute(
        "INSERT OR REPLACE INTO labels(incident_id,label,true_class,source,created_at) "
        "VALUES (?,?,?,?,?)",
        (incident_id, label, cls, source, _now()),
    )
    return cls


def effective(conn):
    """incident_id -> (label, class, source); manual wins over scenario."""
    out = {}
    for row in conn.execute(
        "SELECT incident_id,label,true_class,source FROM labels "
        "ORDER BY CASE source WHEN 'scenario' THEN 0 ELSE 1 END"
    ):
        out[row["incident_id"]] = (row["label"], row["true_class"], row["source"])
    return out


def save_truth(conn, run_id, truth):
    if not isinstance(run_id, str) or not 0 < len(run_id) <= 128:
        raise ValueError("run_id")
    if not isinstance(truth, dict) or truth.get("escenario") not in SCENARIO_CLASS and (
        truth.get("alarma_esperada") is not False
    ):
        raise ValueError("unsupported scenario truth")
    conn.execute(
        "INSERT OR REPLACE INTO truth(run_id,payload_json) VALUES (?,?)",
        (run_id, json.dumps(truth, sort_keys=True)),
    )


def label_from_truth(conn, incidents):
    """Label every incident of runs with stored truth. Returns number labelled."""
    done = 0
    for item in incidents:
        row = conn.execute(
            "SELECT payload_json FROM truth WHERE run_id=?", (item["run_id"],)
        ).fetchone()
        if row is None:
            continue
        truth = json.loads(row["payload_json"])
        name = truth.get("escenario")
        t0_ms = BASE_MS + int(truth.get("t0_s", 0)) * 1000
        first_ms = rd._ts_ms(item["first_seen"])
        if truth.get("alarma_esperada") and name in SCENARIO_CLASS and first_ms >= t0_ms:
            label, cls = SCENARIO_CLASS[name]
        else:  # alarm without fire (dust, fog, sun, or before the fire started)
            label, cls = "incorrect", SCENARIO_CLASS.get(name, ("", "normal"))[1]
            if cls in FIRE_CLASSES:
                cls = "normal"
        set_label(conn, item["incident_id"], label, cls, "scenario")
        done += 1
    return done


def training_rows(conn, incidents):
    """Yield one reviewed-train record per labelled reading.

    Fire/other class applies from each node's first alarm (minus a small margin
    for non-fire) to its last alarm + margin. Readings more than PRE_MS before
    the first alarm are clean 'normal' context; the gap in between is skipped
    because it is ambiguous. Message format is the Edge's schema v1.
    """
    labels = effective(conn)
    for item in incidents:
        if item["incident_id"] not in labels:
            continue
        label, cls, _ = labels[item["incident_id"]]
        run = item["run_id"]
        last_ms = rd._ts_ms(item["last_seen"])
        for step in item["timeline"]:
            first = rd._ts_ms(step["first_alarm"])
            lo = first if label == "correct" else first - MARGIN_MS
            for r in conn.execute(
                "SELECT * FROM readings WHERE run_id=? AND node_id=? ORDER BY ts_ms, seq",
                (run, step["node"]),
            ):
                t = r["ts_ms"]
                if lo <= t <= last_ms + MARGIN_MS:
                    klass = cls
                elif t < first - PRE_MS:
                    klass = "normal"
                else:
                    continue
                yield {
                    "group": f"reviewed-train-{run}",
                    "class": klass,
                    "message": {
                        "schema_version": 1,
                        "type": "reading",
                        "run_id": run,
                        "node_id": r["node_id"],
                        "seq": r["seq"],
                        "timestamp": datetime.fromtimestamp(
                            t / 1000, timezone.utc
                        ).isoformat(),
                        "timebase": "simulated",
                        "boot_id": r["boot_id"],
                        "measurements": {
                            "pm25_ug_m3": r["pm25"],
                            "pm10_ug_m3": r["pm10"],
                            "air_temperature_c": r["temperature"],
                            "relative_humidity_pct": r["humidity"],
                        },
                    },
                }
