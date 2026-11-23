"""Reading batches from the Edge: sliding-window store plus Cloud-side tables.

Alerts stay the immutable event log. Readings are a bounded window used for the
panel, incident peaks and (with labels) retraining. Dedup key: (run, node, boot, seq).
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone

READINGS_TOPIC = "wildfiresense/edge/evidence"
MAX_BATCH_BYTES = 1 << 20
MAX_BATCH_READINGS = 5000
WINDOW_S = 6 * 3600
FIELDS = ("PM2.5", "PM10", "temperature", "humidity")

SCHEMA = (
    """CREATE TABLE IF NOT EXISTS readings (
        run_id TEXT NOT NULL,
        node_id TEXT NOT NULL,
        boot_id TEXT NOT NULL,
        seq INTEGER NOT NULL,
        ts_ms INTEGER NOT NULL,
        kind TEXT NOT NULL,
        pm25 REAL, pm10 REAL, temperature REAL, humidity REAL,
        batch_id TEXT NOT NULL,
        PRIMARY KEY (run_id, node_id, boot_id, seq)
    )""",
    "CREATE INDEX IF NOT EXISTS idx_readings_time ON readings(run_id, ts_ms)",
    "CREATE INDEX IF NOT EXISTS idx_readings_node ON readings(run_id, node_id, ts_ms)",
    """CREATE TABLE IF NOT EXISTS labels (
        incident_id TEXT NOT NULL,
        label TEXT NOT NULL CHECK (label IN ('correct', 'incorrect')),
        true_class TEXT,
        source TEXT NOT NULL CHECK (source IN ('scenario', 'manual')),
        created_at TEXT NOT NULL,
        PRIMARY KEY (incident_id, source)
    )""",
    """CREATE TABLE IF NOT EXISTS nodes (
        node_id TEXT PRIMARY KEY,
        zone TEXT NOT NULL,
        x INTEGER NOT NULL,
        y INTEGER NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS truth (
        run_id TEXT PRIMARY KEY,
        payload_json TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS models (
        version TEXT PRIMARY KEY,
        status TEXT NOT NULL CHECK (status IN ('active', 'candidate', 'previous', 'rejected')),
        metrics_json TEXT,
        created_at TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS retrain_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        started_at TEXT NOT NULL,
        finished_at TEXT,
        trigger TEXT NOT NULL,
        labels_used INTEGER,
        result TEXT,
        model_version TEXT
    )""",
)


GRID_WIDTH = 32
GRID_NODES = 1024
ZONE_SIZE = 4


def grid_position(node):
    """node_0496 -> (zone, x, y) on the 32x32 grid, or None outside it."""
    name = node_name(node)
    if name is None or not 1 <= int(name[5:]) <= GRID_NODES:
        return None
    idx = int(name[5:]) - 1
    x, y = idx % GRID_WIDTH, idx // GRID_WIDTH
    return f"z_{y // ZONE_SIZE}_{x // ZONE_SIZE}", x, y


def create_tables(conn):
    for statement in SCHEMA:
        conn.execute(statement)
    if conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0] != GRID_NODES:
        for n in range(1, GRID_NODES + 1):
            name = f"node_{n:04d}"
            zone, x, y = grid_position(name)
            conn.execute(
                "INSERT OR IGNORE INTO nodes(node_id,zone,x,y) VALUES (?,?,?,?)",
                (name, zone, x, y),
            )


def node_name(raw):
    """Accepts 496, "496" or "node_0496"; returns "node_0496" or None."""
    if isinstance(raw, bool):
        return None
    if isinstance(raw, str) and raw.startswith("node_"):
        raw = raw[5:]
    try:
        num = int(raw)
    except (TypeError, ValueError):
        return None
    return f"node_{num:04d}" if 1 <= num <= 9999 else None


def _ts_ms(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value) if math.isfinite(value) and value > 0 else None
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    return None


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def parse_batch(payload):
    """Returns (batch_id, run_id, rows) or raises ValueError on a malformed batch."""
    if not isinstance(payload, bytes) or not 0 < len(payload) <= MAX_BATCH_BYTES:
        raise ValueError("size")
    try:
        data = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("json") from exc
    if not isinstance(data, dict):
        raise ValueError("shape")
    batch_id = data.get("batch_id")
    run_id = data.get("run_id", "default")
    items = data.get("readings")
    if (
        not isinstance(batch_id, str)
        or not 0 < len(batch_id) <= 128
        or not isinstance(run_id, str)
        or not 0 < len(run_id) <= 128
        or not isinstance(items, list)
        or len(items) > MAX_BATCH_READINGS
    ):
        raise ValueError("fields")
    rows = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("reading")
        node = node_name(item.get("id", item.get("node_id")))
        ts = _ts_ms(item.get("ts"))
        seq = item.get("seq")
        boot = item.get("boot_id", "default")
        values = item.get("values")
        kind = item.get("kind", "periodic")
        if (
            node is None
            or ts is None
            or isinstance(seq, bool)
            or not isinstance(seq, int)
            or seq < 0
            or not isinstance(boot, str)
            or not 0 < len(boot) <= 64
            or not isinstance(kind, str)
            or not isinstance(values, dict)
        ):
            raise ValueError("reading")
        numbers = [_number(values.get(f)) for f in FIELDS]
        if all(n is None for n in numbers):
            raise ValueError("reading")
        rows.append((run_id, node, boot, seq, ts, kind[:32], *numbers, batch_id))
    return batch_id, run_id, rows


def ingest_batch(conn, payload, window_s=WINDOW_S):
    """Returns (result, batch_id, inserted, duplicates).

    result is inserted|duplicate|invalid. A repeated batch is a delivered
    duplicate (safe to ACK); readings are deduplicated individually.
    """
    try:
        batch_id, run_id, rows = parse_batch(payload)
    except ValueError:
        return "invalid", None, 0, 0
    inserted = 0
    for row in rows:
        inserted += conn.execute(
            """INSERT OR IGNORE INTO readings
               (run_id,node_id,boot_id,seq,ts_ms,kind,pm25,pm10,temperature,humidity,batch_id)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            row,
        ).rowcount
    if rows:
        conn.execute(
            "DELETE FROM readings WHERE run_id=? AND ts_ms < "
            "(SELECT MAX(ts_ms) FROM readings WHERE run_id=?) - ?",
            (run_id, run_id, int(window_s * 1000)),
        )
    duplicates = len(rows) - inserted
    return ("inserted" if inserted or not rows else "duplicate"), batch_id, inserted, duplicates


def query(conn, run_id=None, node_id=None, since_ms=None, until_ms=None, limit=500):
    where, args = [], []
    for column, value in (("run_id", run_id), ("node_id", node_id)):
        if value is not None:
            where.append(f"{column} = ?")
            args.append(value)
    if since_ms is not None:
        where.append("ts_ms >= ?")
        args.append(since_ms)
    if until_ms is not None:
        where.append("ts_ms <= ?")
        args.append(until_ms)
    sql = (
        "SELECT run_id,node_id,boot_id,seq,ts_ms,kind,pm25,pm10,temperature,humidity "
        "FROM readings"
        + (" WHERE " + " AND ".join(where) if where else "")
        + " ORDER BY ts_ms DESC LIMIT ?"
    )
    args.append(max(1, min(1000, int(limit))))
    return [dict(r) for r in conn.execute(sql, args).fetchall()]


def peak_pm25(conn, run_id, nodes, start_ms, end_ms):
    if not nodes:
        return None
    marks = ",".join("?" for _ in nodes)
    row = conn.execute(
        f"SELECT MAX(pm25) FROM readings WHERE run_id=? AND node_id IN ({marks}) "
        "AND ts_ms BETWEEN ? AND ?",
        (run_id, *nodes, start_ms, end_ms),
    ).fetchone()
    return row[0] if row else None


def counts(conn):
    row = conn.execute(
        "SELECT COUNT(*), COUNT(DISTINCT node_id), MIN(ts_ms), MAX(ts_ms) FROM readings"
    ).fetchone()
    return dict(
        stored_readings=row[0],
        reading_nodes=row[1],
        window_from_ms=row[2],
        window_to_ms=row[3],
    )
