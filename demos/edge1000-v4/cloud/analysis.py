"""Non-critical calculation offloaded when the measured link is healthy.

Summaries are per zone (4x4 node blocks). The legacy network-wide format
(no zone field) is still accepted.
"""

from edge.transport import aggregate
from .readings import node_name
import math
import re

ZONE = re.compile(r"z_\d{1,2}_\d{1,2}")


def _finite(x):
    return type(x) in (float, int) and math.isfinite(x)


def _span(value):
    if type(value) in (int, float) and math.isfinite(value):
        return value
    if isinstance(value, str) and 0 < len(value) <= 64:
        return value
    raise ValueError("Invalid time bound")


def summarize(data):
    mode = data["computed_by"]
    zone = data.get("zone")
    extra = {}
    if zone is not None:
        if not isinstance(zone, str) or not ZONE.fullmatch(zone):
            raise ValueError("Invalid zone")
        extra = dict(
            zone=zone, from_ts=_span(data["from_ts"]), to_ts=_span(data["to_ts"])
        )
    if mode == "cloud":
        samples = data["samples"]
        if not isinstance(samples, list) or not 1 <= len(samples) <= 512:
            raise ValueError("Invalid samples")
        if zone is not None:
            if not all(
                isinstance(s, dict)
                and node_name(s.get("node_id")) is not None
                and _finite(s.get("pm25"))
                for s in samples
            ):
                raise ValueError("Invalid zone samples")
            peak = max(samples, key=lambda s: s["pm25"])
            result = aggregate([s["pm25"] for s in samples])
            extra["max_node_id"] = node_name(peak["node_id"])
        else:
            if not all(_finite(x) for x in samples):
                raise ValueError("Invalid samples")
            result = aggregate(samples)
    elif mode == "edge":
        result = {k: data[k] for k in ("n", "mean_pm25", "max_pm25")}
        if (
            type(result["n"]) != int
            or not 1 <= result["n"] <= 512
            or not all(_finite(result[k]) for k in ("mean_pm25", "max_pm25"))
        ):
            raise ValueError("Invalid summary")
        if zone is not None:
            if node_name(data.get("max_node_id")) is None:
                raise ValueError("Invalid max_node_id")
            extra["max_node_id"] = node_name(data["max_node_id"])
    else:
        raise ValueError("Invalid routing mode")
    return dict(
        **result,
        **extra,
        computed_by=mode,
        generated_at=data["generated_at"],
        event_id=data["event_id"],
    )
