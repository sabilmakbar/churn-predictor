"""Shared metrics persistence: training scripts append run records to one JSONL file."""
import json
import os
from datetime import datetime, timezone

DEFAULT_METRICS_PATH = "results/metrics.jsonl"


def save_metrics(records, path=DEFAULT_METRICS_PATH, script=None):
    """
    Append run record(s) to a shared JSONL file (one JSON object per line).
    Each record gets a UTC timestamp and, if given, the originating script tag.
    """
    if isinstance(records, dict):
        records = [records]

    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    tagged = []
    for rec in records:
        rec = {"ts": ts, **({"script": script} if script else {}), **rec}
        tagged.append(rec)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        for rec in tagged:
            f.write(json.dumps(rec, default=str) + "\n")

    return path
