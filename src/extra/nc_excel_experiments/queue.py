from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from .run import DEFAULT_PROTOCOL, _load_protocol, _result_root, _utc_now, _write_json, run_selected


def _alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _available_memory_gb() -> float:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, value = line.split(":", 1)
        values[key] = int(value.strip().split()[0])
    return values["MemAvailable"] / 1024 / 1024


def main() -> None:
    parser = argparse.ArgumentParser(description="Resource-aware background queue for the executable Excel experiments.")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--wait-pid", type=int, default=None)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    protocol = _load_protocol(args.protocol)
    root = _result_root(protocol)
    root.mkdir(parents=True, exist_ok=True)
    queue_path = root / "queue_status.json"
    wait_pid = args.wait_pid or protocol["execution"].get("wait_for_pid")
    minimum_memory = float(protocol["execution"]["minimum_available_memory_gb"])
    _write_json(queue_path, {
        "status": "waiting",
        "started_at": _utc_now(),
        "wait_pid": wait_pid,
        "minimum_available_memory_gb": minimum_memory,
    })
    while _alive(wait_pid) or _available_memory_gb() < minimum_memory:
        _write_json(queue_path, {
            "status": "waiting",
            "updated_at": _utc_now(),
            "wait_pid": wait_pid,
            "wait_pid_alive": _alive(wait_pid),
            "available_memory_gb": _available_memory_gb(),
            "minimum_available_memory_gb": minimum_memory,
        })
        time.sleep(args.poll_seconds)
    _write_json(queue_path, {"status": "running", "started_experiments_at": _utc_now()})
    try:
        result = run_selected(args.protocol, list(protocol["execution"]["experiments"]))
    except Exception as exc:
        payload = {"status": "failed", "failed_at": _utc_now(), "error": repr(exc)}
        _write_json(queue_path, payload)
        raise
    payload = {"status": "completed", "completed_at": _utc_now(), "result": result}
    _write_json(queue_path, payload)
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
