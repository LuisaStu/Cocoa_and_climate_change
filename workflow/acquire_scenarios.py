"""Run the full acquisition and subsequent independent audit as a logged job."""

import hashlib
import json
import msvcrt
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / "07_runs/logs/scenario_acquisition_status.json"


def update(phase, **extra):
    record = {
        "phase": phase,
        "pid": os.getpid(),
        "updated_utc": datetime.now(UTC).isoformat(),
        **extra,
    }
    temp = STATUS.with_suffix(".json.tmp")
    temp.write_text(json.dumps(record, indent=2), encoding="utf-8")
    temp.replace(STATUS)
    print(json.dumps(record), flush=True)


def main():
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    # The OS releases this byte-range lock even if the process crashes.
    with (STATUS.parent / "scenario_acquisition.lock").open("a+b") as lock:
        lock.seek(0, 2)
        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        inputs = [
            "02_preprocessing/climate/download_nex_scenarios.py",
            "02_preprocessing/climate/download_nex_gddp_historical.py",
            "02_preprocessing/climate/audit_nex_scenario_endpoints.py",
            "config/gcms.yml",
            "config/gcm_ensemble_proposal.yml",
            "config/study.yml",
        ]
        hashes = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in inputs}
        (STATUS.parent / "scenario_acquisition_code_snapshot.json").write_text(
            json.dumps(hashes, indent=2), encoding="utf-8"
        )
        for attempt in range(1, 3):
            update("downloading", attempt=attempt)
            result = subprocess.run(
                [sys.executable, "-u", inputs[0], "--all-years", "--workers", "4"],
                cwd=ROOT,
                check=False,
            )
            if result.returncode == 0:
                break
        else:
            update("download_failed", exit_code=result.returncode)
            return result.returncode
        update("auditing")
        result = subprocess.run(
            [sys.executable, "-u", inputs[2], "--all-years"], cwd=ROOT, check=False
        )
        update(
            "complete" if result.returncode == 0 else "audit_failed", exit_code=result.returncode
        )
        return result.returncode


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError) as error:
        print(f"Scenario workflow failed: {error}", file=sys.stderr, flush=True)
        sys.exit(1)
