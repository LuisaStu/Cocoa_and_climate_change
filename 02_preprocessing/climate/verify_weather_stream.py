"""Exercise the bounded reader on every downloaded scenario endpoint combination."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from download_nex_gddp_historical import load_yaml
from weather_stream import year_blocks


def main():
    rows = []
    models = load_yaml(Path("config/gcm_ensemble_proposal.yml"))["primary"]["models"]
    for model in models:
        for scenario in ["ssp245", "ssp585"]:
            for year in [2060, 2080]:
                days = 0
                maximum = 0
                digest = hashlib.sha256()
                for block in year_blocks(model, scenario, year):
                    n = len(block["day_of_year"])
                    np.testing.assert_array_equal(
                        block["day_of_year"], np.arange(days + 1, days + n + 1)
                    )
                    days += n
                    arrays = [block[k] for k in ["precip_mm", "tmin_c", "tmax_c", "srad_mj_m2"]]
                    maximum = max(maximum, sum(a.nbytes for a in arrays))
                    for array in arrays:
                        digest.update(array.tobytes())
                rows.append(
                    {
                        "model": model,
                        "scenario": scenario,
                        "year": year,
                        "calendar": block["calendar"],
                        "days": days,
                        "cells": len(block["cell_ids"]),
                        "maximum_returned_weather_bytes": maximum,
                        "stream_sha256": digest.hexdigest(),
                    }
                )
        print(f"Stream verified: {model}", flush=True)
    out = Path("06_validation/forcing_stream")
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "completed_utc": datetime.now(UTC).isoformat(),
        "combinations": len(rows),
        "rows": rows,
        "derived_daily_files_written": 0,
        "code_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ["weather_stream.py", "verify_weather_stream.py"]
        },
    }
    (out / "endpoint_stream_verification.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
