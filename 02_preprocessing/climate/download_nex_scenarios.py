"""Download resumable regional scenario batches with incremental manifests."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import xarray as xr
from download_nex_gddp_historical import historical_bbox, load_yaml, ncss_url, validate_payload

ROOT = Path("01_inputs/raw/nex_gddp_cmip6_scenarios")
MANIFEST = Path("01_inputs/manifests/nex_gddp_scenario_endpoints.json")


def jobs_for(config, proposal, years):
    if not years or any(y < 2060 or y > 2080 for y in years):
        raise ValueError("Years must fall in 2060–2080")
    members = {m["id"]: m["member"] for m in config["candidate_models"]}
    models = proposal["primary"]["models"]
    if len(set(models)) != len(models) or not set(models).issubset(members):
        raise ValueError("Invalid ensemble model list")
    return [
        (m, s, members[m], v, y)
        for m, s, v, y in itertools.product(
            models,
            config["study"]["scenarios"],
            config["study"]["required_variables"],
            sorted(set(years)),
        )
    ]


def fetch_one(job, bbox, root=ROOT, known_hashes=None):
    model, scenario, member, variable, year = job
    filename = f"{variable}_day_{model}_{scenario}_{member}_gn_{year}_v2.0.nc"
    path = root / model / scenario / member / variable / filename
    url = ncss_url(model, scenario, member, variable, year, bbox)
    resumed = path.is_file()
    if resumed:
        payload = path.read_bytes()
        expected = (known_hashes or {}).get(path.as_posix())
        if expected and hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError(f"Existing file differs from its recorded checksum: {path}")
    else:
        request = Request(url, headers={"User-Agent": "SIMPLE-Cacao-WA research workflow"})
        for attempt in range(3):
            try:
                with urlopen(request, timeout=180) as response:
                    payload = response.read()
                break
            except (URLError, TimeoutError):
                if attempt == 2:
                    raise
                time.sleep(2 * (attempt + 1))
    metadata = validate_payload(payload, variable, year, bbox)
    with xr.open_dataset(BytesIO(payload), engine="scipy", decode_times=False) as ds:
        version = str(ds.attrs.get("version", ""))
        license = str(ds.attrs.get("cmip6_license", ""))
        if version != "2.0" or not license:
            raise ValueError(f"Missing/unexpected source version or licence: {path}")
        for key, expected in [
            ("cmip6_source_id", model),
            ("cmip6_variant_label", member),
            ("cmip6_experiment_id", scenario),
        ]:
            if key in ds.attrs and ds.attrs[key] != expected:
                raise ValueError(f"Source identity mismatch: {key}")
    if not resumed:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".nc.part")
        temporary.write_bytes(payload)
        temporary.replace(path)
    return {
        "model": model,
        "scenario": scenario,
        "member": member,
        **metadata,
        "path": path.as_posix(),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "source_url": url,
        "version": version,
        "cmip6_license": license,
        "resumed": resumed,
    }


def checkpoint(path, jobs, entries, errors, bbox):
    complete = len(entries) == len(jobs) and not errors
    record = {
        "status": "complete_for_requested_batch" if complete else "incomplete",
        "updated_utc": datetime.now(UTC).isoformat(),
        "expected_files": len(jobs),
        "verified_files": len(entries),
        "full_period_expected_files": 1344,
        "years": sorted({j[-1] for j in jobs}),
        "bbox_lonlat": list(bbox),
        "files": sorted(entries, key=lambda e: e["path"]),
        "errors": errors,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(record, indent=2), encoding="utf-8")
    # Windows readers/virus scanners can briefly deny replacement of an open file.
    # Keep the previous manifest intact and retry the atomic replacement.
    for attempt in range(20):
        try:
            temp.replace(path)
            break
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, nargs="+", default=[2060, 2080])
    parser.add_argument("--all-years", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("Use one to four workers")
    config = load_yaml(Path("config/gcms.yml"))
    proposal = load_yaml(Path("config/gcm_ensemble_proposal.yml"))
    years = list(range(2060, 2081)) if args.all_years else args.years
    jobs = jobs_for(config, proposal, years)
    bbox = historical_bbox(load_yaml(Path("config/study.yml")))
    manifest = args.manifest or (
        Path("01_inputs/manifests/nex_gddp_scenario_full.json") if args.all_years else MANIFEST
    )
    print(f"Requested {len(jobs)} regional files for years {years}", flush=True)
    if args.dry_run:
        return
    known_hashes = {}
    for saved_manifest in {manifest, MANIFEST}:
        if saved_manifest.is_file():
            saved = json.loads(saved_manifest.read_text(encoding="utf-8"))
            for entry in saved["files"]:
                key = Path(entry["path"]).as_posix()
                if key in known_hashes and known_hashes[key] != entry["sha256"]:
                    raise ValueError(f"Conflicting recorded checksums for {key}")
                known_hashes[key] = entry["sha256"]
    entries, errors = [], []
    checkpoint(manifest, jobs, entries, errors, bbox)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_one, j, bbox, ROOT, known_hashes): j for j in jobs}
        for index, future in enumerate(as_completed(futures), 1):
            job = futures[future]
            try:
                entries.append(future.result())
                print(f"[{index}/{len(jobs)}] OK {job}", flush=True)
            except (OSError, ValueError, RuntimeError) as error:
                errors.append({"job": list(job), "error": str(error)})
                print(f"[{index}/{len(jobs)}] ERROR {job}: {error}", flush=True)
            checkpoint(manifest, jobs, entries, errors, bbox)
    if errors:
        raise RuntimeError(f"{len(errors)} files failed; rerun to resume. See {manifest}")
    print(f"Verified batch complete: {manifest}", flush=True)


if __name__ == "__main__":
    main()
