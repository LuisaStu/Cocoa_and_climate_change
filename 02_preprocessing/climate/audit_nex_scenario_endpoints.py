"""Independently verify the completed scenario endpoint batch and country coverage."""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import rasterio
import xarray as xr
from download_nex_gddp_historical import load_yaml, validate_payload
from download_nex_scenarios import MANIFEST, jobs_for
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-years", action="store_true")
    args = parser.parse_args()
    manifest_path = (
        Path("01_inputs/manifests/nex_gddp_scenario_full.json") if args.all_years else MANIFEST
    )
    batch = json.loads(manifest_path.read_text())
    cfg = load_yaml(Path("config/gcms.yml"))
    proposal = load_yaml(Path("config/gcm_ensemble_proposal.yml"))
    jobs = jobs_for(cfg, proposal, list(range(2060, 2081)) if args.all_years else [2060, 2080])
    fields = ["model", "scenario", "member", "variable", "year"]
    entries = {tuple(e[k] for k in fields): e for e in batch["files"]}
    if set(entries) != set(jobs) or len(batch["files"]) != len(jobs) or batch["errors"]:
        raise ValueError("Requested scenario batch is not complete and unique")
    grid_path = Path("01_inputs/processed/grid_full_extent/model_grid_EPSG6933_11000m.tif")
    with rasterio.open(grid_path) as grid:
        codes = grid.read(1)
        active = np.isin(codes, [1, 2])
        shape, transform, crs = grid.shape, grid.transform, grid.crs
    licenses = {}
    temperatures = {}
    for i, job in enumerate(jobs, 1):
        model, scenario, member, variable, year = job
        entry = entries[job]
        payload = Path(entry["path"]).read_bytes()
        if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
            raise ValueError(f"Checksum mismatch: {job}")
        validate_payload(payload, variable, year, tuple(batch["bbox_lonlat"]))
        historical = Path(
            f"01_inputs/raw/nex_gddp_cmip6_historical_subset/{model}/historical/{member}/{variable}/{variable}_day_{model}_historical_{member}_gn_2014_v2.0.nc"
        )
        with (
            xr.open_dataset(entry["path"], engine="scipy", decode_times=False) as ds,
            xr.open_dataset(historical, engine="scipy", decode_times=False) as history,
        ):
            if not np.array_equal(ds.lat, history.lat) or not np.array_equal(ds.lon, history.lon):
                raise ValueError(f"Historical/scenario grid mismatch: {job}")
            data = ds[variable].transpose("time", "lat", "lon").values
            finite = np.isfinite(data)
            if not (finite.all(axis=0) | (~finite).all(axis=0)).all():
                raise ValueError(f"Intermittent daily missingness: {job}")
            if variable in {"pr", "rsds"} and np.any(data[finite] < 0):
                raise ValueError(f"Negative flux: {job}")
            lon = (ds.lon.values + 180) % 360 - 180
            xi, yi = np.argsort(lon), np.argsort(-ds.lat.values)
            # With a static mask, one coverage warp verifies every day's geometry.
            coverage = np.where(finite[0][yi][:, xi], 1.0, np.nan).astype("float32")
            target = np.full(shape, np.nan, dtype="float32")
            reproject(
                coverage,
                target,
                src_transform=from_origin(
                    float(lon[xi][0] - 0.125), float(ds.lat.values[yi][0] + 0.125), 0.25, 0.25
                ),
                src_crs="EPSG:4326",
                dst_transform=transform,
                dst_crs=crs,
                src_nodata=np.nan,
                dst_nodata=np.nan,
                resampling=Resampling.average,
            )
            if not np.isfinite(target[active]).all():
                raise ValueError(f"Missing country coverage: {job}")
            license = ds.attrs.get("cmip6_license", "missing")
            if str(ds.attrs.get("version")) != "2.0" or license != entry["cmip6_license"]:
                raise ValueError(f"Version/licence mismatch: {job}")
            licenses[license] = licenses.get(license, 0) + 1
            if variable == "tasmin":
                temperatures[model, scenario, year] = (
                    data.copy(),
                    ds.time.values.copy(),
                    dict(ds.time.attrs),
                )
            elif variable == "tasmax":
                low, times, attrs = temperatures.pop((model, scenario, year))
                if not np.array_equal(times, ds.time.values) or attrs != dict(ds.time.attrs):
                    raise ValueError(f"Temperature time mismatch: {job}")
                if not np.array_equal(np.isfinite(low), finite) or np.any(
                    low[finite] > data[finite]
                ):
                    raise ValueError(f"Temperature mask mismatch or inversion: {job}")
        if i % 16 == 0:
            print(f"Audited {i}/{len(jobs)} scenario files", flush=True)
    if temperatures:
        raise ValueError("Unpaired temperatures")
    report = {
        "completed_utc": datetime.now(UTC).isoformat(),
        "verified_files": len(jobs),
        "country_cells": int(active.sum()),
        "temperature_pairs": sum(j[3] == "tasmin" for j in jobs),
        "temperature_inversions": 0,
        "intermittent_missingness_files": 0,
        "negative_flux_files": 0,
        "license_counts": licenses,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    out = Path("06_validation/historical_gcm") / (
        "scenario_full_audit.json" if args.all_years else "scenario_endpoint_audit.json"
    )
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
