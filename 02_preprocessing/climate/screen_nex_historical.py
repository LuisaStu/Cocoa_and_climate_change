"""Reproducible country monthly-climatology screening, without model selection."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from download_nex_gddp_historical import calendar_ordinal, load_yaml, validate_payload
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject
from scipy.stats import wasserstein_distance

GRID = Path("01_inputs/processed/grid_full_extent/model_grid_EPSG6933_11000m.tif")
OUT = Path("06_validation/historical_gcm")
VARIABLES = {
    "pr": ("precipitation", 86400.0, 0.0, "mm d-1"),
    "tasmin": ("tmin", 1.0, -273.15, "degC"),
    "tasmax": ("tmax", 1.0, -273.15, "degC"),
    "rsds": ("solar_radiation", 0.0864, 0.0, "MJ m-2 d-1"),
}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def month_lengths(year, calendar):
    starts = [calendar_ordinal(year, month, 1, calendar) for month in range(1, 13)]
    return np.diff(starts + [calendar_ordinal(year + 1, 1, 1, calendar)])


def monthly_means(values, lengths, allow_static_mask=False):
    finite = np.isfinite(values)
    valid = (
        finite.all()
        if not allow_static_mask
        else (finite.all(axis=0) | (~finite).all(axis=0)).all()
    )
    if len(values) != sum(lengths) or not valid:
        raise ValueError("Incomplete or nonfinite daily data")
    return np.stack([block.mean(axis=0) for block in np.split(values, np.cumsum(lengths)[:-1])])


def metrics(reference, candidate):
    """Compare 15x12 monthly daily means; no date-paired weather skill score."""
    ref_cycle = reference.mean(axis=0)
    mod_cycle = candidate.mean(axis=0)
    corr = float("nan")
    if np.std(ref_cycle) > 0 and np.std(mod_cycle) > 0:
        corr = float(np.corrcoef(ref_cycle, mod_cycle)[0, 1])
    return {
        "reference_mean": float(reference.mean()),
        "model_mean": float(candidate.mean()),
        "bias": float(candidate.mean() - reference.mean()),
        "seasonal_cycle_rmse": float(np.sqrt(np.mean((mod_cycle - ref_cycle) ** 2))),
        "seasonal_cycle_correlation": corr,
        "monthly_distribution_wasserstein": float(
            wasserstein_distance(reference.ravel(), candidate.ravel())
        ),
        "monthly_p10_bias": float(np.quantile(candidate, 0.1) - np.quantile(reference, 0.1)),
        "monthly_p90_bias": float(np.quantile(candidate, 0.9) - np.quantile(reference, 0.9)),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_yaml(Path("config/gcms.yml"))
    manifest_path = Path("01_inputs/manifests/nex_gddp_historical_subset_manifest.yml")
    manifest = load_yaml(manifest_path)
    expected = {
        (m["id"], v, y)
        for m in cfg["candidate_models"]
        for v in VARIABLES
        for y in range(2000, 2015)
    }
    entries = manifest["files"]
    actual = {(e["model"], e["variable"], e["year"]) for e in entries}
    if actual != expected or len(entries) != len(expected):
        raise ValueError("Acquisition manifest is incomplete or contains duplicates")
    with rasterio.open(GRID) as grid:
        codes = grid.read(1)
        transform, crs, shape = grid.transform, grid.crs, grid.shape
    country_masks = {"GHA": codes == 1, "CIV": codes == 2}
    provenance = {str(GRID): digest(GRID), str(manifest_path): digest(manifest_path)}
    rows = []
    ag_manifest = load_yaml(Path("01_inputs/manifests/agera5_full_extent_processed_manifest.yml"))

    def index_records(node):
        if isinstance(node, dict):
            if "path" in node and "sha256" in node:
                yield node["path"], node["sha256"]
            for value in node.values():
                yield from index_records(value)
        elif isinstance(node, list):
            for value in node:
                yield from index_records(value)

    ag_hashes = dict(index_records(ag_manifest))
    for variable, (ag_name, scale, offset, units) in VARIABLES.items():
        for year in range(2000, 2015):
            path = Path(
                f"01_inputs/processed/climate_full_extent/daily/{ag_name}_{year}_EPSG6933_11000m.tif"
            )
            checksum = digest(path)
            if checksum != ag_hashes[path.as_posix()]:
                raise ValueError(f"Checksum mismatch: {path}")
            provenance[path.as_posix()] = checksum
            with rasterio.open(path) as src:
                lengths = month_lengths(year, "proleptic_gregorian")
                expected_dates = (
                    pd.date_range(f"{year}-01-01", f"{year}-12-31").strftime("%Y-%m-%d").tolist()
                )
                if (
                    src.transform != transform
                    or src.crs != crs
                    or src.shape != shape
                    or list(src.descriptions) != expected_dates
                    or src.tags().get("units") != units
                ):
                    raise ValueError(f"AgERA5 geometry/date/unit mismatch: {path}")
                data = src.read(masked=True).filled(np.nan)
            for country, mask in country_masks.items():
                means = monthly_means(data[:, mask], lengths).mean(axis=1)
                for month, value in enumerate(means, 1):
                    rows.append(
                        {
                            "model": "AgERA5",
                            "variable": variable,
                            "country": country,
                            "year": year,
                            "month": month,
                            "value": float(value),
                            "units": units,
                        }
                    )
        print(f"Reference ready: {variable}", flush=True)
    for index, entry in enumerate(entries, 1):
        path = Path(entry["path"])
        payload = path.read_bytes()
        checksum = hashlib.sha256(payload).hexdigest()
        if checksum != entry["sha256"]:
            raise ValueError(f"Checksum mismatch: {path}")
        variable, year = entry["variable"], entry["year"]
        metadata = validate_payload(payload, variable, year, (-9.0, 4.0, 4.0, 12.0))
        provenance[path.as_posix()] = checksum
        with xr.open_dataset(path, engine="scipy", decode_times=False) as ds:
            lon = (ds.lon.values + 180) % 360 - 180
            lat = ds.lat.values
            xi, yi = np.argsort(lon), np.argsort(-lat)
            values = ds[variable].transpose("time", "lat", "lon").values[:, yi][:, :, xi]
            _, scale, offset, units = VARIABLES[variable]
            monthly = monthly_means(
                values * scale + offset,
                month_lengths(year, metadata["calendar"]),
                allow_static_mask=True,
            )
            source_transform = from_origin(
                float(lon[xi][0] - 0.125), float(lat[yi][0] + 0.125), 0.25, 0.25
            )
        for month, source in enumerate(monthly, 1):
            target = np.full(shape, np.nan, dtype=np.float64)
            reproject(
                source,
                target,
                src_transform=source_transform,
                src_crs="EPSG:4326",
                dst_transform=transform,
                dst_crs=crs,
                src_nodata=np.nan,
                dst_nodata=np.nan,
                resampling=Resampling.average,
            )
            for country, mask in country_masks.items():
                if not np.isfinite(target[mask]).all():
                    raise ValueError(f"Missing country coverage: {path} {country}")
                rows.append(
                    {
                        "model": entry["model"],
                        "variable": variable,
                        "country": country,
                        "year": year,
                        "month": month,
                        "value": float(target[mask].mean()),
                        "units": units,
                    }
                )
        if index % 60 == 0:
            print(f"Validated and screened {index}/480 files", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "country_monthly.csv", index=False)
    summaries = []
    for (model, variable, country), group in frame[frame.model != "AgERA5"].groupby(
        ["model", "variable", "country"]
    ):
        ref = (
            frame[
                (frame.model == "AgERA5")
                & (frame.variable == variable)
                & (frame.country == country)
            ]
            .sort_values(["year", "month"])
            .value.to_numpy()
            .reshape(15, 12)
        )
        candidate = group.sort_values(["year", "month"]).value.to_numpy().reshape(15, 12)
        summaries.append(
            dict(
                model=model,
                variable=variable,
                country=country,
                units=group.units.iloc[0],
                **metrics(ref, candidate),
            )
        )
    pd.DataFrame(summaries).to_csv(OUT / "country_diagnostics.csv", index=False)
    audit = {
        "completed_utc": datetime.now(UTC).isoformat(),
        "validated_nex_files": len(entries),
        "reference_files": 60,
        "country_cells": {k: int(v.sum()) for k, v in country_masks.items()},
        "input_sha256": provenance,
        "script_sha256": digest(__file__),
        "outputs": {
            name: digest(OUT / name) for name in ["country_monthly.csv", "country_diagnostics.csv"]
        },
    }
    (OUT / "screening_manifest.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print("Historical country screening complete.", flush=True)


if __name__ == "__main__":
    main()
