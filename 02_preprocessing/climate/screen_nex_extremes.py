"""Daily distribution and spatial screening on the fixed study grid."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject
from screen_nex_historical import GRID, VARIABLES, digest, load_yaml, month_lengths

OUT = Path("06_validation/historical_gcm/daily_spatial")
COCOA = Path("01_inputs/processed/cocoa_mask_full_extent/cocoa_fraction_EPSG6933_11000m.tif")


def annual_indices(data, variable):
    """Time x valid-cell data; dry spells are explicitly truncated at year edges."""
    if data.ndim != 2 or not np.isfinite(data).all():
        raise ValueError("Expected finite time-by-cell data")
    if variable in {"pr", "rsds"} and (data < 0).any():
        raise ValueError("Negative precipitation or radiation")
    result = {
        "mean": data.mean(axis=0),
        "p05": np.quantile(data, 0.05, axis=0),
        "p95": np.quantile(data, 0.95, axis=0),
        "annual_max": data.max(axis=0),
    }
    if variable == "pr":
        result["wet_day_fraction"] = (data >= 1.0).mean(axis=0)
        run = np.zeros(data.shape[1], dtype=int)
        longest = run.copy()
        for day in data:
            run = np.where(day < 1.0, run + 1, 0)
            longest = np.maximum(longest, run)
        result["longest_dry_spell_within_year"] = longest.astype(float)
    if variable == "tasmax":
        result["fraction_days_ge35C"] = (data >= 35.0).mean(axis=0)
    return result


def spatial_metrics(reference, candidate, weights):
    valid = weights > 0
    a, b, w = reference[valid], candidate[valid], weights[valid]
    if not len(w) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Missing weighted spatial coverage")
    w = w / w.sum()
    am, bm = np.sum(w * a), np.sum(w * b)
    av, bv = np.sum(w * (a - am) ** 2), np.sum(w * (b - bm) ** 2)
    corr = np.sum(w * (a - am) * (b - bm)) / np.sqrt(av * bv) if av * bv > 1e-20 else np.nan
    return {
        "reference": float(am),
        "model_value": float(bm),
        "bias": float(bm - am),
        "spatial_rmse": float(np.sqrt(np.sum(w * (b - a) ** 2))),
        "spatial_correlation": float(corr),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    prior_path = Path("06_validation/historical_gcm/screening_manifest.json")
    prior = json.loads(prior_path.read_text())
    prior["input_sha256"] = {Path(k).as_posix(): v for k, v in prior["input_sha256"].items()}
    manifest = load_yaml(Path("01_inputs/manifests/nex_gddp_historical_subset_manifest.yml"))
    cfg = load_yaml(Path("config/gcms.yml"))
    models = [x["id"] for x in cfg["candidate_models"]]
    expected = {(m, v, y) for m in models for v in VARIABLES for y in range(2000, 2015)}
    entries = {(e["model"], e["variable"], e["year"]): e for e in manifest["files"]}
    if set(entries) != expected or len(manifest["files"]) != len(expected):
        raise ValueError("Incomplete/duplicate NEX manifest")
    provenance = {}

    def verify(path):
        value = digest(path)
        if value != prior["input_sha256"][Path(path).as_posix()]:
            raise ValueError(f"Input changed since validated monthly screening: {path}")
        provenance[Path(path).as_posix()] = value

    verify(GRID)
    with rasterio.open(GRID) as src:
        codes, profile = src.read(1), src.profile
        transform, crs, shape = src.transform, src.crs, src.shape
    active = np.isin(codes, [1, 2])
    cocoa_manifest = load_yaml(Path("01_inputs/manifests/cocoa_mask_abu_manifest.yml"))
    cocoa_checksums = {Path(k).as_posix(): v for k, v in cocoa_manifest["outputs"].items()}
    if digest(COCOA) != cocoa_checksums[COCOA.as_posix()]:
        raise ValueError("Cocoa checksum mismatch")
    provenance[COCOA.as_posix()] = digest(COCOA)
    with rasterio.open(COCOA) as src:
        if src.shape != shape or src.transform != transform or src.crs != crs:
            raise ValueError("Cocoa grid mismatch")
        cocoa = src.read(1, masked=True).filled(np.nan)
    if ((cocoa[np.isfinite(cocoa)] < 0) | (cocoa[np.isfinite(cocoa)] > 1)).any():
        raise ValueError("Invalid cocoa fractions")
    weights = {}
    for country, code in [("GHA", 1), ("CIV", 2)]:
        weights[country, "country"] = (codes[active] == code).astype(float)
        weights[country, "cocoa"] = np.where(
            (codes[active] == code) & np.isfinite(cocoa[active]), cocoa[active], 0.0
        )
    annual_rows, summary_rows, metadata_rows, outputs = [], [], [], []

    def record(model, variable, year, indices):
        for metric, field in indices.items():
            for (country, domain), w in weights.items():
                annual_rows.append(
                    {
                        "model": model,
                        "variable": variable,
                        "year": year,
                        "metric": metric,
                        "country": country,
                        "domain": domain,
                        "value": float(np.average(field, weights=w)),
                    }
                )

    def save_fields(model, variable, indices):
        path = OUT / f"{model}_{variable}_climatology.tif"
        p = profile.copy()
        p.update(dtype="float32", nodata=-9999.0, count=len(indices), compress="deflate")
        with rasterio.open(path, "w", **p) as dst:
            for band, (metric, field) in enumerate(indices.items(), 1):
                arr = np.full(shape, -9999.0, dtype="float32")
                arr[active] = field
                dst.write(arr, band)
                dst.set_band_description(band, metric)
            dst.update_tags(
                period="2000-2014",
                aggregation="Equal-year mean of annual cell indices",
                variable=variable,
            )
        outputs.append(path)

    for variable, (ag_name, scale, offset, units) in VARIABLES.items():
        reference_years = []
        for year in range(2000, 2015):
            path = Path(
                f"01_inputs/processed/climate_full_extent/daily/{ag_name}_{year}_EPSG6933_11000m.tif"
            )
            verify(path)
            with rasterio.open(path) as src:
                data = src.read(masked=True).filled(np.nan)[:, active]
            indices = annual_indices(data, variable)
            reference_years.append(indices)
            record("AgERA5", variable, year, indices)
        reference = {
            k: np.mean([x[k] for x in reference_years], axis=0) for k in reference_years[0]
        }
        save_fields("AgERA5", variable, reference)
        for model in models:
            years = []
            for year in range(2000, 2015):
                e = entries[model, variable, year]
                path = Path(e["path"])
                verify(path)
                with xr.open_dataset(path, engine="scipy", decode_times=False) as ds:
                    version = str(ds.attrs.get("version", ""))
                    if version != "2.0":
                        raise ValueError(f"Unexpected source version: {path}")
                    calendar = ds.time.attrs.get("calendar", "standard")
                    lon = (ds.lon.values + 180) % 360 - 180
                    lat = ds.lat.values
                    xi, yi = np.argsort(lon), np.argsort(-lat)
                    data = (
                        ds[variable].transpose("time", "lat", "lon").values[:, yi][:, :, xi] * scale
                        + offset
                    )
                    if len(data) != sum(month_lengths(year, calendar)):
                        raise ValueError("Unexpected year length")
                    finite = np.isfinite(data)
                    if not (finite.all(axis=0) | (~finite).all(axis=0)).all():
                        raise ValueError("Intermittent source gaps")
                    metadata_rows.append(
                        {
                            "model": model,
                            "variable": variable,
                            "year": year,
                            "version": version,
                            "calendar": calendar,
                            "cmip6_license": ds.attrs.get("cmip6_license", "missing"),
                        }
                    )
                    st = from_origin(
                        float(lon[xi][0] - 0.125), float(lat[yi][0] + 0.125), 0.25, 0.25
                    )
                target = np.full((len(data), *shape), np.nan, dtype="float32")
                reproject(
                    data,
                    target,
                    src_transform=st,
                    src_crs="EPSG:4326",
                    dst_transform=transform,
                    dst_crs=crs,
                    src_nodata=np.nan,
                    dst_nodata=np.nan,
                    resampling=Resampling.average,
                )
                indices = annual_indices(target[:, active], variable)
                years.append(indices)
                record(model, variable, year, indices)
            climatology = {k: np.mean([x[k] for x in years], axis=0) for k in years[0]}
            save_fields(model, variable, climatology)
            for metric, field in climatology.items():
                metric_units = (
                    "fraction" if "fraction" in metric else "days" if "spell" in metric else units
                )
                for (country, domain), w in weights.items():
                    summary_rows.append(
                        {
                            "model": model,
                            "variable": variable,
                            "metric": metric,
                            "country": country,
                            "domain": domain,
                            "units": metric_units,
                            **spatial_metrics(reference[metric], field, w),
                        }
                    )
            print(f"Completed daily/spatial: {model} {variable}", flush=True)
    for name, rows in [
        ("annual_indices.csv", annual_rows),
        ("spatial_diagnostics.csv", summary_rows),
        ("source_metadata.csv", metadata_rows),
    ]:
        path = OUT / name
        pd.DataFrame(rows).to_csv(path, index=False)
        outputs.append(path)
    report = {
        "completed_utc": datetime.now(UTC).isoformat(),
        "input_sha256": provenance,
        "script_sha256": digest(__file__),
        "dependencies_sha256": {
            "screen_nex_historical.py": digest(Path(__file__).with_name("screen_nex_historical.py"))
        },
        "outputs": {p.name: digest(p) for p in outputs},
        "missing_cocoa_country_cells": {
            c: int(((codes == code) & ~np.isfinite(cocoa)).sum())
            for c, code in [("GHA", 1), ("CIV", 2)]
        },
        "country_cells": int(active.sum()),
    }
    (OUT / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Daily/spatial screening complete.", flush=True)


if __name__ == "__main__":
    main()
