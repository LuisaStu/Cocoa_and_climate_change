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

# Historical GCM screening
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


def historical_main():
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

# Daily extremes screening
EXTREMES_OUT = Path("06_validation/historical_gcm/daily_spatial")
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


def extremes_main():
    EXTREMES_OUT.mkdir(parents=True, exist_ok=True)
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
        path = EXTREMES_OUT / f"{model}_{variable}_climatology.tif"
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
        path = EXTREMES_OUT / name
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
    (EXTREMES_OUT / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Daily/spatial screening complete.", flush=True)


def main(argv=None):
    import argparse
    import sys

    parser = argparse.ArgumentParser(description='Run historical or daily-extremes climate-model screening.')
    parser.add_argument("operation", choices=('historical', 'extremes'))
    args, forwarded = parser.parse_known_args(argv)
    if argv is None:
        sys.argv = [sys.argv[0], *forwarded]
    if args.operation == 'historical':
        historical_main()
    elif args.operation == 'extremes':
        extremes_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
