from __future__ import annotations

import hashlib
import json
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import rasterio
import xarray as xr
from download_nex_gddp_historical import load_yaml
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject, transform
from screen_nex_historical import GRID, VARIABLES, month_lengths


# Weather-stream reader
# Weather-stream reader
def grid_cells():
    with rasterio.open(GRID) as src:
        codes = src.read(1)
        active = np.isin(codes, [1, 2])
        rows, cols = np.where(active)
        x, y = rasterio.transform.xy(src.transform, rows, cols)
        lon, lat = transform(src.crs, "EPSG:4326", x, y)
        return {
            "active": active,
            "codes": codes[active],
            "cell_ids": np.flatnonzero(active),
            "latitude": np.asarray(lat),
            "longitude": np.asarray(lon),
            "shape": src.shape,
            "transform": src.transform,
            "crs": src.crs,
        }


def year_blocks(model, scenario, year, block_days=31):
    if not 1 <= block_days <= 31:
        raise ValueError("Block size must be 1–31 days")
    if scenario == "historical":
        if not 2000 <= year <= 2014:
            raise ValueError("Historical year outside acquired period")
        manifest = load_yaml(Path("01_inputs/manifests/nex_gddp_historical_subset_manifest.yml"))
    elif scenario in {"ssp245", "ssp585"}:
        if not 2060 <= year <= 2080:
            raise ValueError("Scenario year outside acquired period")
        manifest_path = Path("01_inputs/manifests/nex_gddp_scenario_full.json")
        audit = json.loads(
            Path("06_validation/historical_gcm/scenario_full_audit.json").read_text()
        )
        if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != audit["manifest_sha256"]:
            raise ValueError("Scenario manifest differs from completed audit")
        manifest = json.loads(manifest_path.read_text())
    else:
        raise ValueError("Unsupported scenario")
    records = {
        e["variable"]: e
        for e in manifest["files"]
        if e["model"] == model and e["year"] == year and e.get("scenario", "historical") == scenario
    }
    if set(records) != set(VARIABLES):
        raise ValueError("Incomplete model-year variables")
    grid = grid_cells()
    names = {"pr": "precip_mm", "tasmin": "tmin_c", "tasmax": "tmax_c", "rsds": "srad_mj_m2"}
    with ExitStack() as stack:
        datasets = {}
        for variable, entry in records.items():
            with Path(entry["path"]).open("rb") as handle:
                if hashlib.file_digest(handle, "sha256").hexdigest() != entry["sha256"]:
                    raise ValueError(f"Input checksum mismatch: {entry['path']}")
            ds = stack.enter_context(
                xr.open_dataset(entry["path"], engine="scipy", decode_times=False, cache=False)
            )
            datasets[variable] = ds
        reference = datasets["tasmax"]
        calendar = reference.time.attrs.get("calendar", "standard")
        lengths = month_lengths(year, calendar)
        count = int(sum(lengths))
        if calendar == "360_day":
            raise ValueError("360-day solar-calendar mapping needs an explicit design")
        for ds in datasets.values():
            if (
                ds.time.size != count
                or any(
                    ds.time.attrs.get(key) != reference.time.attrs.get(key)
                    for key in ["units", "calendar"]
                )
                or any(not np.array_equal(ds[k], reference[k]) for k in ["time", "lat", "lon"])
            ):
                raise ValueError("Variables do not share coordinates/calendar")
        lon = (reference.lon.values + 180) % 360 - 180
        lat = reference.lat.values
        xi, yi = np.argsort(lon), np.argsort(-lat)
        st = from_origin(float(lon[xi][0] - 0.125), float(lat[yi][0] + 0.125), 0.25, 0.25)
        months = np.repeat(np.arange(1, 13), lengths)
        for first in range(0, count, block_days):
            last = min(first + block_days, count)
            result = {
                "model": model,
                "scenario": scenario,
                "year": year,
                "calendar": calendar,
                "day_of_year": np.arange(first + 1, last + 1),
                "month": months[first:last],
                "cell_ids": grid["cell_ids"],
                "latitude": grid["latitude"],
            }
            for variable, ds in datasets.items():
                _, scale, offset, _ = VARIABLES[variable]
                source = (
                    ds[variable]
                    .isel(time=slice(first, last))
                    .transpose("time", "lat", "lon")
                    .values[:, yi][:, :, xi]
                    * scale
                    + offset
                )
                target = np.full((last - first, *grid["shape"]), np.nan, dtype="float32")
                reproject(
                    source,
                    target,
                    src_transform=st,
                    src_crs="EPSG:4326",
                    dst_transform=grid["transform"],
                    dst_crs=grid["crs"],
                    src_nodata=np.nan,
                    dst_nodata=np.nan,
                    resampling=Resampling.average,
                )
                result[names[variable]] = target[:, grid["active"]]
            if not all(np.isfinite(result[k]).all() for k in names.values()):
                raise ValueError("Missing country weather")
            if (
                np.any(result["tmin_c"] > result["tmax_c"])
                or np.any(result["precip_mm"] < 0)
                or np.any(result["srad_mj_m2"] < 0)
            ):
                raise ValueError("Nonphysical weather")
            # ET0 is intentionally absent: a forcing provider must be chosen explicitly.
            yield result
