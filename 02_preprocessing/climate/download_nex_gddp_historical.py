"""Download spatially subsetted NEX-GDDP-CMIP6 historical screening data."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import xarray as xr
import yaml  # type: ignore[import-untyped]

DEFAULT_STUDY_CONFIG = Path("config/study.yml")
DEFAULT_GCM_CONFIG = Path("config/gcms.yml")
DEFAULT_OUTPUT = Path("01_inputs/raw/nex_gddp_cmip6_historical_subset")
NCSS_BASE = "https://ds.nccs.nasa.gov/thredds/ncss/grid/AMES/NEX/GDDP-CMIP6"
VARIABLE_UNITS = {
    "hurs": "%",
    "sfcWind": "m s-1",
    "pr": "kg m-2 s-1",
    "rsds": "W m-2",
    "tasmax": "K",
    "tasmin": "K",
}
MAX_ATTEMPTS = 3
TIMEOUT_SECONDS = 180


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, dict):
        raise TypeError(f"Expected a YAML mapping in {path}")
    return document


def historical_bbox(study_config: dict[str, Any]) -> tuple[float, float, float, float]:
    bbox = study_config["domain"]["bbox_lonlat"]
    return (
        float(bbox["west"]),
        float(bbox["south"]),
        float(bbox["east"]),
        float(bbox["north"]),
    )


def ncss_url(
    model: str,
    scenario: str,
    member: str,
    variable: str,
    year: int,
    bbox: tuple[float, float, float, float],
) -> str:
    if variable not in VARIABLE_UNITS:
        raise ValueError(f"Unsupported NEX-GDDP-CMIP6 variable: {variable}")
    west, south, east, north = bbox
    if not (-180 <= west < east <= 180 and -60 <= south < north <= 90):
        raise ValueError(f"Invalid NEX-GDDP-CMIP6 bounding box: {bbox}")
    if scenario not in {"historical", "ssp245", "ssp585"}:
        raise ValueError(f"Unsupported NEX-GDDP-CMIP6 scenario: {scenario}")
    filename = f"{variable}_day_{model}_{scenario}_{member}_gn_{year}_v2.0.nc"
    dataset_path = f"{model}/{scenario}/{member}/{variable}/{filename}"
    query = urlencode(
        {
            "var": variable,
            "north": north,
            "west": west,
            "east": east,
            "south": south,
            "horizStride": 1,
            "time_start": f"{year}-01-01T12:00:00Z",
            "time_end": f"{year}-12-31T12:00:00Z",
            "accept": "netcdf3",
            "addLatLon": "true",
        }
    )
    return f"{NCSS_BASE}/{dataset_path}?{query}"


def expected_days(year: int, calendar: str) -> int:
    if calendar in {"365_day", "noleap"}:
        return 365
    if calendar in {"360_day"}:
        return 360
    if calendar in {"366_day", "all_leap"}:
        return 366
    if calendar == "julian":
        return 366 if year % 4 == 0 else 365
    if calendar in {"standard", "gregorian", "proleptic_gregorian"}:
        return (date(year + 1, 1, 1) - date(year, 1, 1)).days
    raise ValueError(f"Unsupported NEX-GDDP-CMIP6 calendar: {calendar!r}")


def calendar_ordinal(year: int, month: int, day: int, calendar: str) -> int:
    if calendar in {"standard", "gregorian", "proleptic_gregorian"}:
        return date(year, month, day).toordinal()
    if calendar == "julian":
        leap_year = year % 4 == 0
        month_lengths = [31, 29 if leap_year else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        days_before_year = 365 * (year - 1) + (year - 1) // 4
    elif calendar in {"365_day", "noleap"}:
        month_lengths = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        days_before_year = 365 * (year - 1)
    elif calendar == "360_day":
        month_lengths = [30] * 12
        days_before_year = 360 * (year - 1)
    elif calendar in {"366_day", "all_leap"}:
        month_lengths = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        days_before_year = 366 * (year - 1)
    else:
        raise ValueError(f"Unsupported NEX-GDDP-CMIP6 calendar: {calendar!r}")
    if month < 1 or month > 12 or day < 1 or day > month_lengths[month - 1]:
        raise ValueError(f"Invalid date for calendar {calendar}: {year}-{month:02d}-{day:02d}")
    return days_before_year + sum(month_lengths[: month - 1]) + day - 1


def validate_payload(
    payload: bytes,
    variable: str,
    year: int,
    bbox: tuple[float, float, float, float],
) -> dict[str, Any]:
    if not payload.startswith((b"CDF\x01", b"CDF\x02")):
        raise ValueError(f"NCSS response is not a NetCDF3 file for {variable} {year}")
    west, south, east, north = bbox
    with xr.open_dataset(BytesIO(payload), engine="scipy", decode_times=False) as dataset:
        if variable not in dataset:
            raise ValueError(f"NCSS file is missing variable {variable} for {year}")
        if not {"time", "lat", "lon"}.issubset(dataset.coords):
            raise ValueError(f"NCSS file lacks time/lat/lon coordinates for {variable} {year}")
        data = dataset[variable]
        units = str(data.attrs.get("units", "")).strip()
        if units != VARIABLE_UNITS[variable]:
            raise ValueError(
                f"Unexpected {variable} units in {year}: {units!r}; "
                f"expected {VARIABLE_UNITS[variable]!r}"
            )
        if set(data.dims) != {"time", "lat", "lon"}:
            raise ValueError(f"Unexpected {variable} dimensions in {year}: {data.dims}")
        time_coordinate = dataset["time"]
        calendar = str(time_coordinate.attrs.get("calendar", "standard")).lower()
        time_units = str(time_coordinate.attrs.get("units", "")).strip()
        if calendar not in {
            "standard",
            "gregorian",
            "proleptic_gregorian",
            "julian",
            "365_day",
            "noleap",
            "360_day",
            "366_day",
            "all_leap",
        }:
            raise ValueError(f"Unsupported NEX-GDDP-CMIP6 calendar: {calendar!r}")
        origin_match = re.match(
            r"^days since (\d{4})-(\d{1,2})-(\d{1,2})(?:[ T].*)?$",
            time_units,
        )
        if origin_match is None:
            raise ValueError(f"Unexpected time units for {variable} {year}: {time_units!r}")
        origin_year, origin_month, origin_day = map(int, origin_match.groups())
        expected_start = (
            calendar_ordinal(year, 1, 1, calendar)
            - calendar_ordinal(origin_year, origin_month, origin_day, calendar)
            + 0.5
        )
        time_values = np.asarray(time_coordinate.values, dtype=np.float64)
        day_count = expected_days(year, calendar)
        if time_values.size != day_count:
            raise ValueError(
                f"Unexpected number of daily records for {variable} {year}: "
                f"{time_values.size}; expected {day_count} for {calendar}"
            )
        if (
            not np.isfinite(time_values).all()
            or not np.isclose(time_values[0], expected_start)
            or not np.isclose(time_values[-1], expected_start + day_count - 1)
        ):
            raise ValueError(
                f"Unexpected daily time coverage for {variable} {year} "
                f"using {calendar} calendar: "
                f"{time_values[0] if time_values.size else 'empty'} through "
                f"{time_values[-1] if time_values.size else 'empty'}"
            )
        if not np.allclose(np.diff(time_values), 1.0, rtol=0.0, atol=1e-8):
            raise ValueError(f"Non-daily or unordered time coordinate for {variable} {year}")
        latitudes = np.asarray(dataset["lat"].values, dtype=np.float64)
        longitudes = np.asarray(dataset["lon"].values, dtype=np.float64)
        normalized_longitudes = (longitudes + 180.0) % 360.0 - 180.0
        if latitudes.ndim != 1 or longitudes.ndim != 1:
            raise ValueError(f"Expected 1-D latitude/longitude coordinates for {variable} {year}")
        if not np.allclose(np.abs(np.diff(latitudes)), 0.25) or not np.allclose(
            np.abs(np.diff(longitudes)), 0.25
        ):
            raise ValueError(f"Unexpected spatial resolution in {variable} {year}")
        tolerance = 0.25
        if (
            latitudes.min() < south - tolerance
            or latitudes.max() > north + tolerance
            or normalized_longitudes.min() < west - tolerance
            or normalized_longitudes.max() > east + tolerance
        ):
            raise ValueError(
                f"NCSS returned coordinates outside requested bounds for {variable} {year}"
            )
        if not np.isfinite(data.values).any():
            raise ValueError(f"NCSS returned no finite {variable} values for {year}")
        return {
            "variable": variable,
            "year": year,
            "units": units,
            "days": int(time_values.size),
            "calendar": calendar,
            "time_units": time_units,
            "latitude_count": int(latitudes.size),
            "longitude_count": int(longitudes.size),
            "latitude_bounds": [float(latitudes.min()), float(latitudes.max())],
            "longitude_bounds": [
                float(normalized_longitudes.min()),
                float(normalized_longitudes.max()),
            ],
            "longitude_convention": "0-360 normalized to -180-180 for bounds validation",
        }


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def output_path(
    root: Path,
    model: str,
    member: str,
    variable: str,
    year: int,
) -> Path:
    filename = f"{variable}_day_{model}_historical_{member}_gn_{year}_v2.0.nc"
    return root / model / "historical" / member / variable / filename


def download_one(
    root: Path,
    model: str,
    member: str,
    variable: str,
    year: int,
    bbox: tuple[float, float, float, float],
) -> dict[str, Any]:
    destination = output_path(root, model, member, variable, year)
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = ncss_url(model, "historical", member, variable, year, bbox)
    if destination.is_file():
        payload = destination.read_bytes()
        metadata = validate_payload(payload, variable, year, bbox)
        return {
            **metadata,
            "path": destination.as_posix(),
            "sha256": sha256(payload),
            "size_bytes": len(payload),
            "source_url": url,
        }

    request = Request(url, headers={"User-Agent": "SIMPLE-Cacao-WA research workflow"})
    last_error: URLError | TimeoutError | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            with urlopen(request, timeout=TIMEOUT_SECONDS) as response:
                payload = response.read()
            metadata = validate_payload(payload, variable, year, bbox)
            temporary = destination.with_suffix(destination.suffix + ".part")
            temporary.write_bytes(payload)
            temporary.replace(destination)
            return {
                **metadata,
                "path": destination.as_posix(),
                "sha256": sha256(payload),
                "size_bytes": len(payload),
                "source_url": url,
            }
        except (URLError, TimeoutError) as error:
            last_error = error
            if attempt < MAX_ATTEMPTS:
                time.sleep(attempt * 2)
    if last_error is None:
        raise RuntimeError(f"Download failed without a reported network error: {url}")
    raise RuntimeError(f"Download failed after {MAX_ATTEMPTS} attempts: {url}") from last_error


def write_manifest(
    path: Path,
    entries: list[dict[str, Any]],
    bbox: tuple[float, float, float, float],
    models: list[str],
    variables: list[str],
    start_year: int,
    end_year: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    west, south, east, north = bbox
    manifest = {
        "status": "complete_for_requested_selection",
        "source": "NASA NEX-GDDP-CMIP6 v2.0 via NCCS THREDDS NCSS",
        "scenario": "historical",
        "requested_selection": {
            "models": models,
            "variables": variables,
            "start_year": start_year,
            "end_year": end_year,
        },
        "spatial_subset_bbox_lonlat": {
            "west": west,
            "south": south,
            "east": east,
            "north": north,
        },
        "selection_use": (
            "Historical GCM skill screening only; these are spatial subsets "
            "at the source 0.25-degree grid."
        ),
        "files": sorted(entries, key=lambda item: item["path"]),
    }
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        yaml.safe_dump(manifest, stream, sort_keys=False)


def download_screening_period(
    study_config_path: Path,
    gcm_config_path: Path,
    output: Path,
    manifest_path: Path,
    start_year: int | None = None,
    end_year: int | None = None,
    models: list[str] | None = None,
    variables: list[str] | None = None,
    workers: int = 4,
) -> None:
    study_config = load_yaml(study_config_path)
    gcm_config = load_yaml(gcm_config_path)
    configured_period = gcm_config["study"]["historical_comparison_years"]
    first_year = int(configured_period[0]) if start_year is None else start_year
    last_year = int(configured_period[1]) if end_year is None else end_year
    if first_year > last_year:
        raise ValueError("Start year must not be later than end year")
    if first_year < int(configured_period[0]) or last_year > int(configured_period[1]):
        raise ValueError(
            "Requested years must fall within the configured historical comparison period "
            f"{configured_period[0]}-{configured_period[1]}"
        )
    if workers < 1 or workers > 8:
        raise ValueError("Worker count must be between 1 and 8")
    candidates = gcm_config["candidate_models"]
    model_members = {item["id"]: item["member"] for item in candidates}
    selected_models = list(dict.fromkeys(models or list(model_members)))
    unknown_models = sorted(set(selected_models) - set(model_members))
    if unknown_models:
        raise ValueError(f"Unknown candidate GCM(s): {unknown_models}")
    selected_variables = list(
        dict.fromkeys(variables or list(gcm_config["study"]["required_variables"]))
    )
    unknown_variables = sorted(set(selected_variables) - set(VARIABLE_UNITS))
    if unknown_variables:
        raise ValueError(f"Unsupported screening variable(s): {unknown_variables}")
    bbox = historical_bbox(study_config)
    jobs = list(
        itertools.product(
            selected_models,
            selected_variables,
            range(first_year, last_year + 1),
        )
    )
    print(
        f"Downloading {len(jobs)} historical spatial subsets "
        f"({first_year}-{last_year}; {len(selected_models)} GCMs; "
        f"{len(selected_variables)} variables) to {output}",
        flush=True,
    )
    entries = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                download_one,
                output,
                model,
                model_members[model],
                variable,
                year,
                bbox,
            ): (model, variable, year)
            for model, variable, year in jobs
        }
        for index, future in enumerate(as_completed(futures), start=1):
            model, variable, year = futures[future]
            entry = {"model": model, **future.result()}
            entries.append(entry)
            print(
                f"[{index}/{len(jobs)}] {model} {variable} {year}: "
                f"{entry['size_bytes']} bytes sha256={entry['sha256']}",
                flush=True,
            )
    write_manifest(
        manifest_path,
        entries,
        bbox,
        selected_models,
        selected_variables,
        first_year,
        last_year,
    )
    print(f"Wrote acquisition manifest: {manifest_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-config", type=Path, default=DEFAULT_STUDY_CONFIG)
    parser.add_argument("--gcm-config", type=Path, default=DEFAULT_GCM_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("01_inputs/manifests/nex_gddp_historical_subset_manifest.yml"),
    )
    parser.add_argument("--start-year", type=int)
    parser.add_argument("--end-year", type=int)
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--variable", action="append", dest="variables")
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    download_screening_period(
        args.study_config,
        args.gcm_config,
        args.output,
        args.manifest,
        args.start_year,
        args.end_year,
        args.models,
        args.variables,
        args.workers,
    )


if __name__ == "__main__":
    main()
