"""Compare CHIRPS v3 monthly precipitation with AgERA5 over Ghana and Cote d'Ivoire."""

from __future__ import annotations

import argparse
import calendar
import csv
import hashlib
import json
import math
import re
import urllib.request
import zipfile
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any

import h5py  # type: ignore[import-untyped]
import numpy as np
import rasterio  # type: ignore[import-untyped]
import xarray as xr
import yaml  # type: ignore[import-untyped]
from rasterio.features import rasterize  # type: ignore[import-untyped]
from rasterio.transform import from_origin  # type: ignore[import-untyped]
from rasterio.windows import Window  # type: ignore[import-untyped]

START_YEAR = 2000
END_YEAR = 2020
DEFAULT_AGERA5_DIRECTORY = Path("01_inputs/raw/agera5_full_extent")
DEFAULT_OUTPUT_DIRECTORY = Path("01_inputs/processed/validation_full_extent")
VARIABLE_NAME = "Precipitation_Flux"
FILL_VALUE = -9999.0
EARTH_RADIUS_M = 6_371_008.8
CHIRPS_URL = (
    "https://data.chc.ucsb.edu/products/CHIRPS/v3.0/monthly/africa/tifs/"
    "chirps-v3.0.{year}.{month:02d}.tif"
)
GEOBOUNDARIES_API = "https://www.geoboundaries.org/api/current/gbOpen/{iso3}/ADM0/"
BOUNDARIES = {
    "GHA": {
        "boundary_id": "GHA-ADM0-71095829",
        "boundary_source": "Stanford Digital Repository",
        "country_code": 1,
    },
    "CIV": {
        "boundary_id": "CIV-ADM0-2848817",
        "boundary_source": "Natural Earth",
        "country_code": 2,
    },
}


def axis_edges(centres: np.ndarray) -> np.ndarray:
    """Return cell edges for a regularly spaced coordinate vector."""
    values = np.asarray(centres, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("Grid coordinates must be one-dimensional with at least two cells")
    differences = np.diff(values)
    if not np.allclose(differences, differences[0], rtol=0.0, atol=1e-8):
        raise ValueError("Expected a regular grid coordinate")
    edges = np.empty(values.size + 1, dtype=np.float64)
    edges[1:-1] = (values[:-1] + values[1:]) / 2.0
    edges[0] = values[0] - differences[0] / 2.0
    edges[-1] = values[-1] + differences[-1] / 2.0
    return edges


def spherical_rectangle_area(west: float, south: float, east: float, north: float) -> float:
    """Calculate a latitude/longitude rectangle's area on a spherical Earth."""
    return (
        EARTH_RADIUS_M**2
        * math.radians(east - west)
        * (math.sin(math.radians(north)) - math.sin(math.radians(south)))
    )


def _source_indices(
    west: float,
    south: float,
    east: float,
    north: float,
    transform: rasterio.Affine,
    width: int | None,
    height: int | None,
) -> tuple[int, int, int, int]:
    pixel_width = transform.a
    pixel_height = -transform.e
    epsilon = 1e-7
    col_start = math.floor((west - transform.c) / pixel_width + epsilon)
    col_end = math.ceil((east - transform.c) / pixel_width - epsilon)
    row_start = math.floor((transform.f - north) / pixel_height + epsilon)
    row_end = math.ceil((transform.f - south) / pixel_height - epsilon)
    col_start = max(0, col_start)
    row_start = max(0, row_start)
    if width is not None:
        col_end = min(width, col_end)
    if height is not None:
        row_end = min(height, row_end)
    return row_start, row_end, col_start, col_end


def area_weighted_chirps_to_grid(
    source: np.ndarray,
    source_transform: rasterio.Affine,
    window: Window,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate CHIRPS pixels by spherical overlap area onto the AgERA5 grid."""
    latitude_edges = axis_edges(latitudes)
    longitude_edges = axis_edges(longitudes)
    result = np.full((latitudes.size, longitudes.size), np.nan, dtype=np.float32)
    coverage = np.zeros_like(result)
    window_row = int(window.row_off)
    window_col = int(window.col_off)
    pixel_width = source_transform.a
    pixel_height = -source_transform.e

    for row, latitude in enumerate(latitudes):
        south = min(latitude_edges[row], latitude_edges[row + 1])
        north = max(latitude_edges[row], latitude_edges[row + 1])
        for col, longitude in enumerate(longitudes):
            west = min(longitude_edges[col], longitude_edges[col + 1])
            east = max(longitude_edges[col], longitude_edges[col + 1])
            total_area = spherical_rectangle_area(west, south, east, north)
            row_start, row_end, col_start, col_end = _source_indices(
                west,
                south,
                east,
                north,
                source_transform,
                None,
                None,
            )
            row_start = max(row_start, window_row)
            row_end = min(row_end, window_row + source.shape[0])
            col_start = max(col_start, window_col)
            col_end = min(col_end, window_col + source.shape[1])
            valid_weight = 0.0
            weighted_sum = 0.0
            for source_row in range(row_start, row_end):
                source_north = source_transform.f - source_row * pixel_height
                source_south = source_north - pixel_height
                overlap_north = min(north, source_north)
                overlap_south = max(south, source_south)
                if overlap_north <= overlap_south:
                    continue
                for source_col in range(col_start, col_end):
                    source_west = source_transform.c + source_col * pixel_width
                    source_east = source_west + pixel_width
                    overlap_west = max(west, source_west)
                    overlap_east = min(east, source_east)
                    if overlap_east <= overlap_west:
                        continue
                    value = float(source[source_row - window_row, source_col - window_col])
                    if not math.isfinite(value) or value == FILL_VALUE:
                        continue
                    area = spherical_rectangle_area(
                        overlap_west, overlap_south, overlap_east, overlap_north
                    )
                    valid_weight += area
                    weighted_sum += value * area
            if valid_weight > 0.0:
                result[row, col] = weighted_sum / valid_weight
            coverage[row, col] = min(valid_weight / total_area, 1.0)
    return result, coverage


def _read_json_url(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "simple-cacao-wa/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read()
    parsed = json.loads(payload)
    if not isinstance(parsed, dict):
        raise TypeError(f"Expected a JSON object from {url}")
    return parsed


def ensure_country_boundaries(directory: Path) -> dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for iso3, metadata in BOUNDARIES.items():
        output = directory / f"geoboundaries_{iso3}_ADM0.geojson"
        if output.is_file():
            document = json.loads(output.read_text(encoding="utf-8"))
        else:
            api = _read_json_url(GEOBOUNDARIES_API.format(iso3=iso3))
            if api.get("boundaryID") != metadata["boundary_id"]:
                raise RuntimeError(
                    f"geoBoundaries version for {iso3} changed: "
                    f"expected {metadata['boundary_id']}, received {api.get('boundaryID')!r}"
                )
            download_url = api.get("gjDownloadURL")
            if not isinstance(download_url, str):
                raise ValueError(f"geoBoundaries returned no GeoJSON URL for {iso3}")
            request = urllib.request.Request(
                download_url, headers={"User-Agent": "simple-cacao-wa/0.1"}
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = response.read()
            document = json.loads(payload)
            output.write_text(json.dumps(document), encoding="utf-8")
        if document.get("type") != "FeatureCollection" or not document.get("features"):
            raise ValueError(f"Invalid geoBoundaries FeatureCollection: {output}")
        paths[iso3] = output
    return paths


def country_grid(
    boundary_paths: dict[str, Path], latitudes: np.ndarray, longitudes: np.ndarray
) -> np.ndarray:
    lat_edges = axis_edges(latitudes)
    lon_edges = axis_edges(longitudes)
    transform = from_origin(
        lon_edges.min(),
        lat_edges.max(),
        abs(lon_edges[1] - lon_edges[0]),
        abs(lat_edges[1] - lat_edges[0]),
    )
    output = np.zeros((latitudes.size, longitudes.size), dtype=np.uint8)
    for iso3, metadata in BOUNDARIES.items():
        document = json.loads(boundary_paths[iso3].read_text(encoding="utf-8"))
        shapes = [
            (feature["geometry"], metadata["country_code"])
            for feature in document["features"]
            if feature.get("geometry")
        ]
        country = rasterize(
            shapes,
            out_shape=output.shape,
            transform=transform,
            fill=0,
            all_touched=False,
            dtype=np.uint8,
        )
        overlap = (output != 0) & (country != 0)
        if overlap.any():
            raise ValueError(f"Country boundary masks overlap for {iso3}")
        output[country != 0] = country[country != 0]
    if not np.any(output == 1) or not np.any(output == 2):
        raise ValueError("Country grid is missing Ghana or Cote d'Ivoire cells")
    return output


def _decode_attribute(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8")
    return str(value)


def load_agera5_monthly(
    archive_directory: Path, start_year: int, end_year: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    first_latitudes: np.ndarray | None = None
    first_longitudes: np.ndarray | None = None
    months = [(year, month) for year in range(start_year, end_year + 1) for month in range(1, 13)]
    totals: dict[tuple[int, int], np.ndarray] = {}
    valid_days: dict[tuple[int, int], np.ndarray] = {}
    expected_dates: dict[int, set[date]] = {
        year: {
            date(year, month, day)
            for month in range(1, 13)
            for day in range(1, calendar.monthrange(year, month)[1] + 1)
        }
        for year in range(start_year, end_year + 1)
    }
    found_dates: dict[int, set[date]] = {year: set() for year in expected_dates}

    for year in range(start_year, end_year + 1):
        archive_path = archive_directory / f"agera5_precipitation_{year}.zip"
        if not archive_path.is_file():
            raise FileNotFoundError(f"Missing AgERA5 archive: {archive_path}")
        with zipfile.ZipFile(archive_path) as archive:
            members = [name for name in archive.namelist() if name.lower().endswith(".nc")]
            if len(members) != len(expected_dates[year]):
                raise ValueError(
                    f"Unexpected daily NetCDF count in {archive_path.name}: {len(members)}"
                )
            for member in members:
                match = re.search(r"_(\d{8})_final", Path(member).name)
                if not match:
                    raise ValueError(f"Cannot read date from AgERA5 member name: {member}")
                date_text = match.group(1)
                day = date(int(date_text[:4]), int(date_text[4:6]), int(date_text[6:8]))
                if day.year != year or day in found_dates[year]:
                    raise ValueError(f"Duplicate or out-of-year AgERA5 date: {member}")
                found_dates[year].add(day)
                key = (day.year, day.month)
                with h5py.File(BytesIO(archive.read(member)), "r") as dataset:
                    if VARIABLE_NAME not in dataset:
                        raise ValueError(f"{VARIABLE_NAME} missing in {member}")
                    data = dataset[VARIABLE_NAME]
                    units = _decode_attribute(data.attrs.get("units", ""))
                    if units != "mm d-1":
                        raise ValueError(f"Unexpected AgERA5 precipitation units {units!r}")
                    latitudes = np.asarray(dataset["lat"][:], dtype=np.float64)
                    longitudes = np.asarray(dataset["lon"][:], dtype=np.float64)
                    if first_latitudes is None or first_longitudes is None:
                        first_latitudes = latitudes
                        first_longitudes = longitudes
                    elif not (
                        np.allclose(latitudes, first_latitudes)
                        and np.allclose(longitudes, first_longitudes)
                    ):
                        raise ValueError(f"AgERA5 grid changed in {member}")
                    values = np.asarray(data[0, :, :], dtype=np.float32)
                    valid = np.isfinite(values) & (values != FILL_VALUE)
                    if key not in totals:
                        totals[key] = np.zeros(values.shape, dtype=np.float64)
                        valid_days[key] = np.zeros(values.shape, dtype=np.uint16)
                    totals[key][valid] += values[valid]
                    valid_days[key][valid] += 1
        if found_dates[year] != expected_dates[year]:
            missing = sorted(expected_dates[year] - found_dates[year])
            raise ValueError(f"AgERA5 daily date coverage incomplete for {year}: {missing[:5]}")
        print(f"AgERA5 precipitation read for {year}", flush=True)

    if first_latitudes is None or first_longitudes is None:
        raise ValueError("No AgERA5 coordinates found")
    expected_days_by_month = np.array(
        [calendar.monthrange(year, month)[1] for year, month in months], dtype=np.uint16
    )
    agera5 = np.full(
        (len(months), first_latitudes.size, first_longitudes.size), np.nan, dtype=np.float32
    )
    valid_day_array = np.zeros(agera5.shape, dtype=np.uint16)
    for index, month_key in enumerate(months):
        day_count = expected_days_by_month[index]
        valid_day_array[index] = valid_days[month_key]
        complete = valid_days[month_key] == day_count
        agera5[index, complete] = totals[month_key][complete]
    return agera5, valid_day_array, first_latitudes, first_longitudes


def _target_window(
    dataset: rasterio.io.DatasetReader,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
) -> Window:
    lat_edges = axis_edges(latitudes)
    lon_edges = axis_edges(longitudes)
    row_start, row_end, col_start, col_end = _source_indices(
        float(lon_edges.min()),
        float(lat_edges.min()),
        float(lon_edges.max()),
        float(lat_edges.max()),
        dataset.transform,
        dataset.width,
        dataset.height,
    )
    if row_start >= row_end or col_start >= col_end:
        raise ValueError("AgERA5 target grid does not overlap the CHIRPS source raster")
    return Window(col_start, row_start, col_end - col_start, row_end - row_start)


def download_chirps_monthly(
    latitudes: np.ndarray, longitudes: np.ndarray, start_year: int, end_year: int
) -> tuple[np.ndarray, np.ndarray]:
    months = [(year, month) for year in range(start_year, end_year + 1) for month in range(1, 13)]
    values = np.full((len(months), latitudes.size, longitudes.size), np.nan, dtype=np.float32)
    area_fraction = np.zeros(values.shape, dtype=np.float32)
    with rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
        GDAL_HTTP_MAX_RETRY="3",
        GDAL_HTTP_RETRY_DELAY="2",
        GDAL_CACHEMAX=64,
    ):
        for index, (year, month) in enumerate(months):
            url = CHIRPS_URL.format(year=year, month=month)
            with rasterio.open(f"/vsicurl/{url}") as dataset:
                if dataset.crs is None or dataset.crs.to_epsg() != 4326:
                    raise ValueError(f"Unexpected CHIRPS CRS: {dataset.crs}")
                if not math.isclose(dataset.transform.a, 0.05, abs_tol=1e-8):
                    raise ValueError(f"Unexpected CHIRPS grid resolution: {dataset.transform.a}")
                window = _target_window(dataset, latitudes, longitudes)
                source = dataset.read(1, window=window)
                values[index], area_fraction[index] = area_weighted_chirps_to_grid(
                    source, dataset.transform, window, latitudes, longitudes
                )
            print(f"CHIRPS v3 monthly raster read: {year}-{month:02d}", flush=True)
    return values, area_fraction


def _pearson(x: np.ndarray, y: np.ndarray, axis: int = 0) -> np.ndarray:
    x_count = np.isfinite(x).sum(axis=axis, keepdims=True)
    y_count = np.isfinite(y).sum(axis=axis, keepdims=True)
    x_mean = np.full(x_count.shape, np.nan, dtype=np.float64)
    y_mean = np.full(y_count.shape, np.nan, dtype=np.float64)
    np.divide(
        np.nansum(x, axis=axis, keepdims=True),
        x_count,
        out=x_mean,
        where=x_count > 0,
    )
    np.divide(
        np.nansum(y, axis=axis, keepdims=True),
        y_count,
        out=y_mean,
        where=y_count > 0,
    )
    x_centered = np.where(np.isfinite(x), x - x_mean, 0.0)
    y_centered = np.where(np.isfinite(y), y - y_mean, 0.0)
    covariance = np.sum(x_centered * y_centered, axis=axis)
    x_variance = np.sum(x_centered**2, axis=axis)
    y_variance = np.sum(y_centered**2, axis=axis)
    denominator = np.sqrt(x_variance * y_variance)
    result = np.full(covariance.shape, np.nan, dtype=np.float64)
    np.divide(covariance, denominator, out=result, where=denominator > 0.0)
    return result


def calculate_cell_metrics(
    agera5: np.ndarray, chirps: np.ndarray, coverage: np.ndarray, years: int
) -> dict[str, np.ndarray]:
    result = {
        name: np.full((12, *agera5.shape[1:]), np.nan, dtype=np.float64)
        for name in ("bias", "percent_bias", "correlation", "mean_coverage")
    }
    result["valid_pairs"] = np.zeros((12, *agera5.shape[1:]), dtype=np.uint16)
    result["missing_agera5"] = np.zeros_like(result["valid_pairs"])
    result["missing_chirps"] = np.zeros_like(result["valid_pairs"])
    for month in range(12):
        ag_month = agera5[month::12]
        chirps_month = chirps[month::12]
        coverage_month = coverage[month::12]
        valid = np.isfinite(ag_month) & np.isfinite(chirps_month)
        count = valid.sum(axis=0)
        ag_sum = np.where(valid, ag_month, 0.0).sum(axis=0)
        bias_sum = np.where(valid, chirps_month - ag_month, 0.0).sum(axis=0)
        result["valid_pairs"][month] = count
        result["missing_agera5"][month] = years - np.isfinite(ag_month).sum(axis=0)
        result["missing_chirps"][month] = years - np.isfinite(chirps_month).sum(axis=0)
        result["mean_coverage"][month] = np.nanmean(coverage_month, axis=0)
        np.divide(
            bias_sum,
            count,
            out=result["bias"][month],
            where=count > 0,
        )
        np.divide(
            100.0 * bias_sum,
            ag_sum,
            out=result["percent_bias"][month],
            where=(count > 0) & (ag_sum != 0.0),
        )
        result["correlation"][month] = _pearson(
            np.where(valid, ag_month, np.nan),
            np.where(valid, chirps_month, np.nan),
            axis=0,
        )
    return result


def calculate_country_metrics(
    agera5: np.ndarray,
    chirps: np.ndarray,
    coverage: np.ndarray,
    cell_metrics: dict[str, np.ndarray],
    countries: np.ndarray,
    latitudes: np.ndarray,
) -> list[dict[str, Any]]:
    latitude_edges = axis_edges(latitudes)
    cell_area = np.array(
        [
            spherical_rectangle_area(
                0.0,
                min(latitude_edges[row], latitude_edges[row + 1]),
                1.0,
                max(latitude_edges[row], latitude_edges[row + 1]),
            )
            for row in range(latitudes.size)
        ]
    )[:, None]
    results: list[dict[str, Any]] = []
    for iso3, metadata in BOUNDARIES.items():
        country_mask = countries == metadata["country_code"]
        weights = np.broadcast_to(cell_area, country_mask.shape)
        total_area = float(weights[country_mask].sum())
        if total_area <= 0.0:
            raise ValueError(f"No analysis-grid area found for {iso3}")
        for month in range(12):
            ag_month = agera5[month::12]
            chirps_month = chirps[month::12]
            valid = np.isfinite(ag_month) & np.isfinite(chirps_month) & country_mask[None, :, :]
            pair_weights = weights[None, :, :] * valid
            weighted_ag = float(np.sum(np.where(valid, ag_month, 0.0) * pair_weights))
            weighted_chirps = float(np.sum(np.where(valid, chirps_month, 0.0) * pair_weights))
            weighted_error = weighted_chirps - weighted_ag
            valid_area_time = float(pair_weights.sum())
            bias = weighted_error / valid_area_time if valid_area_time > 0 else math.nan
            percent_bias = 100.0 * weighted_error / weighted_ag if weighted_ag != 0.0 else math.nan
            cell_valid = cell_metrics["valid_pairs"][month] > 0
            bias_weights = weights * country_mask * cell_valid
            bias_weight_sum = float(bias_weights.sum())
            mean_cell_bias = (
                float(np.nansum(cell_metrics["bias"][month] * bias_weights) / bias_weight_sum)
                if bias_weight_sum > 0.0
                else math.nan
            )
            correlations = cell_metrics["correlation"][month]
            correlation_mask = country_mask & np.isfinite(correlations)
            correlation_weight = weights * correlation_mask
            correlation_weight_sum = float(correlation_weight.sum())
            mean_cell_correlation = (
                float(np.nansum(correlations * correlation_weight) / correlation_weight_sum)
                if correlation_weight_sum > 0.0
                else math.nan
            )
            complete_cells = (
                cell_metrics["valid_pairs"][month] == ag_month.shape[0]
            ) & country_mask
            complete_area = float(weights[complete_cells].sum())
            country_ag_series = np.full(ag_month.shape[0], np.nan, dtype=np.float64)
            country_chirps_series = np.full_like(country_ag_series, np.nan)
            if complete_cells.any():
                complete_weights = weights[complete_cells]
                country_ag_series = np.average(
                    ag_month[:, complete_cells], axis=1, weights=complete_weights
                )
                country_chirps_series = np.average(
                    chirps_month[:, complete_cells], axis=1, weights=complete_weights
                )
            country_correlation = float(_pearson(country_ag_series, country_chirps_series).item())
            expected_pairs = int(country_mask.sum()) * ag_month.shape[0]
            valid_pairs = int(valid.sum())
            agera5_pairs = int((np.isfinite(ag_month) & country_mask[None, :, :]).sum())
            chirps_pairs = int((np.isfinite(chirps_month) & country_mask[None, :, :]).sum())
            chirps_coverage = coverage[month::12, :, :]
            results.append(
                {
                    "country_iso3": iso3,
                    "calendar_month": month + 1,
                    "analysis_grid_cells": int(country_mask.sum()),
                    "valid_grid_year_pairs": valid_pairs,
                    "missing_pair_fraction": (
                        1.0 - valid_pairs / expected_pairs if expected_pairs else math.nan
                    ),
                    "missing_agera5_fraction": (
                        1.0 - agera5_pairs / expected_pairs if expected_pairs else math.nan
                    ),
                    "missing_chirps_fraction": (
                        1.0 - chirps_pairs / expected_pairs if expected_pairs else math.nan
                    ),
                    "mean_chirps_valid_area_fraction": float(
                        np.nanmean(chirps_coverage[:, country_mask])
                    ),
                    "area_weighted_agera5_climatology_mm": (
                        weighted_ag / valid_area_time if valid_area_time else math.nan
                    ),
                    "area_weighted_chirps_climatology_mm": (
                        weighted_chirps / valid_area_time if valid_area_time else math.nan
                    ),
                    "area_weighted_bias_mm": bias,
                    "area_weighted_percent_bias_pct": percent_bias,
                    "area_weighted_mean_cell_bias_mm": mean_cell_bias,
                    "area_weighted_mean_cell_correlation": mean_cell_correlation,
                    "country_time_series_correlation": country_correlation,
                    "complete_cell_count": int(complete_cells.sum()),
                    "complete_cell_area_fraction": complete_area / total_area,
                }
            )
    return results


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_outputs(
    output_directory: Path,
    agera5: np.ndarray,
    agera5_valid_days: np.ndarray,
    chirps: np.ndarray,
    coverage: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    countries: np.ndarray,
    start_year: int,
    end_year: int,
) -> dict[str, Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    monthly_times = np.array(
        [
            np.datetime64(f"{year:04d}-{month:02d}-01")
            for year in range(start_year, end_year + 1)
            for month in range(1, 13)
        ],
        dtype="datetime64[ns]",
    )
    dataset = xr.Dataset(
        data_vars={
            "agera5_precipitation_mm": (("time", "lat", "lon"), agera5),
            "agera5_valid_days": (("time", "lat", "lon"), agera5_valid_days),
            "chirps_precipitation_mm": (("time", "lat", "lon"), chirps),
            "chirps_valid_area_fraction": (("time", "lat", "lon"), coverage),
            "country_code": (("lat", "lon"), countries),
        },
        coords={
            "time": monthly_times,
            "lat": latitudes,
            "lon": longitudes,
        },
        attrs={
            "title": "Monthly AgERA5 and CHIRPS v3 precipitation comparison",
            "period": f"{start_year}-{end_year}",
            "spatial_method": "Spherical overlap-area-weighted mean from 0.05 degree CHIRPS cells to AgERA5 cells",
            "country_code_mapping": "0=outside Ghana/Cote d'Ivoire, 1=GHA, 2=CIV",
            "calendar_note": "Monthly validation uses all days in each calendar month, including February 29 in leap years.",
        },
    )
    dataset["agera5_precipitation_mm"].attrs.update(
        units="mm month-1", long_name="AgERA5 monthly precipitation total"
    )
    dataset["agera5_valid_days"].attrs.update(
        units="days", long_name="Number of valid AgERA5 daily values used"
    )
    dataset["chirps_precipitation_mm"].attrs.update(
        units="mm month-1", long_name="Area-weighted CHIRPS v3 monthly precipitation"
    )
    dataset["chirps_valid_area_fraction"].attrs.update(
        units="1", long_name="Fraction of target AgERA5 cell area covered by valid CHIRPS pixels"
    )
    dataset["country_code"].attrs.update(
        flag_values=np.array([0, 1, 2], dtype=np.uint8),
        flag_meanings="outside Ghana Cote_d_Ivoire",
    )
    cube_path = output_directory / f"chirps_v3_agera5_monthly_{start_year}_{end_year}.nc"
    encoding = {
        name: {"zlib": True, "complevel": 4, "_FillValue": np.float32(FILL_VALUE)}
        for name in (
            "agera5_precipitation_mm",
            "chirps_precipitation_mm",
            "chirps_valid_area_fraction",
        )
    }
    encoding["agera5_valid_days"] = {"zlib": True, "complevel": 4}
    encoding["country_code"] = {"zlib": True, "complevel": 4}
    dataset.to_netcdf(cube_path, engine="h5netcdf", encoding=encoding)
    dataset.close()
    return {"cube": cube_path}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty diagnostics table: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(
    project_root: Path,
    start_year: int = START_YEAR,
    end_year: int = END_YEAR,
    output_directory: Path | None = None,
    agera5_directory: Path | None = None,
) -> dict[str, Any]:
    if start_year < 1981 or end_year < start_year:
        raise ValueError("CHIRPS validation requires years >= 1981 and end >= start")
    raw_agera5 = (
        agera5_directory
        if agera5_directory is not None
        else project_root / DEFAULT_AGERA5_DIRECTORY
    )
    boundary_paths = ensure_country_boundaries(project_root / "01_inputs" / "raw" / "boundaries")
    agera5, valid_days, latitudes, longitudes = load_agera5_monthly(
        raw_agera5, start_year, end_year
    )
    countries = country_grid(boundary_paths, latitudes, longitudes)
    chirps, valid_area_fraction = download_chirps_monthly(
        latitudes, longitudes, start_year, end_year
    )
    years = end_year - start_year + 1
    cell_metrics = calculate_cell_metrics(agera5, chirps, valid_area_fraction, years)
    if output_directory is None:
        output_directory = project_root / DEFAULT_OUTPUT_DIRECTORY
    output_directory.mkdir(parents=True, exist_ok=True)
    outputs = write_outputs(
        output_directory,
        agera5,
        valid_days,
        chirps,
        valid_area_fraction,
        latitudes,
        longitudes,
        countries,
        start_year,
        end_year,
    )

    cell_rows: list[dict[str, Any]] = []
    for month in range(12):
        for row, latitude in enumerate(latitudes):
            for col, longitude in enumerate(longitudes):
                cell_rows.append(
                    {
                        "latitude": float(latitude),
                        "longitude": float(longitude),
                        "country_code": int(countries[row, col]),
                        "calendar_month": month + 1,
                        "valid_year_pairs": int(cell_metrics["valid_pairs"][month, row, col]),
                        "missing_agera5_years": int(
                            cell_metrics["missing_agera5"][month, row, col]
                        ),
                        "missing_chirps_years": int(
                            cell_metrics["missing_chirps"][month, row, col]
                        ),
                        "mean_chirps_valid_area_fraction": float(
                            cell_metrics["mean_coverage"][month, row, col]
                        ),
                        "bias_mm_month": float(cell_metrics["bias"][month, row, col]),
                        "percent_bias_pct": float(cell_metrics["percent_bias"][month, row, col]),
                        "pearson_r": float(cell_metrics["correlation"][month, row, col]),
                    }
                )
    cell_path = output_directory / f"chirps_v3_cell_month_metrics_{start_year}_{end_year}.csv"
    _write_csv(cell_path, cell_rows)
    country_rows = calculate_country_metrics(
        agera5, chirps, valid_area_fraction, cell_metrics, countries, latitudes
    )
    country_path = output_directory / f"chirps_v3_country_month_metrics_{start_year}_{end_year}.csv"
    _write_csv(country_path, country_rows)
    outputs.update({"cell_metrics": cell_path, "country_metrics": country_path})
    checksums = {name: sha256_file(path) for name, path in outputs.items()}
    summary = {
        "period": f"{start_year}-{end_year}",
        "months_compared": int(agera5.shape[0]),
        "grid_shape": [int(latitudes.size), int(longitudes.size)],
        "country_grid_cells": {
            iso3: int(np.count_nonzero(countries == metadata["country_code"]))
            for iso3, metadata in BOUNDARIES.items()
        },
        "agera5_missing_cell_months": int(np.count_nonzero(~np.isfinite(agera5))),
        "chirps_missing_cell_months": int(np.count_nonzero(~np.isfinite(chirps))),
        "mean_chirps_area_coverage_fraction": float(np.mean(valid_area_fraction)),
        "source": {
            "product": "Final CHIRPS v3 monthly Africa GeoTIFF",
            "landing_page": "https://www.chc.ucsb.edu/data/chirps3",
            "download_url_pattern": CHIRPS_URL,
            "grid_resolution_degrees": 0.05,
            "units": "mm/month",
        },
        "country_boundary_ids": {
            iso3: {
                "boundary_id": metadata["boundary_id"],
                "boundary_source": metadata["boundary_source"],
                "path": boundary_paths[iso3].relative_to(project_root).as_posix(),
                "sha256": sha256_file(boundary_paths[iso3]),
            }
            for iso3, metadata in BOUNDARIES.items()
        },
        "study_extent_note": (
            "Country summaries use cells inside the full-country AgERA5 domain "
            f"({min(longitudes):.2f} to {max(longitudes):.2f} E, "
            f"{min(latitudes):.2f} to {max(latitudes):.2f} N)."
        ),
        "outputs": {
            name: {
                "path": path.relative_to(project_root).as_posix(),
                "sha256": checksums[name],
                "size_bytes": path.stat().st_size,
            }
            for name, path in outputs.items()
        },
    }
    summary_path = output_directory / f"chirps_v3_validation_summary_{start_year}_{end_year}.yml"
    summary_path.write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")
    print(yaml.safe_dump(summary, sort_keys=False), flush=True)
    return summary


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument("--start-year", type=int, default=START_YEAR)
    parser.add_argument("--end-year", type=int, default=END_YEAR)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument(
        "--agera5-directory",
        type=Path,
        help="Directory containing the selected AgERA5 precipitation archives",
    )
    args = parser.parse_args()
    output_directory = (
        args.output_directory.resolve() if args.output_directory is not None else None
    )
    agera5_directory = (
        args.agera5_directory.resolve() if args.agera5_directory is not None else None
    )
    run(
        args.project_root.resolve(),
        args.start_year,
        args.end_year,
        output_directory,
        agera5_directory,
    )


if __name__ == "__main__":
    main()
