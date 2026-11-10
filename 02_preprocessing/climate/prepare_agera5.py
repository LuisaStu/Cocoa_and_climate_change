"""Convert full-extent daily AgERA5 archives to the common model grid."""

from __future__ import annotations

import argparse
import calendar
import hashlib
import re
import zipfile
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from io import BytesIO
from pathlib import Path
from typing import Any

import h5py  # type: ignore[import-untyped]
import numpy as np
import rasterio  # type: ignore[import-untyped]
import yaml  # type: ignore[import-untyped]
from rasterio.enums import Resampling  # type: ignore[import-untyped]
from rasterio.transform import from_origin  # type: ignore[import-untyped]
from rasterio.warp import reproject  # type: ignore[import-untyped]

SOURCE_CRS = "EPSG:4326"
TARGET_GRID = Path("01_inputs/processed/grid_full_extent/model_grid_EPSG6933_11000m.tif")
INPUT_DIRECTORY = Path("01_inputs/raw/agera5_full_extent")
INPUT_CHECKSUMS = Path("01_inputs/manifests/agera5_full_extent_sha256.txt")
OUTPUT_DIRECTORY = Path("01_inputs/processed/climate_full_extent/daily")
OUTPUT_MANIFEST = Path("01_inputs/manifests/agera5_full_extent_processed_manifest.yml")
OUTPUT_NODATA = -9999.0
MEMBER_DATE = re.compile(r"_(\d{8})_final")
COUNTRY_CODES = (1, 2)


@dataclass(frozen=True)
class VariableSpec:
    key: str
    dataset_name: str
    source_units: str
    output_units: str
    scale: float
    offset: float = 0.0
    valid_min: float | None = None
    valid_max: float | None = None


VARIABLES = (
    VariableSpec(
        "precipitation",
        "Precipitation_Flux",
        "mm d-1",
        "mm d-1",
        1.0,
        0.0,
        0.0,
        None,
    ),
    VariableSpec(
        "reference_et",
        "ReferenceET_PenmanMonteith_FAO56",
        "mm d-1",
        "mm d-1",
        1.0,
        0.0,
        0.0,
        30.0,
    ),
    VariableSpec(
        "solar_radiation",
        "Solar_Radiation_Flux",
        "J m-2 d-1",
        "MJ m-2 d-1",
        1e-6,
        0.0,
        0.0,
        60.0,
    ),
    VariableSpec("tmax", "Temperature_Air_2m_Max_24h", "K", "degC", 1.0, -273.15, -80.0, 70.0),
    VariableSpec("tmean", "Temperature_Air_2m_Mean_24h", "K", "degC", 1.0, -273.15, -80.0, 70.0),
    VariableSpec("tmin", "Temperature_Air_2m_Min_24h", "K", "degC", 1.0, -273.15, -80.0, 70.0),
    VariableSpec(
        "vpd", "Vapour_Pressure_Deficit_at_Maximum_Temperature", "hPa", "kPa", 0.1, 0.0, 0.0, 20.0
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_expected_checksums(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"AgERA5 checksum inventory is missing: {path}")
    checksums: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or not re.fullmatch(r"[0-9a-fA-F]{64}", parts[0]):
            raise ValueError(f"Malformed SHA-256 entry at {path}:{line_number}")
        checksums[parts[1].strip().replace("\\", "/")] = parts[0].lower()
    if not checksums:
        raise ValueError(f"AgERA5 checksum inventory is empty: {path}")
    return checksums


def expected_dates(year: int) -> list[date]:
    return [
        date(year, 1, 1) + timedelta(days=offset) for offset in range(365 + calendar.isleap(year))
    ]


def validate_member_dates(members: list[str], year: int) -> list[tuple[date, str]]:
    dated_members: list[tuple[date, str]] = []
    for member in members:
        match = MEMBER_DATE.search(Path(member).name)
        if match is None:
            raise ValueError(f"Cannot read daily date from AgERA5 member name: {member}")
        member_date = date.fromisoformat(
            f"{match.group(1)[:4]}-{match.group(1)[4:6]}-{match.group(1)[6:]}"
        )
        if member_date.year != year:
            raise ValueError(f"Unexpected year in AgERA5 member {member}: expected {year}")
        dated_members.append((member_date, member))
    dated_members.sort()
    dates = [member_date for member_date, _ in dated_members]
    expected = expected_dates(year)
    if dates != expected:
        missing = sorted(set(expected) - set(dates))
        duplicates = sorted({day for day in dates if dates.count(day) > 1})
        raise ValueError(
            f"AgERA5 daily member dates are incomplete for {year}; "
            f"missing={missing[:3]}, duplicates={duplicates[:3]}"
        )
    return dated_members


def _text_attribute(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8")
    return str(value)


def source_transform(latitudes: np.ndarray, longitudes: np.ndarray) -> rasterio.Affine:
    """Build a north-up geographic affine from regular coordinate centers."""
    latitudes = np.asarray(latitudes, dtype=np.float64)
    longitudes = np.asarray(longitudes, dtype=np.float64)
    for name, coordinates in (("latitude", latitudes), ("longitude", longitudes)):
        if coordinates.ndim != 1 or coordinates.size < 2:
            raise ValueError(f"AgERA5 {name} coordinates must have at least two values")
        differences = np.diff(coordinates)
        if not (np.all(differences > 0.0) or np.all(differences < 0.0)) or not np.allclose(
            differences, differences[0], rtol=0.0, atol=1e-8
        ):
            raise ValueError(f"AgERA5 {name} coordinates must be regularly spaced")
    longitude_step = abs(float(np.diff(longitudes)[0]))
    latitude_step = abs(float(np.diff(latitudes)[0]))
    west = float(np.min(longitudes) - longitude_step / 2.0)
    north = float(np.max(latitudes) + latitude_step / 2.0)
    return from_origin(west, north, longitude_step, latitude_step)


def orient_source(values: np.ndarray, latitudes: np.ndarray, longitudes: np.ndarray) -> np.ndarray:
    """Orient daily arrays north-to-south and west-to-east."""
    output = np.asarray(values)
    if latitudes[0] < latitudes[-1]:
        output = output[::-1, :]
    if longitudes[0] > longitudes[-1]:
        output = output[:, ::-1]
    return output


def standardize_values(values: np.ndarray, spec: VariableSpec, source_units: str) -> np.ndarray:
    if source_units.strip() != spec.source_units:
        raise ValueError(
            f"Unexpected units for {spec.key}: {source_units!r}; expected {spec.source_units!r}"
        )
    result = np.asarray(values, dtype=np.float32) * spec.scale + spec.offset
    finite = np.isfinite(result)
    if spec.valid_min is not None and np.any(result[finite] < spec.valid_min):
        raise ValueError(f"Converted {spec.key} values are below {spec.valid_min}")
    if spec.valid_max is not None and np.any(result[finite] > spec.valid_max):
        raise ValueError(f"Converted {spec.key} values exceed {spec.valid_max}")
    return result


def _read_member(
    payload: bytes,
    spec: VariableSpec,
    expected_coordinates: tuple[np.ndarray, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(BytesIO(payload), "r") as dataset:
        for required in (spec.dataset_name, "lat", "lon", "crs"):
            if required not in dataset:
                raise ValueError(f"AgERA5 NetCDF member lacks required dataset {required!r}")
        variable = dataset[spec.dataset_name]
        if variable.ndim != 3 or variable.shape[0] != 1:
            raise ValueError(
                f"Expected one daily slice in {spec.dataset_name}, got {variable.shape}"
            )
        latitudes = np.asarray(dataset["lat"][:], dtype=np.float64)
        longitudes = np.asarray(dataset["lon"][:], dtype=np.float64)
        if variable.shape[1:] != (latitudes.size, longitudes.size):
            raise ValueError(f"Coordinate dimensions do not match {spec.dataset_name}")
        if expected_coordinates is not None and (
            not np.array_equal(latitudes, expected_coordinates[0])
            or not np.array_equal(longitudes, expected_coordinates[1])
        ):
            raise ValueError("AgERA5 coordinates changed within or between annual archives")
        if variable.attrs.get("grid_mapping") not in (b"crs", "crs", np.bytes_("crs")):
            raise ValueError(f"Unexpected grid mapping for {spec.dataset_name}")
        units = _text_attribute(variable.attrs.get("units", ""))
        raw = np.asarray(variable[0], dtype=np.float32)
        fill_value = variable.attrs.get("_FillValue", variable.attrs.get("missing_value"))
        if fill_value is not None:
            raw[raw == float(np.asarray(fill_value).ravel()[0])] = np.nan
        raw = orient_source(raw, latitudes, longitudes)
        standardized = standardize_values(raw, spec, units)
        return standardized, latitudes, longitudes


def reproject_daily(
    values: np.ndarray,
    source_affine: rasterio.Affine,
    target: rasterio.DatasetReader,
    country_mask: np.ndarray,
) -> np.ndarray:
    output = np.full((target.height, target.width), np.nan, dtype=np.float32)
    reproject(
        source=values,
        destination=output,
        src_transform=source_affine,
        src_crs=SOURCE_CRS,
        src_nodata=np.nan,
        dst_transform=target.transform,
        dst_crs=target.crs,
        dst_nodata=np.nan,
        resampling=Resampling.average,
        init_dest_nodata=True,
    )
    output[~country_mask] = np.nan
    if np.any(country_mask & ~np.isfinite(output)):
        count = int(np.count_nonzero(country_mask & ~np.isfinite(output)))
        raise ValueError(f"AgERA5 daily source does not cover {count} country grid cells")
    return output


def _validate_existing_output(
    output_path: Path,
    spec: VariableSpec,
    year: int,
    input_sha256: str,
    target_grid_path: Path,
) -> dict[str, Any]:
    days = expected_dates(year)
    with rasterio.open(target_grid_path) as target, rasterio.open(output_path, "r+") as dataset:
        expected_country_mask = np.isin(target.read(1), COUNTRY_CODES)
        if (
            dataset.width != target.width
            or dataset.height != target.height
            or dataset.crs != target.crs
            or dataset.transform != target.transform
            or dataset.count != len(days)
            or dataset.nodata != OUTPUT_NODATA
        ):
            raise ValueError(
                f"Existing AgERA5 output has incorrect grid or dimensions: {output_path}"
            )
        if (
            dataset.tags().get("variable") != spec.key
            or dataset.tags().get("units") != spec.output_units
        ):
            raise ValueError(
                f"Existing AgERA5 output has incorrect variable metadata: {output_path}"
            )
        expected_descriptions = tuple(day.isoformat() for day in days)
        if dataset.descriptions != expected_descriptions:
            raise ValueError(
                f"Existing AgERA5 output has incorrect daily band dates: {output_path}"
            )
        tagged_sha = dataset.tags().get("input_sha256")
        if tagged_sha is not None and tagged_sha != input_sha256:
            raise ValueError(
                f"Existing AgERA5 output references a different source archive: {output_path}"
            )
        values = dataset.read(masked=True)
        valid = ~np.ma.getmaskarray(values) & np.isfinite(np.asarray(values))
        country_by_band = np.broadcast_to(expected_country_mask, valid.shape)
        if not np.all(valid[country_by_band]) or np.any(valid[~country_by_band]):
            raise ValueError(
                f"Existing AgERA5 output has missing or out-of-domain cells: {output_path}"
            )
        finite_values = np.asarray(values)[valid]
        if spec.valid_min is not None and np.any(finite_values < spec.valid_min):
            raise ValueError(f"Existing AgERA5 output contains invalid low values: {output_path}")
        if spec.valid_max is not None and np.any(finite_values > spec.valid_max):
            raise ValueError(f"Existing AgERA5 output contains invalid high values: {output_path}")
        dataset.update_tags(input_sha256=input_sha256, processor="prepare_agera5.py")
    return {
        "path": output_path.as_posix(),
        "sha256": sha256_file(output_path),
        "size_bytes": output_path.stat().st_size,
        "days": len(days),
        "valid_country_grid_cells_per_day": int(np.count_nonzero(expected_country_mask)),
        "minimum": float(np.min(finite_values)),
        "maximum": float(np.max(finite_values)),
        "units": spec.output_units,
        "input_sha256": input_sha256,
        "resumed_after_validation": True,
    }


def process_archive(
    archive_path: Path,
    spec: VariableSpec,
    year: int,
    expected_sha256: str,
    output_path: Path,
    target_grid_path: Path,
) -> dict[str, Any]:
    if not archive_path.is_file():
        raise FileNotFoundError(f"Missing AgERA5 archive: {archive_path}")
    actual_sha256 = sha256_file(archive_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"SHA-256 mismatch for AgERA5 archive: {archive_path}")
    if output_path.is_file():
        return _validate_existing_output(output_path, spec, year, actual_sha256, target_grid_path)

    with zipfile.ZipFile(archive_path) as archive:
        daily_members = validate_member_dates(
            [name for name in archive.namelist() if name.lower().endswith(".nc")], year
        )
        with rasterio.open(target_grid_path) as target:
            country_mask = np.isin(target.read(1), COUNTRY_CODES)
            if not np.any(country_mask):
                raise ValueError(
                    f"Target grid contains no assigned country cells: {target_grid_path}"
                )
            profile = target.profile.copy()
            profile.update(
                driver="GTiff",
                count=len(daily_members),
                dtype="float32",
                nodata=OUTPUT_NODATA,
                compress="deflate",
                predictor=3,
                tiled=False,
            )
            profile.pop("blockxsize", None)
            profile.pop("blockysize", None)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = output_path.with_suffix(output_path.suffix + ".part")
            source_coordinates: tuple[np.ndarray, np.ndarray] | None = None
            source_affine: rasterio.Affine | None = None
            minimum = float("inf")
            maximum = float("-inf")
            valid_country_cells = int(np.count_nonzero(country_mask))
            try:
                with rasterio.open(temporary_path, "w", **profile) as destination:
                    destination.update_tags(
                        variable=spec.key,
                        units=spec.output_units,
                        time_coverage=f"{year}-01-01/{year}-12-31",
                        input_sha256=actual_sha256,
                        processor="prepare_agera5.py",
                        source="AgERA5 daily weather variables, Copernicus CDS",
                        resampling="GDAL average",
                    )
                    for band, (day, member) in enumerate(daily_members, start=1):
                        values, latitudes, longitudes = _read_member(
                            archive.read(member), spec, source_coordinates
                        )
                        if source_coordinates is None:
                            source_coordinates = (latitudes.copy(), longitudes.copy())
                            source_affine = source_transform(latitudes, longitudes)
                        if source_affine is None:
                            raise RuntimeError("AgERA5 source affine was not initialized")
                        reprojected = reproject_daily(values, source_affine, target, country_mask)
                        finite = reprojected[country_mask]
                        minimum = min(minimum, float(np.min(finite)))
                        maximum = max(maximum, float(np.max(finite)))
                        band_values = np.where(country_mask, reprojected, OUTPUT_NODATA).astype(
                            np.float32
                        )
                        destination.write(band_values, band)
                        destination.set_band_description(band, day.isoformat())
                temporary_path.replace(output_path)
            except Exception:
                temporary_path.unlink(missing_ok=True)
                raise
    return {
        "path": output_path.as_posix(),
        "sha256": sha256_file(output_path),
        "size_bytes": output_path.stat().st_size,
        "days": len(daily_members),
        "valid_country_grid_cells_per_day": valid_country_cells,
        "minimum": minimum,
        "maximum": maximum,
        "units": spec.output_units,
        "input_sha256": actual_sha256,
    }


def prepare_agera5(
    input_directory: Path = INPUT_DIRECTORY,
    grid_path: Path = TARGET_GRID,
    output_directory: Path = OUTPUT_DIRECTORY,
    input_checksums_path: Path = INPUT_CHECKSUMS,
    manifest_path: Path = OUTPUT_MANIFEST,
    start_year: int = 2000,
    end_year: int = 2020,
) -> dict[str, Any]:
    if start_year > end_year:
        raise ValueError("Start year must not exceed end year")
    checksums = load_expected_checksums(input_checksums_path)
    outputs: dict[str, dict[str, Any]] = {}
    for spec in VARIABLES:
        outputs[spec.key] = {}
        for year in range(start_year, end_year + 1):
            archive_path = input_directory / f"agera5_{spec.key}_{year}.zip"
            inventory_path = archive_path.as_posix()
            expected_sha256 = checksums.get(inventory_path)
            if expected_sha256 is None:
                raise ValueError(f"No SHA-256 inventory entry for {inventory_path}")
            output_path = output_directory / (f"{spec.key}_{year}_EPSG6933_11000m.tif")
            outputs[spec.key][str(year)] = process_archive(
                archive_path,
                spec,
                year,
                expected_sha256,
                output_path,
                grid_path,
            )
            print(f"Prepared {spec.key} {year}: {output_path}", flush=True)

    with rasterio.open(grid_path) as grid:
        grid_codes = grid.read(1)
        country_grid_cells = {
            "GHA": int(np.count_nonzero(grid_codes == 1)),
            "CIV": int(np.count_nonzero(grid_codes == 2)),
        }
        ambiguous_cells = int(np.count_nonzero(grid_codes == 3))
    manifest = {
        "status": "complete",
        "source_dataset": "AgERA5 daily weather variables (Copernicus Climate Data Store)",
        "source_reference": "https://doi.org/10.24381/cds.6c68c9bb",
        "period": {"start_year": start_year, "end_year": end_year},
        "target_grid": {
            "path": grid_path.as_posix(),
            "crs": "EPSG:6933",
            "resolution_m": 11000,
            "country_grid_cells": country_grid_cells,
            "ambiguous_boundary_cells_excluded": ambiguous_cells,
        },
        "method": {
            "temporal_resolution": "daily",
            "spatial_resampling": "GDAL average to equal-area target cells",
            "country_mask": "grid codes 1 and 2 only; outside and ambiguous cells are NoData",
            "missing_data": "No imputation; all assigned country cells required valid per day",
            "variables": {spec.key: asdict(spec) for spec in VARIABLES},
            "storage": "One daily-band GeoTIFF per variable and year; band descriptions are ISO dates",
        },
        "outputs": outputs,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", type=Path, default=INPUT_DIRECTORY)
    parser.add_argument("--grid", type=Path, default=TARGET_GRID)
    parser.add_argument("--output-directory", type=Path, default=OUTPUT_DIRECTORY)
    parser.add_argument("--input-checksums", type=Path, default=INPUT_CHECKSUMS)
    parser.add_argument("--manifest", type=Path, default=OUTPUT_MANIFEST)
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--end-year", type=int, default=2020)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_agera5(
        args.input_directory,
        args.grid,
        args.output_directory,
        args.input_checksums,
        args.manifest,
        args.start_year,
        args.end_year,
    )
    print(f"Processing manifest: {args.manifest}")


if __name__ == "__main__":
    main()
