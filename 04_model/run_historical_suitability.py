from __future__ import annotations

import argparse
import csv
import hashlib
import json
from contextlib import ExitStack
from dataclasses import asdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import yaml
from simple_cacao import published_cacao_parameters, relative_suitability_index
from simple_cacao.vectorized import run_vectorized_cohort

ROOT = Path(__file__).resolve().parents[1]
VARIABLES = {
    "precipitation": "mm d-1",
    "reference_et": "mm d-1",
    "solar_radiation": "MJ m-2 d-1",
    "tmax": "degC",
    "tmin": "degC",
}
COUNTRIES = (("GHA", 1), ("CIV", 2))
NODATA = -9999.0


def _load_seasonal_calendar() -> tuple[dict[str, Any], dict[int, str], int]:
    path = ROOT / "03_parameters/flowering/cacao_seasonal_cohort_calendar.yml"
    calendar = yaml.safe_load(path.read_text(encoding="utf-8"))
    calendar_config = calendar.get("calendar", {})
    start_day = calendar_config.get("start_day_of_month")
    if not isinstance(start_day, int) or not 1 <= start_day <= 28:
        raise ValueError(f"Invalid cohort start day in {path}")

    seasons = calendar_config.get("seasons")
    if not isinstance(seasons, list) or not seasons:
        raise ValueError(f"No seasonal cases configured in {path}")
    season_by_month: dict[int, str] = {}
    for season in seasons:
        season_id = season.get("id")
        months = season.get("months")
        if not isinstance(season_id, str) or not season_id:
            raise ValueError(f"Invalid season id in {path}")
        if not isinstance(months, list) or not months:
            raise ValueError(f"Invalid months for {season_id} in {path}")
        for month in months:
            if not isinstance(month, int) or not 1 <= month <= 12:
                raise ValueError(f"Invalid month for {season_id} in {path}")
            if month in season_by_month:
                raise ValueError(f"Month {month} appears in multiple seasons in {path}")
            season_by_month[month] = season_id
    return calendar, season_by_month, start_day


def _manifest_records(node: Any):
    if isinstance(node, dict):
        if "path" in node and "sha256" in node:
            yield str(node["path"]).replace("\\", "/"), str(node["sha256"])
        for value in node.values():
            yield from _manifest_records(value)
    elif isinstance(node, list):
        for value in node:
            yield from _manifest_records(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dates_in_year(year: int) -> list[str]:
    day_count = 366 if _is_leap_year(year) else 365
    first = date(year, 1, 1)
    return [(first + timedelta(days=offset)).isoformat() for offset in range(day_count)]


def _is_leap_year(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def _validate_source(
    src: rasterio.io.DatasetReader,
    *,
    expected_shape: tuple[int, int],
    expected_transform: rasterio.Affine,
    expected_crs: rasterio.crs.CRS,
    expected_dates: list[str],
    expected_units: str,
    path: Path,
) -> None:
    if (
        src.shape != expected_shape
        or src.transform != expected_transform
        or src.crs != expected_crs
    ):
        raise ValueError(f"Climate raster grid mismatch: {path}")
    if src.count != len(expected_dates) or list(src.descriptions) != expected_dates:
        raise ValueError(f"Climate raster date/band mismatch: {path}")
    if src.tags().get("units") != expected_units:
        raise ValueError(
            f"Climate raster units mismatch for {path}: "
            f"{src.tags().get('units')!r} != {expected_units!r}"
        )


def _read_cohort_weather(
    sources: dict[tuple[str, int], rasterio.io.DatasetReader],
    start_date: date,
    n_days: int,
    cell_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    pieces: dict[str, list[np.ndarray]] = {variable: [] for variable in VARIABLES}
    cursor = start_date
    remaining = n_days
    while remaining:
        segment_days = min(
            remaining,
            (date(cursor.year, 12, 31) - cursor).days + 1,
        )
        band_indexes = list(
            range(cursor.timetuple().tm_yday, cursor.timetuple().tm_yday + segment_days)
        )
        for variable in VARIABLES:
            masked = sources[variable, cursor.year].read(band_indexes, masked=True)
            values = masked.filled(np.nan).reshape(segment_days, -1)[:, cell_indices]
            pieces[variable].append(np.asarray(values, dtype=np.float64))
        cursor += timedelta(days=segment_days)
        remaining -= segment_days

    result = {name: np.concatenate(chunks, axis=0) for name, chunks in pieces.items()}
    if any(not np.isfinite(values).all() for values in result.values()):
        raise ValueError(f"Missing climate values in cohort beginning {start_date}")
    if (
        np.any(result["tmin"] > result["tmax"])
        or np.any(result["precipitation"] < 0)
        or np.any(result["solar_radiation"] < 0)
        or np.any(result["reference_et"] < 0)
    ):
        raise ValueError(f"Nonphysical climate values in cohort beginning {start_date}")
    return result


def _load_inputs():
    input_manifest_path = ROOT / "01_inputs/manifests/agera5_full_extent_processed_manifest.yml"
    input_manifest = yaml.safe_load(input_manifest_path.read_text(encoding="utf-8"))
    expected_hashes = dict(_manifest_records(input_manifest))
    if not expected_hashes:
        raise ValueError(f"No checksums found in {input_manifest_path}")

    grid_path = ROOT / "01_inputs/processed/grid_full_extent/model_grid_EPSG6933_11000m.tif"
    cocoa_path = (
        ROOT
        / "01_inputs/processed/cocoa_mask_full_extent/cocoa_fraction_EPSG6933_11000m.tif"
    )
    awc_path = ROOT / "01_inputs/processed/soil_full_extent/soil_awc_EPSG6933_11000m.tif"
    static_source_hashes = {
        path.relative_to(ROOT).as_posix(): _sha256(path)
        for path in (grid_path, cocoa_path, awc_path)
    }
    with rasterio.open(grid_path) as grid_src:
        grid_codes = grid_src.read(1)
        profile = grid_src.profile.copy()
        country_cells = np.isin(grid_codes, [1, 2])
        if int(country_cells.sum()) != 4621:
            raise ValueError("Expected the audited 4,621 assigned country grid cells")

    with rasterio.open(cocoa_path) as cocoa_src:
        if (
            cocoa_src.shape != (profile["height"], profile["width"])
            or cocoa_src.transform != profile["transform"]
            or cocoa_src.crs != profile["crs"]
        ):
            raise ValueError("Cocoa fraction grid mismatch")
        cocoa_fraction = cocoa_src.read(1, masked=True).filled(np.nan).astype(np.float64)
    if np.any((cocoa_fraction[country_cells] < 0) | (cocoa_fraction[country_cells] > 1)):
        raise ValueError("Cocoa fractions must be between zero and one")

    selected = country_cells & np.isfinite(cocoa_fraction) & (cocoa_fraction > 0)
    selected_indices = np.flatnonzero(selected)
    if selected_indices.size == 0:
        raise ValueError("No valid mapped-cocoa cells were found")
    country_codes = grid_codes.reshape(-1)[selected_indices]
    cocoa_weights = cocoa_fraction.reshape(-1)[selected_indices]

    with rasterio.open(awc_path) as awc_src:
        if (
            awc_src.shape != (profile["height"], profile["width"])
            or awc_src.transform != profile["transform"]
            or awc_src.crs != profile["crs"]
            or awc_src.count != 3
            or awc_src.descriptions
            != (
                "awc_root_depth_1000mm",
                "awc_root_depth_1500mm",
                "awc_root_depth_2000mm",
            )
            or awc_src.tags(2).get("units") != "mm"
        ):
            raise ValueError("AWC raster layout does not match the recorded baseline")
        awc = awc_src.read(2, masked=True).filled(np.nan).reshape(-1)[selected_indices]
    if not np.isfinite(awc).all() or np.any(awc <= 0):
        raise ValueError("Invalid AWC values in mapped-cocoa cells")

    return (
        profile,
        grid_codes,
        country_cells,
        cocoa_fraction,
        selected_indices,
        country_codes,
        cocoa_weights,
        awc,
        expected_hashes,
        static_source_hashes,
    )


def _run(output_dir: Path, overwrite: bool = False) -> dict[str, Path]:
    seasonal_calendar, season_by_month, start_day = _load_seasonal_calendar()
    months = list(season_by_month)
    (
        grid_profile,
        grid_codes,
        country_cells,
        cocoa_fraction,
        cell_indices,
        country_codes,
        cocoa_weights,
        awc,
        manifest_hashes,
        static_source_hashes,
    ) = _load_inputs()

    response_path = output_dir / "historical_cohort_biomass_response.tif"
    index_path = output_dir / "historical_relative_suitability_index.tif"
    summary_path = ROOT / "08_outputs/country/historical_country_summary.csv"
    metadata_path = output_dir / "historical_run_manifest.json"
    outputs = (response_path, index_path, summary_path, metadata_path)
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Output exists; rerun with --overwrite to replace: "
            + ", ".join(str(path) for path in existing)
        )

    temp_paths = [path.with_name(path.name + ".tmp") for path in outputs]
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    years = tuple(range(2000, 2020))
    forcing_years = tuple(range(2000, 2021))
    source_hashes: dict[str, str] = {}
    parameters = published_cacao_parameters()
    weather_sources: dict[tuple[str, int], rasterio.io.DatasetReader] = {}
    country_total_cells = {
        country: int((grid_codes == code).sum()) for country, code in COUNTRIES
    }
    country_fraction_cells = {
        country: int(((grid_codes == code) & np.isfinite(cocoa_fraction)).sum())
        for country, code in COUNTRIES
    }

    try:
        with ExitStack() as stack:
            for year in forcing_years:
                expected_dates = _dates_in_year(year)
                for variable, units in VARIABLES.items():
                    rel_path = (
                        Path("01_inputs/processed/climate_full_extent/daily")
                        / f"{variable}_{year}_EPSG6933_11000m.tif"
                    )
                    checksum = manifest_hashes.get(rel_path.as_posix())
                    if checksum is None:
                        raise ValueError(f"No manifest checksum for {rel_path}")
                    path = ROOT / rel_path
                    actual_hash = _sha256(path)
                    if actual_hash != checksum:
                        raise ValueError(f"Input checksum mismatch: {rel_path}")
                    source_hashes[rel_path.as_posix()] = actual_hash
                    src = stack.enter_context(rasterio.open(path))
                    _validate_source(
                        src,
                        expected_shape=(grid_profile["height"], grid_profile["width"]),
                        expected_transform=grid_profile["transform"],
                        expected_crs=grid_profile["crs"],
                        expected_dates=expected_dates,
                        expected_units=units,
                        path=path,
                    )
                    weather_sources[variable, year] = src

            response_values = np.empty((len(months) * len(years), cell_indices.size), dtype=np.float32)
            days_run_values = np.empty((len(months) * len(years), cell_indices.size), dtype=np.int16)
            heat_stress_values = np.empty_like(days_run_values)
            water_stress_values = np.empty_like(days_run_values)
            band_months: list[int] = []
            band_years: list[int] = []
            band_number = 0
            for month in months:
                for start_year in years:
                    start_date = date(start_year, month, start_day)
                    weather = _read_cohort_weather(
                        weather_sources,
                        start_date,
                        parameters.max_cycle_days,
                        cell_indices,
                    )
                    simulation = run_vectorized_cohort(
                        weather["tmax"],
                        weather["tmin"],
                        weather["solar_radiation"],
                        weather["precipitation"],
                        weather["reference_et"],
                        awc,
                        parameters,
                    )
                    if not simulation.mature.all():
                        raise ValueError(f"Incomplete cohort cycle beginning {start_date}")
                    if not np.isfinite(simulation.biomass_g_m2).all():
                        raise ValueError(f"Invalid biomass response for cohort beginning {start_date}")
                    response_values[band_number] = simulation.biomass_g_m2.astype(np.float32)
                    days_run_values[band_number] = simulation.days_run.astype(np.int16)
                    heat_stress_values[band_number] = simulation.heat_stress_days.astype(np.int16)
                    water_stress_values[band_number] = simulation.water_stress_days.astype(np.int16)
                    band_months.append(month)
                    band_years.append(start_year)
                    band_number += 1
                print(f"Completed start-month {month:02d} for 2000-2019", flush=True)

        index_values = np.empty_like(response_values)
        for month in months:
            positions = [i for i, value in enumerate(band_months) if value == month]
            reference = response_values[positions].mean(axis=0, dtype=np.float64)
            index_values[positions] = relative_suitability_index(
                reference,
                response_values[positions].astype(np.float64),
            ).astype(np.float32)

        summary_rows: list[dict[str, Any]] = []
        selected_country_weights: dict[str, np.ndarray] = {}
        for country, code in COUNTRIES:
            country_mask = country_codes == code
            weights = cocoa_weights[country_mask]
            if weights.size == 0 or weights.sum() <= 0:
                raise ValueError(f"No mapped-cocoa weights for {country}")
            selected_country_weights[country] = weights / weights.sum()
        for band, (month, start_year) in enumerate(zip(band_months, band_years, strict=True)):
            for country, code in COUNTRIES:
                country_mask = country_codes == code
                weights = selected_country_weights[country]
                response_mean = float(np.average(response_values[band, country_mask], weights=weights))
                index_mean = float(np.average(index_values[band, country_mask], weights=weights))
                summary_rows.append(
                    {
                        "season": season_by_month[month],
                        "cohort_start_month": month,
                        "cohort_start_year": start_year,
                        "country": country,
                        "assigned_country_grid_cells": country_total_cells[country],
                        "mapped_cocoa_source_cells": country_fraction_cells[country],
                        "mapped_cocoa_cells_missing_source_fraction": (
                            country_total_cells[country] - country_fraction_cells[country]
                        ),
                        "mapped_cocoa_source_cell_coverage": (
                            country_fraction_cells[country] / country_total_cells[country]
                        ),
                        "mapped_cocoa_cells": int(country_mask.sum()),
                        "mapped_cocoa_weight_sum": float(cocoa_weights[country_codes == code].sum()),
                        "model_input_valid_cocoa_cell_coverage": 1.0,
                        "mean_model_biomass_response_g_m2": response_mean,
                        "cocoa_fraction_weighted_relative_index": index_mean,
                        "mean_cohort_days_to_maturity": float(
                            np.average(days_run_values[band, country_mask], weights=weights)
                        ),
                        "mean_heat_stress_days_per_cohort": float(
                            np.average(heat_stress_values[band, country_mask], weights=weights)
                        ),
                        "mean_water_stress_days_per_cohort": float(
                            np.average(water_stress_values[band, country_mask], weights=weights)
                        ),
                    }
                )

        response_temp, index_temp, summary_temp, metadata_temp = temp_paths
        profile = grid_profile.copy()
        profile.update(
            count=len(response_values),
            dtype="float32",
            nodata=NODATA,
            compress="deflate",
            predictor=3,
            tiled=False,
        )
        with rasterio.open(response_temp, "w", **profile) as dst:
            dst.update_tags(
                title="Historical SIMPLE-Cacao-WA cohort biomass response",
                units="g m-2",
                interpretation="internal model response; not actual or predicted cocoa yield",
                cohort_start_years="2000-2019",
                forcing_years="2000-2020",
                start_day_of_month=str(start_day),
                water_stress_reference="S_water=0; water deficit does not reduce biomass response",
            )
            for band, (month, start_year) in enumerate(zip(band_months, band_years, strict=True), 1):
                raster = np.full((profile["height"], profile["width"]), NODATA, dtype=np.float32)
                raster.reshape(-1)[cell_indices] = response_values[band - 1]
                dst.write(raster, band)
                dst.set_band_description(
                    band,
                    f"month={month:02d}; cohort_start_year={start_year}; biomass_g_m2",
                )

        with rasterio.open(index_temp, "w", **profile) as dst:
            dst.update_tags(
                title="Historical relative climatic-suitability proxy",
                units="index; historical same-month mean=100",
                interpretation="relative climate-response proxy; not actual yield or binary suitability",
                formula="100 * cohort_response / cellwise_mean_same_month_response_2000_2019",
                cohort_start_years="2000-2019",
                forcing_years="2000-2020",
                start_day_of_month=str(start_day),
                season_aggregation="none; monthly start-date cases are reported separately",
            )
            for band, (month, start_year) in enumerate(zip(band_months, band_years, strict=True), 1):
                raster = np.full((profile["height"], profile["width"]), NODATA, dtype=np.float32)
                raster.reshape(-1)[cell_indices] = index_values[band - 1]
                dst.write(raster, band)
                dst.set_band_description(
                    band,
                    f"month={month:02d}; cohort_start_year={start_year}; relative_index",
                )

        with summary_temp.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
            writer.writeheader()
            writer.writerows(summary_rows)

        metadata = {
            "status": "complete_historical_cohort_start_analysis",
            "created_at_local": datetime.now().astimezone().isoformat(),
            "interpretation": (
                "Relative climatic-suitability proxy from internal SIMPLE-Cacao-WA biomass "
                "response; not actual or predictive cocoa yield."
            ),
            "reference_parameters": asdict(parameters),
            "seasonal_calendar": "03_parameters/flowering/cacao_seasonal_cohort_calendar.yml",
            "cohort_calendar": {
                "start_months": months,
                "start_day_of_month": start_day,
                "seasons": seasonal_calendar["calendar"]["seasons"],
                "within_season_weights": seasonal_calendar["calendar"].get(
                    "monthly_weights"
                ),
                "season_aggregation": seasonal_calendar["calendar"].get(
                    "annual_production_aggregation", False
                ),
            },
            "time_windows": {
                "historical_cohort_start_years_inclusive": [2000, 2019],
                "historical_forcing_years_inclusive": [2000, 2020],
                "forcing_for_late_2019_starts": "2020 is used only to complete the maximum cycle",
            },
            "soil": {
                "awc_raster": "01_inputs/processed/soil_full_extent/soil_awc_EPSG6933_11000m.tif",
                "awc_band": 2,
                "awc_band_description": "awc_root_depth_1500mm",
                "awc_units": "mm",
            },
            "coverage": {
                "assigned_country_grid_cells": int(country_cells.sum()),
                "valid_mapped_cocoa_cells": int(cell_indices.size),
                "mapped_cocoa_cells_with_missing_source_fraction": {
                    country: country_total_cells[country] - country_fraction_cells[country]
                    for country, _ in COUNTRIES
                },
                "country_mapped_cocoa_cells": {
                    country: int((country_codes == code).sum()) for country, code in COUNTRIES
                },
                "country_mapped_cocoa_source_cell_coverage": {
                    country: country_fraction_cells[country] / country_total_cells[country]
                    for country, _ in COUNTRIES
                },
                "weighting": "cocoa fraction times equal-area cell area",
                "missing_climate_or_soil_values": 0,
            },
            "input_sha256": source_hashes,
            "static_input_sha256": static_source_hashes,
            "outputs": {
                "biomass_response": response_path.relative_to(ROOT).as_posix(),
                "relative_index": index_path.relative_to(ROOT).as_posix(),
                "country_summary": summary_path.relative_to(ROOT).as_posix(),
            },
        }
        metadata_temp.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

        for temporary, final in zip(temp_paths, outputs, strict=True):
            temporary.replace(final)
    except Exception:
        for temporary in temp_paths:
            temporary.unlink(missing_ok=True)
        raise

    return {
        "biomass_response": response_path,
        "relative_index": index_path,
        "country_summary": summary_path,
        "manifest": metadata_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run historical monthly flowering-date suitability cases on the full cocoa grid."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing historical outputs after a successful run.",
    )
    args = parser.parse_args()
    output_dir = ROOT / "08_outputs/gridded/historical_climatic_suitability"
    try:
        outputs = _run(output_dir, overwrite=args.overwrite)
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        parser.error(str(exc))
    for name, path in outputs.items():
        print(f"{name}: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
