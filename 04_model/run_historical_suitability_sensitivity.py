from __future__ import annotations

import csv
import json
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import yaml
from run_historical_suitability import (
    COUNTRIES,
    ROOT,
    VARIABLES,
    _dates_in_year,
    _load_inputs,
    _load_seasonal_calendar,
    _read_cohort_weather,
    _sha256,
    _validate_source,
)
from simple_cacao import published_cacao_parameters, relative_suitability_index
from simple_cacao.parameters import CacaoParameters
from simple_cacao.vectorized import run_vectorized_cohort

SENSITIVITY_CONFIG_PATH = (
    ROOT / "03_parameters/crop/cacao_suitability_sensitivity.yml"
)
OUTPUT_DIR = ROOT / "07_runs/sensitivity/historical_screening"
ANNUAL_OUTPUT_PATH = OUTPUT_DIR / "country_month_year_screening.csv"
SUMMARY_OUTPUT_PATH = OUTPUT_DIR / "country_month_screening_summary.csv"
MANIFEST_OUTPUT_PATH = OUTPUT_DIR / "screening_manifest.json"

PARAMETER_ATTRIBUTES = {
    "tbase_c": "tbase",
    "topt_c": "topt",
    "tsum_c_days": "tsum",
    "i50a_c_days": "i50a",
    "i50b_c_days": "i50b",
    "t_heat_c": "t_heat",
    "t_extreme_c": "t_extreme",
    "swater": "swater",
}


@dataclass(frozen=True)
class SensitivityCase:
    case_id: str
    varied_parameter: str
    level: str
    parameter_value: float | None
    rooting_depth_mm: int
    parameters: CacaoParameters


def _validate_parameters(parameters: CacaoParameters) -> None:
    if parameters.tsum <= 0 or parameters.max_cycle_days <= 0:
        raise ValueError("Thermal sum and maximum cycle length must be positive")
    if parameters.tbase >= parameters.topt:
        raise ValueError("tbase must be lower than topt")
    if parameters.t_heat >= parameters.t_extreme:
        raise ValueError("t_heat must be lower than t_extreme")
    if parameters.swater < 0:
        raise ValueError("swater must be non-negative")


def build_sensitivity_cases(
    config: dict[str, Any],
    reference: CacaoParameters,
    *,
    reference_rooting_depth_mm: int,
) -> list[SensitivityCase]:
    cases = [
        SensitivityCase(
            case_id="published_reference",
            varied_parameter="none",
            level="reference",
            parameter_value=None,
            rooting_depth_mm=reference_rooting_depth_mm,
            parameters=reference,
        )
    ]
    for name, attribute in PARAMETER_ATTRIBUTES.items():
        entry = config.get("parameters", {}).get(name)
        if entry is None:
            continue
        reference_value = float(entry["reference"])
        actual_reference = float(getattr(reference, attribute))
        if not np.isclose(reference_value, actual_reference):
            raise ValueError(
                f"Sensitivity reference mismatch for {name}: "
                f"{reference_value} != {actual_reference}"
            )
        interval = entry.get("range")
        if interval is None:
            continue
        if not isinstance(interval, list) or len(interval) != 2:
            raise ValueError(f"Expected a two-value screening range for {name}")
        for level, raw_value in zip(("low", "high"), interval, strict=True):
            value = float(raw_value)
            parameters = replace(reference, **{attribute: value})
            _validate_parameters(parameters)
            cases.append(
                SensitivityCase(
                    case_id=f"{name}_{level}",
                    varied_parameter=name,
                    level=level,
                    parameter_value=value,
                    rooting_depth_mm=reference_rooting_depth_mm,
                    parameters=parameters,
                )
            )

    soil_config = config.get("parameters", {}).get("rooting_depth_mm")
    if soil_config is not None:
        if int(float(soil_config["reference"])) != reference_rooting_depth_mm:
            raise ValueError("Rooting-depth sensitivity reference does not match baseline")
        interval = soil_config.get("range")
        if not isinstance(interval, list) or len(interval) != 2:
            raise ValueError("Expected two rooting-depth screening values")
        for level, raw_value in zip(("low", "high"), interval, strict=True):
            depth = int(raw_value)
            cases.append(
                SensitivityCase(
                    case_id=f"rooting_depth_mm_{level}",
                    varied_parameter="rooting_depth_mm",
                    level=level,
                    parameter_value=float(depth),
                    rooting_depth_mm=depth,
                    parameters=reference,
                )
            )
    return cases


def _load_awc_by_depth(
    cell_indices: np.ndarray,
    *,
    expected_shape: tuple[int, int],
    expected_transform: rasterio.Affine,
    expected_crs: rasterio.crs.CRS,
) -> dict[int, np.ndarray]:
    path = ROOT / "01_inputs/processed/soil_full_extent/soil_awc_EPSG6933_11000m.tif"
    depths = (1000, 1500, 2000)
    descriptions = tuple(f"awc_root_depth_{depth}mm" for depth in depths)
    with rasterio.open(path) as source:
        if (
            source.shape != expected_shape
            or source.transform != expected_transform
            or source.crs != expected_crs
            or source.count != len(depths)
            or source.descriptions != descriptions
            or any(source.tags(index).get("units") != "mm" for index in range(1, 4))
        ):
            raise ValueError("AWC raster layout does not match rooting-depth sensitivities")
        values = {
            depth: source.read(index, masked=True)
            .filled(np.nan)
            .reshape(-1)[cell_indices]
            .astype(np.float64)
            for index, depth in enumerate(depths, start=1)
        }
    if any(not np.isfinite(value).all() or np.any(value <= 0) for value in values.values()):
        raise ValueError("Invalid AWC values in one or more rooting-depth cases")
    return values


def _load_sensitivity_config() -> dict[str, Any]:
    config = yaml.safe_load(SENSITIVITY_CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("status") != "exploratory_screening_envelopes_not_probability_distributions":
        raise ValueError("Sensitivity config must remain explicitly exploratory")
    if config.get("design", {}).get("method") != "one_at_a_time_low_reference_high":
        raise ValueError("Only the documented one-at-a-time sensitivity is supported")
    return config


def _country_weights(
    country_codes: np.ndarray,
    cocoa_weights: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    output = {}
    for country, code in COUNTRIES:
        selected = country_codes == code
        weights = cocoa_weights[selected]
        if not weights.size or not np.isfinite(weights).all() or weights.sum() <= 0:
            raise ValueError(f"Invalid mapped-cocoa weights for {country}")
        output[country] = (selected, weights / weights.sum())
    return output


def _add_reference_differences(
    annual_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
) -> None:
    annual_reference = {
        (row["country"], row["cohort_start_month"], row["cohort_start_year"]): float(
            row["mean_relative_index"]
        )
        for row in annual_rows
        if row["case_id"] == "published_reference"
    }
    summary_reference = {
        (row["country"], row["cohort_start_month"]): (
            float(row["mean_historical_relative_index"]),
            float(row["linear_index_trend_points_per_year"]),
        )
        for row in summary_rows
        if row["case_id"] == "published_reference"
    }
    if not annual_reference or not summary_reference:
        raise ValueError("Published-reference rows are required for sensitivity deltas")

    for row in annual_rows:
        key = (row["country"], row["cohort_start_month"], row["cohort_start_year"])
        try:
            reference = annual_reference[key]
        except KeyError as exc:
            raise ValueError(f"Missing published-reference annual row for {key}") from exc
        row["relative_index_difference_from_published_reference"] = (
            float(row["mean_relative_index"]) - reference
        )

    for row in summary_rows:
        key = (row["country"], row["cohort_start_month"])
        try:
            reference_mean, reference_trend = summary_reference[key]
        except KeyError as exc:
            raise ValueError(f"Missing published-reference summary row for {key}") from exc
        row["mean_historical_relative_index_difference_from_published_reference"] = (
            float(row["mean_historical_relative_index"]) - reference_mean
        )
        row["index_trend_difference_from_published_reference_points_per_year"] = (
            float(row["linear_index_trend_points_per_year"]) - reference_trend
        )


def _run_screening() -> tuple[Path, Path, Path]:
    sensitivity_config = _load_sensitivity_config()
    seasonal_calendar, season_by_month, start_day = _load_seasonal_calendar()
    months = list(season_by_month)
    (
        grid_profile,
        grid_codes,
        _,
        cocoa_fraction,
        cell_indices,
        country_codes,
        cocoa_weights,
        reference_awc,
        manifest_hashes,
        static_hashes,
    ) = _load_inputs()
    awc_by_depth = _load_awc_by_depth(
        cell_indices,
        expected_shape=(grid_profile["height"], grid_profile["width"]),
        expected_transform=grid_profile["transform"],
        expected_crs=grid_profile["crs"],
    )
    if not np.array_equal(reference_awc, awc_by_depth[1500]):
        raise ValueError("Baseline AWC values do not match the 1500 mm sensitivity band")

    baseline_config = yaml.safe_load(
        (ROOT / "03_parameters/crop/cacao_suitability_baseline.yml").read_text(
            encoding="utf-8"
        )
    )
    reference_root_depth = int(baseline_config["soil"]["rooting_depth_mm"])
    reference_parameters = published_cacao_parameters()
    cases = build_sensitivity_cases(
        sensitivity_config,
        reference_parameters,
        reference_rooting_depth_mm=reference_root_depth,
    )
    weights_by_country = _country_weights(country_codes, cocoa_weights)
    source_coverage_by_country = {
        country: int(((grid_codes == code) & np.isfinite(cocoa_fraction)).sum())
        / int((grid_codes == code).sum())
        for country, code in COUNTRIES
    }
    year_range = baseline_config["scenarios"]["historical_cohort_start_years_inclusive"]
    start_year, end_year = map(int, year_range)
    years = tuple(range(start_year, end_year + 1))
    forcing_years = tuple(
        map(int, baseline_config["forcing"]["historical_forcing_years_inclusive"])
    )
    if (
        len(forcing_years) != 2
        or forcing_years[0] != start_year
        or forcing_years[1] < end_year + 1
    ):
        raise ValueError("Historical forcing must include the full trailing cohort year")
    forcing_years = tuple(range(forcing_years[0], forcing_years[1] + 1))

    outputs = (ANNUAL_OUTPUT_PATH, SUMMARY_OUTPUT_PATH, MANIFEST_OUTPUT_PATH)
    existing = [path for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(
            "Sensitivity output exists; preserve it or move it before rerunning: "
            + ", ".join(path.relative_to(ROOT).as_posix() for path in existing)
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    temporary_outputs = tuple(path.with_name(path.name + ".tmp") for path in outputs)
    annual_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    climate_hashes: dict[str, str] = {}
    sources: dict[tuple[str, int], rasterio.io.DatasetReader] = {}
    try:
        with ExitStack() as stack:
            for year in forcing_years:
                expected_dates = _dates_in_year(year)
                for variable, units in VARIABLES.items():
                    relative_path = (
                        Path("01_inputs/processed/climate_full_extent/daily")
                        / f"{variable}_{year}_EPSG6933_11000m.tif"
                    )
                    expected_hash = manifest_hashes.get(relative_path.as_posix())
                    if expected_hash is None:
                        raise ValueError(f"No climate checksum for {relative_path}")
                    path = ROOT / relative_path
                    actual_hash = _sha256(path)
                    if actual_hash != expected_hash:
                        raise ValueError(f"Climate checksum mismatch: {relative_path}")
                    climate_hashes[relative_path.as_posix()] = actual_hash
                    source = stack.enter_context(rasterio.open(path))
                    _validate_source(
                        source,
                        expected_shape=(grid_profile["height"], grid_profile["width"]),
                        expected_transform=grid_profile["transform"],
                        expected_crs=grid_profile["crs"],
                        expected_dates=expected_dates,
                        expected_units=units,
                        path=path,
                    )
                    sources[variable, year] = source

            response_by_case = {
                case.case_id: np.empty(
                    (len(months) * len(years), cell_indices.size), dtype=np.float64
                )
                for case in cases
            }
            days_by_case = {
                case.case_id: np.empty(
                    (len(months) * len(years), cell_indices.size), dtype=np.int16
                )
                for case in cases
            }
            heat_by_case = {
                case.case_id: np.empty_like(days_by_case[case.case_id]) for case in cases
            }
            water_by_case = {
                case.case_id: np.empty_like(days_by_case[case.case_id]) for case in cases
            }
            capped_by_case = {
                case.case_id: np.empty_like(days_by_case[case.case_id], dtype=bool)
                for case in cases
            }
            band_months = [month for month in months for _ in years]
            band_years = [year for _ in months for year in years]

            for band, (month, year) in enumerate(
                zip(band_months, band_years, strict=True)
            ):
                start_date = date(year, month, start_day)
                weather = _read_cohort_weather(
                    sources,
                    start_date,
                    max(case.parameters.max_cycle_days for case in cases),
                    cell_indices,
                )
                for case in cases:
                    simulation = run_vectorized_cohort(
                        weather["tmax"],
                        weather["tmin"],
                        weather["solar_radiation"],
                        weather["precipitation"],
                        weather["reference_et"],
                        awc_by_depth[case.rooting_depth_mm],
                        case.parameters,
                    )
                    if not simulation.mature.all():
                        raise ValueError(
                            f"Incomplete cohort cycle beginning {start_date} "
                            f"for sensitivity case {case.case_id}"
                        )
                    if not np.isfinite(simulation.biomass_g_m2).all():
                        raise ValueError(f"Non-finite response in case {case.case_id}")
                    response_by_case[case.case_id][band] = simulation.biomass_g_m2
                    days_by_case[case.case_id][band] = simulation.days_run
                    heat_by_case[case.case_id][band] = simulation.heat_stress_days
                    water_by_case[case.case_id][band] = simulation.water_stress_days
                    capped_by_case[case.case_id][band] = (
                        simulation.days_run >= case.parameters.max_cycle_days
                    )
                print(
                    f"Completed cohort month {month:02d}, year {year} "
                    f"({band + 1}/{len(band_months)}) across {len(cases)} cases",
                    flush=True,
                )

        for case in cases:
            response_values = response_by_case[case.case_id]
            day_values = days_by_case[case.case_id]
            heat_values = heat_by_case[case.case_id]
            water_values = water_by_case[case.case_id]
            capped_values = capped_by_case[case.case_id]
            index_values = np.empty_like(response_values)
            for month in months:
                positions = [
                    index for index, value in enumerate(band_months) if value == month
                ]
                reference = response_values[positions].mean(axis=0, dtype=np.float64)
                index_values[positions] = relative_suitability_index(
                    reference,
                    response_values[positions],
                )

            monthly_country_values: dict[tuple[str, int], list[dict[str, Any]]] = (
                defaultdict(list)
            )
            for position, (month, year) in enumerate(
                zip(band_months, band_years, strict=True)
            ):
                for country, (country_mask, country_weights) in weights_by_country.items():
                    response = response_values[position, country_mask]
                    index = index_values[position, country_mask]
                    days = day_values[position, country_mask]
                    heat = heat_values[position, country_mask]
                    water = water_values[position, country_mask]
                    capped = capped_values[position, country_mask]
                    row = {
                        "case_id": case.case_id,
                        "design_status": "exploratory_screening_not_probability_bound",
                        "varied_parameter": case.varied_parameter,
                        "sensitivity_level": case.level,
                        "parameter_value": case.parameter_value,
                        "rooting_depth_mm": case.rooting_depth_mm,
                        "country": country,
                        "season": season_by_month[month],
                        "cohort_start_month": month,
                        "cohort_start_year": year,
                        "mapped_cocoa_cells": int(country_mask.sum()),
                        "mapped_cocoa_source_cell_coverage": source_coverage_by_country[
                            country
                        ],
                        "model_input_valid_cocoa_cell_coverage": 1.0,
                        "mean_relative_index": float(
                            np.average(index, weights=country_weights)
                        ),
                        "mean_model_biomass_response_g_m2": float(
                            np.average(response, weights=country_weights)
                        ),
                        "mean_cohort_days": float(np.average(days, weights=country_weights)),
                        "mean_heat_stress_days": float(
                            np.average(heat, weights=country_weights)
                        ),
                        "mean_water_stress_days": float(
                            np.average(water, weights=country_weights)
                        ),
                        "cycle_cap_cell_count": int(capped.sum()),
                        "cycle_cap_cell_fraction": float(capped.mean()),
                    }
                    annual_rows.append(row)
                    monthly_country_values[country, month].append(row)

            for (country, month), values in monthly_country_values.items():
                indexes = [float(row["mean_relative_index"]) for row in values]
                biomass = [
                    float(row["mean_model_biomass_response_g_m2"]) for row in values
                ]
                year_values = [int(row["cohort_start_year"]) for row in values]
                year_mean = float(np.mean(year_values))
                index_mean = float(np.mean(indexes))
                year_ss = float(np.sum((np.asarray(year_values) - year_mean) ** 2))
                index_slope = float(
                    np.sum(
                        (np.asarray(year_values) - year_mean)
                        * (np.asarray(indexes) - index_mean)
                    )
                    / year_ss
                )
                summary_rows.append(
                    {
                        "case_id": case.case_id,
                        "design_status": "exploratory_screening_not_probability_bound",
                        "varied_parameter": case.varied_parameter,
                        "sensitivity_level": case.level,
                        "parameter_value": case.parameter_value,
                        "rooting_depth_mm": case.rooting_depth_mm,
                        "country": country,
                        "season": season_by_month[month],
                        "cohort_start_month": month,
                        "cohort_year_count": len(values),
                        "mean_historical_relative_index": index_mean,
                        "interannual_sd_relative_index": float(np.std(indexes, ddof=1)),
                        "min_annual_relative_index": min(indexes),
                        "max_annual_relative_index": max(indexes),
                        "linear_index_trend_points_per_year": index_slope,
                        "mean_model_biomass_response_g_m2": float(np.mean(biomass)),
                        "mean_heat_stress_days": float(
                            np.mean([float(row["mean_heat_stress_days"]) for row in values])
                        ),
                        "maximum_mean_water_stress_days": max(
                            float(row["mean_water_stress_days"]) for row in values
                        ),
                        "maximum_cycle_cap_cell_fraction": max(
                            float(row["cycle_cap_cell_fraction"]) for row in values
                        ),
                    }
                )

        _add_reference_differences(annual_rows, summary_rows)

        for path, rows in (
            (temporary_outputs[0], annual_rows),
            (temporary_outputs[1], summary_rows),
        ):
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

        manifest = {
            "status": "complete_exploratory_historical_sensitivity_screen",
            "interpretation": (
                "Deterministic one-at-a-time screening of explicitly exploratory "
                "candidate envelopes; not calibrated uncertainty bounds, probability "
                "distributions, or validated West African parameter ranges."
            ),
            "created_at_local": datetime.now().astimezone().isoformat(),
            "sensitivity_config_status": sensitivity_config["status"],
            "reference_parameters": asdict(reference_parameters),
            "case_count": len(cases),
            "case_ids": [case.case_id for case in cases],
            "cohort_start_years_inclusive": [start_year, end_year],
            "forcing_years_inclusive": [forcing_years[0], forcing_years[-1]],
            "cohort_months": months,
            "calendar_status": seasonal_calendar.get("status"),
            "season_aggregation": False,
            "mapped_cocoa_cells": int(cell_indices.size),
            "missing_cocoa_mask_cells": {
                country: int((grid_codes == code).sum())
                - int(((grid_codes == code) & np.isfinite(cocoa_fraction)).sum())
                for country, code in COUNTRIES
            },
            "country_mapped_cocoa_source_cell_coverage": source_coverage_by_country,
            "input_sha256": {
                **climate_hashes,
                **static_hashes,
                SENSITIVITY_CONFIG_PATH.relative_to(ROOT).as_posix(): _sha256(
                    SENSITIVITY_CONFIG_PATH
                ),
                (ROOT / "03_parameters/crop/cacao_suitability_baseline.yml")
                .relative_to(ROOT)
                .as_posix(): _sha256(ROOT / "03_parameters/crop/cacao_suitability_baseline.yml"),
                (ROOT / "03_parameters/flowering/cacao_seasonal_cohort_calendar.yml")
                .relative_to(ROOT)
                .as_posix(): _sha256(ROOT / "03_parameters/flowering/cacao_seasonal_cohort_calendar.yml"),
                Path(__file__).resolve().relative_to(ROOT).as_posix(): _sha256(
                    Path(__file__).resolve()
                ),
            },
            "outputs": {
                "country_month_year": ANNUAL_OUTPUT_PATH.relative_to(ROOT).as_posix(),
                "country_month_summary": SUMMARY_OUTPUT_PATH.relative_to(ROOT).as_posix(),
            },
            "output_sha256": {
                ANNUAL_OUTPUT_PATH.name: _sha256(temporary_outputs[0]),
                SUMMARY_OUTPUT_PATH.name: _sha256(temporary_outputs[1]),
            },
        }
        temporary_outputs[2].write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        for temporary, final in zip(temporary_outputs, outputs, strict=True):
            temporary.replace(final)
    except Exception:
        for path in temporary_outputs:
            path.unlink(missing_ok=True)
        raise

    return ANNUAL_OUTPUT_PATH, SUMMARY_OUTPUT_PATH, MANIFEST_OUTPUT_PATH


if __name__ == "__main__":
    annual, summary, manifest = _run_screening()
    print(f"annual: {annual.relative_to(ROOT)}")
    print(f"summary: {summary.relative_to(ROOT)}")
    print(f"manifest: {manifest.relative_to(ROOT)}")
