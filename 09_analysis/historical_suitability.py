from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

import yaml

# Summary historical analysis
ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_COUNTRY_SUMMARY_PATH = ROOT / "08_outputs/country/historical_country_summary.csv"
HISTORICAL_RUN_MANIFEST_PATH = (
    ROOT
    / "08_outputs/gridded/historical_climatic_suitability/historical_run_manifest.json"
)
CALENDAR_PATH = ROOT / "03_parameters/flowering/cacao_seasonal_cohort_calendar.yml"
HISTORICAL_DIAGNOSTICS_DIR = ROOT / "09_analysis/historical_suitability"
HISTORICAL_DIAGNOSTICS_PATH = HISTORICAL_DIAGNOSTICS_DIR / "baseline_diagnostics.csv"
HISTORICAL_DIAGNOSTICS_MANIFEST_PATH = HISTORICAL_DIAGNOSTICS_DIR / "baseline_diagnostics_manifest.json"

INDEX_COLUMN = "cocoa_fraction_weighted_relative_index"
BIOMASS_COLUMN = "mean_model_biomass_response_g_m2"
HEAT_COLUMN = "mean_heat_stress_days_per_cohort"
WATER_COLUMN = "mean_water_stress_days_per_cohort"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _linear_slope(years: list[int], values: list[float]) -> float:
    if len(years) != len(values) or len(years) < 2:
        raise ValueError("A trend requires at least two paired year/value observations")
    year_mean = fmean(years)
    value_mean = fmean(values)
    denominator = math.fsum((year - year_mean) ** 2 for year in years)
    if denominator == 0:
        raise ValueError("A trend requires at least two distinct years")
    return math.fsum(
        (year - year_mean) * (value - value_mean)
        for year, value in zip(years, values, strict=True)
    ) / denominator


def _month_seasons(calendar: dict[str, Any]) -> dict[int, str]:
    seasons = calendar.get("calendar", {}).get("seasons", [])
    month_seasons: dict[int, str] = {}
    for season in seasons:
        season_id = season["id"]
        for month in season["months"]:
            if month in month_seasons:
                raise ValueError(f"Cohort month {month} is assigned to multiple seasons")
            month_seasons[month] = season_id
    if not month_seasons:
        raise ValueError("The seasonal calendar contains no cohort start months")
    return month_seasons


def _summarize_group(
    records: list[dict[str, str]],
    *,
    country: str,
    month: int,
    season: str,
    expected_years: tuple[int, ...],
) -> dict[str, Any]:
    years = [int(record["cohort_start_year"]) for record in records]
    if sorted(years) != list(expected_years):
        raise ValueError(
            f"Incomplete or duplicate years for {country} month {month}: {sorted(years)}"
        )
    if any(record["season"] != season for record in records):
        raise ValueError(f"Season label mismatch for {country} month {month}")
    if any(float(record["model_input_valid_cocoa_cell_coverage"]) != 1.0 for record in records):
        raise ValueError(f"Invalid model-input coverage for {country} month {month}")

    ordered = sorted(records, key=lambda record: int(record["cohort_start_year"]))
    years = [int(record["cohort_start_year"]) for record in ordered]
    indexes = [float(record[INDEX_COLUMN]) for record in ordered]
    biomass = [float(record[BIOMASS_COLUMN]) for record in ordered]
    heat_days = [float(record[HEAT_COLUMN]) for record in ordered]
    water_days = [float(record[WATER_COLUMN]) for record in ordered]
    numeric_series = indexes + biomass + heat_days + water_days
    if any(not math.isfinite(value) for value in numeric_series):
        raise ValueError(f"Non-finite diagnostics for {country} month {month}")
    if any(value < 0 for value in biomass + heat_days + water_days):
        raise ValueError(f"Negative response or stress diagnostic for {country} month {month}")

    mean_index = fmean(indexes)
    if not math.isclose(mean_index, 100.0, rel_tol=0.0, abs_tol=1e-3):
        raise ValueError(
            f"Historical index mean is not 100 for {country} month {month}: {mean_index}"
        )

    first = ordered[0]
    return {
        "country": country,
        "season": season,
        "cohort_start_month": month,
        "cohort_start_year_first": years[0],
        "cohort_start_year_last": years[-1],
        "cohort_year_count": len(years),
        "mapped_cocoa_cells": int(first["mapped_cocoa_cells"]),
        "mapped_cocoa_source_cell_coverage": float(
            first["mapped_cocoa_source_cell_coverage"]
        ),
        "model_input_valid_cocoa_cell_coverage": 1.0,
        "mean_historical_relative_index": mean_index,
        "interannual_sd_relative_index": stdev(indexes),
        "min_historical_relative_index": min(indexes),
        "max_historical_relative_index": max(indexes),
        "linear_index_trend_points_per_year": _linear_slope(years, indexes),
        "mean_model_biomass_response_g_m2": fmean(biomass),
        "linear_biomass_response_trend_g_m2_per_year": _linear_slope(years, biomass),
        "mean_cohort_days_to_maturity": fmean(
            float(record["mean_cohort_days_to_maturity"]) for record in ordered
        ),
        "mean_heat_stress_days_per_cohort": fmean(heat_days),
        "years_with_nonzero_country_mean_heat_stress": sum(value > 0 for value in heat_days),
        "mean_water_stress_days_per_cohort": fmean(water_days),
        "maximum_water_stress_days_per_cohort": max(water_days),
    }


def build_diagnostics(
    records: list[dict[str, str]],
    *,
    calendar: dict[str, Any],
    start_year: int = 2000,
    end_year: int = 2019,
) -> list[dict[str, Any]]:
    if end_year <= start_year:
        raise ValueError("The cohort-start year range must contain at least two years")
    month_seasons = _month_seasons(calendar)
    expected_years = tuple(range(start_year, end_year + 1))
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for record in records:
        try:
            key = (record["country"], int(record["cohort_start_month"]))
        except (KeyError, ValueError) as exc:
            raise ValueError("Country or cohort start month is missing or invalid") from exc
        grouped[key].append(record)

    countries = sorted({country for country, _ in grouped})
    if set(countries) != {"CIV", "GHA"}:
        raise ValueError(f"Expected GHA and CIV summaries, found {countries}")
    expected_keys = {
        (country, month) for country in countries for month in month_seasons
    }
    if set(grouped) != expected_keys:
        raise ValueError("Summary rows do not cover every country and configured cohort month")

    return [
        _summarize_group(
            grouped[country, month],
            country=country,
            month=month,
            season=month_seasons[month],
            expected_years=expected_years,
        )
        for country in countries
        for month in month_seasons
    ]


def run_historical_summary() -> tuple[Path, Path]:
    with HISTORICAL_COUNTRY_SUMMARY_PATH.open(newline="", encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))
    calendar = yaml.safe_load(CALENDAR_PATH.read_text(encoding="utf-8"))
    run_manifest = json.loads(HISTORICAL_RUN_MANIFEST_PATH.read_text(encoding="utf-8"))
    if run_manifest.get("status") != "complete_historical_cohort_start_analysis":
        raise ValueError("Historical runner manifest is not marked complete")
    year_range = run_manifest["time_windows"]["historical_cohort_start_years_inclusive"]
    if len(year_range) != 2:
        raise ValueError("Historical run_historical_summary manifest has an invalid cohort-start year range")
    diagnostics = build_diagnostics(
        records,
        calendar=calendar,
        start_year=int(year_range[0]),
        end_year=int(year_range[1]),
    )
    HISTORICAL_DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)

    with HISTORICAL_DIAGNOSTICS_PATH.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(diagnostics[0]))
        writer.writeheader()
        writer.writerows(diagnostics)

    manifest = {
        "status": "complete_historical_reference_diagnostics",
        "created_at": datetime.now().astimezone().isoformat(),
        "interpretation": (
            "Descriptive historical diagnostics of the uncalibrated reference response; "
            "not empirical validation, actual yield, or causal trend attribution."
        ),
        "input_sha256": {
            path.relative_to(ROOT).as_posix(): _sha256(path)
            for path in (
                HISTORICAL_COUNTRY_SUMMARY_PATH,
                HISTORICAL_RUN_MANIFEST_PATH,
                CALENDAR_PATH,
                Path(__file__).resolve(),
            )
        },
        "historical_run_status": run_manifest["status"],
        "years": [2000, 2019],
        "cohort_start_months": list(_month_seasons(calendar)),
        "season_aggregation": False,
        "output": HISTORICAL_DIAGNOSTICS_PATH.relative_to(ROOT).as_posix(),
        "output_sha256": _sha256(HISTORICAL_DIAGNOSTICS_PATH),
        "groups": len(diagnostics),
    }
    HISTORICAL_DIAGNOSTICS_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return HISTORICAL_DIAGNOSTICS_PATH, HISTORICAL_DIAGNOSTICS_MANIFEST_PATH

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run historical suitability analyses.")
    parser.add_argument("analysis", choices=("summary",))
    parser.parse_args(argv)
    csv_path, manifest_path = run_historical_summary()
    print(f"diagnostics: {csv_path.relative_to(ROOT)}")
    print(f"manifest: {manifest_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
