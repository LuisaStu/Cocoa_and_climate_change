from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import yaml
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
CLIMATE_DIR = ROOT / "02_preprocessing" / "climate"
if str(CLIMATE_DIR) not in sys.path:
    sys.path.insert(0, str(CLIMATE_DIR))

from run_historical_suitability import (
    ROOT as MODEL_ROOT,
)
from run_historical_suitability import (
    _load_seasonal_calendar,
)
from simple_cacao import published_cacao_parameters
from simple_cacao.vectorized import run_vectorized_cohort
from weather_stream import year_blocks  # type: ignore[reportMissingImports]

if MODEL_ROOT != ROOT:
    raise RuntimeError("Model and map runner roots do not match")

HISTORICAL_RASTER = (
    ROOT
    / "08_outputs/gridded/historical_climatic_suitability/"
    / "historical_cohort_biomass_response.tif"
)
HISTORICAL_MANIFEST = (
    ROOT
    / "08_outputs/gridded/historical_climatic_suitability/"
    / "historical_run_manifest.json"
)
GRID_PATH = ROOT / "01_inputs/processed/grid_full_extent/model_grid_EPSG6933_11000m.tif"
COCOA_PATH = (
    ROOT / "01_inputs/processed/cocoa_mask_full_extent/cocoa_fraction_EPSG6933_11000m.tif"
)
ENSEMBLE_CONFIG = ROOT / "config/gcm_ensemble_proposal.yml"
CALENDAR_PATH = ROOT / "03_parameters/flowering/cacao_seasonal_cohort_calendar.yml"
SCENARIO_MANIFEST = ROOT / "01_inputs/manifests/nex_gddp_scenario_full.json"
SCENARIO_AUDIT = ROOT / "06_validation/historical_gcm/scenario_full_audit.json"
OUTPUT_DIR = ROOT / "08_outputs/gridded/late_century_suitability_change"
FIGURE_PATH = ROOT / "10_figures/maps/late_century_seasonal_suitability_change.png"
START_YEARS = tuple(range(2060, 2080))
SCENARIOS = {"ssp245": "SSP2-4.5", "ssp585": "SSP5-8.5"}
WEATHER_KEYS = ("tmin_c", "tmax_c", "srad_mj_m2", "precip_mm")
NODATA = -9999.0
MODEL_COUNTRY_CSV = "late_century_individual_model_country_change.csv"
MODEL_COUNTRY_FIGURE = "late_century_individual_model_country_change.png"
COUNTRIES = {"GHA": (1, "Ghana", "#0072B2"), "CIV": (2, "Côte d’Ivoire", "#D55E00")}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_change_percent(
    future_month_means: np.ndarray,
    historical_month_means: np.ndarray,
) -> np.ndarray:
    future = np.asarray(future_month_means, dtype=np.float64)
    historical = np.asarray(historical_month_means, dtype=np.float64)
    if future.ndim != 2 or future.shape != historical.shape or future.shape[0] == 0:
        raise ValueError("Future and historical monthly responses must share shape (months, cells)")
    if (
        not np.isfinite(future).all()
        or not np.isfinite(historical).all()
        or np.any(future < 0)
        or np.any(historical <= 0)
    ):
        raise ValueError("Responses must be finite and historical responses must be positive")
    return (future / historical - 1.0) * 100.0


def summarize_ensemble(
    model_month_means: np.ndarray,
    historical_month_means: np.ndarray,
    month_indices: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    model_changes = _seasonal_model_changes(
        model_month_means, historical_month_means, month_indices
    )
    return np.median(model_changes, axis=0), np.mean(model_changes > 0, axis=0)


def _seasonal_model_changes(
    model_month_means: np.ndarray,
    historical_month_means: np.ndarray,
    month_indices: list[int],
) -> np.ndarray:
    values = np.asarray(model_month_means, dtype=np.float64)
    historical = np.asarray(historical_month_means, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != historical.shape or values.shape[0] == 0:
        raise ValueError("Expected model-by-month-by-cell responses matching historical data")
    if (
        not month_indices
        or len(month_indices) != len(set(month_indices))
        or any(index < 0 or index >= values.shape[1] for index in month_indices)
    ):
        raise ValueError("month_indices must be a nonempty list of unique valid month indices")
    return np.stack(
        [
            relative_change_percent(model, historical)[month_indices].mean(axis=0)
            for model in values
        ]
    )


def _weather_columns(grid_codes: np.ndarray, cell_ids: np.ndarray) -> np.ndarray:
    active_cell_ids = np.flatnonzero(np.isin(grid_codes, [1, 2]))
    columns = np.searchsorted(active_cell_ids, cell_ids)
    if (
        np.any(columns >= active_cell_ids.size)
        or not np.array_equal(active_cell_ids[columns], cell_ids)
    ):
        raise ValueError("Mapped-cocoa cells do not align with scenario weather cells")
    return columns


def _read_year(
    model: str,
    scenario: str,
    year: int,
    weather_columns: np.ndarray,
) -> dict[str, Any]:
    blocks = list(year_blocks(model, scenario, year))
    if not blocks:
        raise ValueError(f"No weather blocks for {model} {scenario} {year}")
    result: dict[str, Any] = {
        "month": np.concatenate([block["month"] for block in blocks]),
    }
    for key in WEATHER_KEYS:
        result[key] = np.concatenate([block[key] for block in blocks], axis=0)[
            :, weather_columns
        ]
    return result


def _cohort_weather(
    current: dict[str, Any],
    following: dict[str, Any],
    month: int,
    max_days: int,
) -> dict[str, np.ndarray]:
    starts = np.flatnonzero(current["month"] == month)
    if starts.size == 0:
        raise ValueError(f"Month {month} missing from the future climate calendar")
    start = int(starts[0])
    current_days = min(max_days, current["month"].size - start)
    next_days = max_days - current_days
    weather = {}
    for key in WEATHER_KEYS:
        parts = [current[key][start : start + current_days]]
        if next_days:
            if following[key].shape[0] < next_days:
                raise ValueError("Trailing forcing does not cover a complete model cycle")
            parts.append(following[key][:next_days])
        weather[key] = np.concatenate(parts, axis=0)
    if any(values.shape[0] != max_days for values in weather.values()):
        raise ValueError("Incomplete daily weather in future cohort")
    return weather


def _load_static_inputs() -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    with rasterio.open(GRID_PATH) as grid:
        codes = grid.read(1)
        profile = grid.profile.copy()
    with rasterio.open(COCOA_PATH) as cocoa_src:
        if (
            cocoa_src.shape != codes.shape
            or cocoa_src.transform != profile["transform"]
            or cocoa_src.crs != profile["crs"]
        ):
            raise ValueError("Cocoa map and model grid do not align")
        cocoa_fraction = cocoa_src.read(1, masked=True).filled(np.nan).astype(np.float64)
    selected = np.isin(codes, [1, 2]) & np.isfinite(cocoa_fraction) & (cocoa_fraction > 0)
    cell_ids = np.flatnonzero(selected)
    if not cell_ids.size:
        raise ValueError("No mapped cocoa cells with valid cocoa fractions")
    return profile, codes, cell_ids, cocoa_fraction.reshape(-1)[cell_ids]


def _load_historical_month_means(
    cell_ids: np.ndarray,
    expected_profile: dict[str, Any],
) -> tuple[np.ndarray, list[int]]:
    if not HISTORICAL_RASTER.exists() or not HISTORICAL_MANIFEST.exists():
        raise FileNotFoundError("Run the historical suitability baseline before future mapping")
    historical_manifest = json.loads(HISTORICAL_MANIFEST.read_text(encoding="utf-8"))
    if (
        historical_manifest.get("outputs", {}).get("biomass_response")
        != HISTORICAL_RASTER.relative_to(ROOT).as_posix()
    ):
        raise ValueError("Historical manifest does not identify the expected response raster")
    with rasterio.open(HISTORICAL_RASTER) as source:
        if (
            source.shape != (expected_profile["height"], expected_profile["width"])
            or source.transform != expected_profile["transform"]
            or source.crs != expected_profile["crs"]
            or source.count != 120
        ):
            raise ValueError("Historical response raster does not match the model grid")
        month_values: dict[int, list[np.ndarray]] = {}
        seen: set[tuple[int, int]] = set()
        for band, description in enumerate(source.descriptions, start=1):
            parts = description.split("; ")
            if len(parts) != 3 or parts[2] != "biomass_g_m2":
                raise ValueError("Invalid historical response band metadata")
            month = int(parts[0].split("=", 1)[1])
            year = int(parts[1].split("=", 1)[1])
            if (month, year) in seen:
                raise ValueError("Invalid or duplicate historical response band metadata")
            seen.add((month, year))
            values = (
                source.read(band, masked=True)
                .filled(np.nan)
                .reshape(-1)[cell_ids]
                .astype(np.float64)
            )
            if not np.isfinite(values).all() or np.any(values <= 0):
                raise ValueError("Invalid historical response in mapped cocoa cells")
            month_values.setdefault(month, []).append(values)
        if len(seen) != 120 or any(len(values) != 20 for values in month_values.values()):
            raise ValueError("Historical response must cover six months and 20 years each")
    months = sorted(month_values)
    historical_means = np.stack(
        [np.mean(month_values[month], axis=0) for month in months]
    )
    return historical_means, months


def _run_model(
    model: str,
    scenario: str,
    months: list[int],
    cell_ids: np.ndarray,
    weather_columns: np.ndarray,
    awc_mm: np.ndarray,
) -> np.ndarray:
    parameters = published_cacao_parameters()
    response_sums = np.zeros((len(months), cell_ids.size), dtype=np.float64)
    current = _read_year(model, scenario, START_YEARS[0], weather_columns)
    for year in START_YEARS:
        following = _read_year(model, scenario, year + 1, weather_columns)
        for month_index, month in enumerate(months):
            weather = _cohort_weather(current, following, month, parameters.max_cycle_days)
            simulation = run_vectorized_cohort(
                weather["tmax_c"],
                weather["tmin_c"],
                weather["srad_mj_m2"],
                weather["precip_mm"],
                np.zeros_like(weather["precip_mm"]),
                awc_mm,
                parameters,
            )
            capped = (simulation.days_run >= parameters.max_cycle_days) & (
                simulation.thermal_time < parameters.tsum
            )
            if capped.any():
                raise ValueError(
                    f"{model} {scenario} {year}-{month:02d}: "
                    f"{int(capped.sum())} cells hit the cycle cap before thermal maturity"
                )
            if not np.isfinite(simulation.biomass_g_m2).all():
                raise ValueError(f"Non-finite model response for {model} {scenario} {year}")
            response_sums[month_index] += simulation.biomass_g_m2
        current = following
        print(f"Completed {model} {scenario} cohort-start year {year}", flush=True)
    return response_sums / len(START_YEARS)


def _write_geotiff(
    path: Path,
    values: np.ndarray,
    cell_ids: np.ndarray,
    profile: dict[str, Any],
    description: str,
) -> None:
    output = np.full(profile["height"] * profile["width"], NODATA, dtype=np.float32)
    output[cell_ids] = values.astype(np.float32)
    output_profile = profile.copy()
    output_profile.update(
        count=1,
        dtype="float32",
        nodata=NODATA,
        compress="deflate",
        predictor=3,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **output_profile) as target:
        target.write(output.reshape(profile["height"], profile["width"]), 1)
        target.set_band_description(1, description)
        target.update_tags(
            units="percent",
            interpretation="modeled climatic-suitability proxy change, not actual yield",
            period="2060-2079 cohort-start years versus 2000-2019 historical reference",
        )


def _plot_maps(
    seasonal_maps: dict[str, dict[str, np.ndarray]],
    season_labels: dict[str, str],
    season_months: dict[str, list[int]],
    cell_ids: np.ndarray,
    profile: dict[str, Any],
    grid_codes: np.ndarray,
    figure_path: Path,
) -> None:
    plot_values = [
        values
        for scenario_maps in seasonal_maps.values()
        for values in scenario_maps.values()
    ]
    limit = max(1.0, float(np.nanpercentile(np.abs(np.concatenate(plot_values)), 98)))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    extent = (
        profile["transform"].c,
        profile["transform"].c + profile["width"] * profile["transform"].a,
        profile["transform"].f + profile["height"] * profile["transform"].e,
        profile["transform"].f,
    )
    figure, axes = plt.subplots(
        len(seasonal_maps),
        len(SCENARIOS),
        figsize=(13, 11),
        squeeze=False,
        constrained_layout=True,
    )
    cmap = plt.get_cmap("RdYlBu_r").copy()
    cmap.set_bad("white")
    image = None
    rows, cols = np.indices(grid_codes.shape)
    x = profile["transform"].c + (cols + 0.5) * profile["transform"].a
    y = profile["transform"].f + (rows + 0.5) * profile["transform"].e
    for row, (season, scenario_maps) in enumerate(seasonal_maps.items()):
        for col, (scenario, values) in enumerate(scenario_maps.items()):
            axis = axes[row, col]
            raster_values = np.full(
                profile["height"] * profile["width"], np.nan, dtype=np.float32
            )
            raster_values[cell_ids] = values
            image = axis.imshow(
                raster_values.reshape(profile["height"], profile["width"]),
                extent=extent,
                origin="upper",
                cmap=cmap,
                norm=norm,
                interpolation="nearest",
            )
            axis.contour(
                x, y, grid_codes == 1, levels=[0.5], colors="black", linewidths=0.45
            )
            axis.contour(
                x, y, grid_codes == 2, levels=[0.5], colors="black", linewidths=0.45
            )
            months = ", ".join(f"{month:02d}" for month in season_months[season])
            axis.set_title(f"{season_labels[season]} ({months}) | {SCENARIOS[scenario]}")
            axis.set_aspect("equal")
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_frame_on(False)
    if image is None:
        raise ValueError("No scenario maps to plot")
    colorbar = figure.colorbar(image, ax=axes, shrink=0.78, pad=0.02)
    colorbar.set_label("Change in modeled climatic response (%)")
    figure.suptitle(
        "Late-century cocoa climatic-suitability change by flowering season\n"
        "2060–2079 versus 2000–2019 | median of eight climate models",
        fontsize=15,
    )
    figure.text(
        0.5,
        0.015,
        "Each panel equally averages the three monthly start cases within that rainy-season window. "
        "This is a timing sensitivity, not flowering-frequency weighting or farm yield. "
        "Water-stress and excess-rainfall damage are not represented.",
        ha="center",
        fontsize=9,
    )
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _write_model_country_outputs(
    changes: dict[tuple[str, str, str], np.ndarray],
    models: list[str],
    season_labels: dict[str, str],
    table_path: Path,
    figure_path: Path,
) -> None:
    with table_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["scenario", "flowering_window", "climate_model", "country_iso3", "change_percent"]
        )
        for (scenario, season, country), values in changes.items():
            writer.writerows(
                (scenario, season, model, country, value)
                for model, value in zip(models, values, strict=True)
            )

    figure, axes = plt.subplots(
        2, 2, figsize=(10, 8), sharex=True, sharey=True, layout="constrained"
    )
    positions = np.arange(len(models))
    for row, (season, label) in enumerate(season_labels.items()):
        for column, (scenario, scenario_label) in enumerate(SCENARIOS.items()):
            ax = axes[row, column]
            ax.axvline(0, color="#555555", linewidth=0.9)
            ax.grid(axis="x", color="#dddddd", linewidth=0.6)
            ax.set_axisbelow(True)
            for offset, (country, (_, name, color)) in zip(
                (-0.14, 0.14), COUNTRIES.items(), strict=True
            ):
                ax.scatter(
                    changes[scenario, season, country], positions + offset, color=color, label=name
                )
            ax.set_title(f"{scenario_label} · {label.lower()} season", loc="left")
            ax.set_yticks(positions, models)
    axes[0, 0].invert_yaxis()
    axes[0, 0].legend(loc="best", frameon=False)
    figure.suptitle(
        "Late-century cocoa response by climate model\n"
        "2060–2079 compared with 2000–2019 cohort starts",
        fontsize=15,
        fontweight="bold",
    )
    figure.supxlabel("Cocoa-area-weighted change in modeled cohort biomass (%)")
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=220)
    plt.close(figure)


def _run(
    output_dir: Path = OUTPUT_DIR,
    figure_path: Path = FIGURE_PATH,
) -> tuple[Path, dict[str, Path], Path]:
    season_ids = ["major_rainy", "minor_rainy"]
    current_output_paths = [
        output_dir / f"{scenario}_{season}_2060_2079_median_change_percent.tif"
        for scenario in SCENARIOS
        for season in season_ids
    ]
    manifest_path = output_dir / "late_century_map_manifest.json"
    table_path = output_dir / MODEL_COUNTRY_CSV
    comparison_path = figure_path.with_name(MODEL_COUNTRY_FIGURE)
    additional_paths = (figure_path, manifest_path, table_path, comparison_path)
    if any(path.exists() for path in (*current_output_paths, *additional_paths)):
        raise FileExistsError(
            "Seasonal map outputs already exist. Use --output-dir for a separate run: "
            f"{output_dir} or {figure_path}"
        )
    ensemble_config = yaml.safe_load(ENSEMBLE_CONFIG.read_text(encoding="utf-8"))
    models = ensemble_config["primary"]["models"]
    if not models or len(models) != len(set(models)):
        raise ValueError("The configured climate-model list must be nonempty and unique")
    calendar, _, start_day = _load_seasonal_calendar()
    if start_day != 1:
        raise ValueError("The future map currently requires first-of-month cohort starts")
    season_months = {
        season["id"]: sorted(season["months"])
        for season in calendar["calendar"]["seasons"]
    }
    expected_season_months = {
        "major_rainy": [4, 5, 6],
        "minor_rainy": [9, 10, 11],
    }
    if season_months != expected_season_months:
        raise ValueError(f"Unexpected seasonal cohort calendar: {season_months}")
    season_labels = {
        season["id"]: season["id"].replace("_", " ").title()
        for season in calendar["calendar"]["seasons"]
    }
    profile, grid_codes, cell_ids, cocoa_weights = _load_static_inputs()
    weather_columns = _weather_columns(grid_codes, cell_ids)
    historical_means, months = _load_historical_month_means(cell_ids, profile)
    if months != [4, 5, 6, 9, 10, 11]:
        raise ValueError(f"Unexpected historical cohort months: {months}")
    soil_path = ROOT / "01_inputs/processed/soil_full_extent/soil_awc_EPSG6933_11000m.tif"
    with rasterio.open(soil_path) as soil:
        if (
            soil.shape != (profile["height"], profile["width"])
            or soil.transform != profile["transform"]
            or soil.crs != profile["crs"]
            or soil.count != 3
            or soil.descriptions[1] != "awc_root_depth_1500mm"
        ):
            raise ValueError("Soil AWC input does not match the published reference configuration")
        awc_mm = (
            soil.read(2, masked=True)
            .filled(np.nan)
            .reshape(-1)[cell_ids]
            .astype(np.float64)
        )
    if not np.isfinite(awc_mm).all() or np.any(awc_mm <= 0):
        raise ValueError("Invalid available-water capacity on mapped cocoa cells")

    seasonal_maps: dict[str, dict[str, np.ndarray]] = {
        season: {} for season in season_months
    }
    direction_agreement: dict[str, dict[str, np.ndarray]] = {
        season: {} for season in season_months
    }
    country_codes = grid_codes.reshape(-1)[cell_ids]
    model_country_changes: dict[tuple[str, str, str], np.ndarray] = {}
    for scenario in SCENARIOS:
        model_month_means = []
        for model in models:
            model_month_means.append(
                _run_model(model, scenario, months, cell_ids, weather_columns, awc_mm)
            )
        model_month_means_array = np.stack(model_month_means)
        for season, season_month_list in season_months.items():
            month_indices = [months.index(month) for month in season_month_list]
            changes = _seasonal_model_changes(
                model_month_means_array,
                historical_means,
                month_indices,
            )
            seasonal_maps[season][scenario] = np.median(changes, axis=0)
            direction_agreement[season][scenario] = np.mean(changes > 0, axis=0)
            for country, (code, _, _) in COUNTRIES.items():
                selected = country_codes == code
                model_country_changes[scenario, season, country] = np.average(
                    changes[:, selected], axis=1, weights=cocoa_weights[selected]
                )

    outputs = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    for season, scenario_maps in seasonal_maps.items():
        for scenario, values in scenario_maps.items():
            path = output_dir / (
                f"{scenario}_{season}_2060_2079_median_change_percent.tif"
            )
            _write_geotiff(
                path,
                values,
                cell_ids,
                profile,
                f"{season_labels[season]} {SCENARIOS[scenario]} ensemble median change, percent",
            )
            outputs[f"{scenario}_{season}"] = path
    _plot_maps(
        seasonal_maps,
        season_labels,
        season_months,
        cell_ids,
        profile,
        grid_codes,
        figure_path,
    )
    country_summary = {}
    for season, scenario_maps in seasonal_maps.items():
        country_summary[season] = {}
        for scenario, values in scenario_maps.items():
            country_summary[season][scenario] = {}
            for country, (code, _, _) in COUNTRIES.items():
                selected = country_codes == code
                weights = cocoa_weights[selected] / cocoa_weights[selected].sum()
                country_summary[season][scenario][country] = {
                    "median_map_mean_change_percent": float(
                        np.average(values[selected], weights=weights)
                    ),
                    "gcm_direction_agreement_fraction_mean": float(
                        np.average(
                            direction_agreement[season][scenario][selected],
                            weights=weights,
                        )
                    ),
                }
    _write_model_country_outputs(
        model_country_changes, models, season_labels, table_path, comparison_path
    )
    manifest = {
        "status": "complete_late_century_modeled_suitability_map",
        "interpretation": (
            "Modelled climatic-response proxy under the published no-fit parameterization; "
            "not actual or predicted cocoa yield."
        ),
        "seasonal_summary": (
            "For each climate model, mean relative change across the three unweighted "
            "monthly start cases within each seasonal window; then median across eight "
            "equally weighted climate models. This is a timing sensitivity, not flowering-"
            "frequency weighting or an annual-production estimate."
        ),
        "cohort_start_years": [START_YEARS[0], START_YEARS[-1]],
        "historical_reference_cohort_start_years": [2000, 2019],
        "scenarios": SCENARIOS,
        "climate_models": models,
        "ensemble_summary": "median across eight equally weighted GCMs",
        "monthly_start_cases": months,
        "seasonal_months": season_months,
        "monthly_aggregation": "equal-weighted within each seasonal window; no flowering-intensity weights",
        "country_summary_definition": (
            "Cocoa-area-weighted mean of the cellwise ensemble median. "
            "The direction fraction is the cocoa-area-weighted model fraction with "
            "strictly positive change in each cell."
        ),
        "parameters": "published cacao anchors; SCO2=0; S_water=0; AWC 1500 mm",
        "water_and_rainfall_note": (
            "The reference S_water=0 disables water-deficit biomass response. The four-variable "
            "CMIP6 input set omits ET0 and does not represent drought or excess-rainfall damage."
        ),
        "country_cocoa_weighted_summary": country_summary,
        "inputs_sha256": {
            path.relative_to(ROOT).as_posix(): _sha256(path)
            for path in (
                HISTORICAL_RASTER,
                HISTORICAL_MANIFEST,
                GRID_PATH,
                COCOA_PATH,
                soil_path,
                ENSEMBLE_CONFIG,
                CALENDAR_PATH,
                SCENARIO_MANIFEST,
                SCENARIO_AUDIT,
            )
        },
        "outputs_sha256": {
            **{path.name: _sha256(path) for path in outputs.values()},
            table_path.name: _sha256(table_path),
            comparison_path.relative_to(ROOT).as_posix(): _sha256(comparison_path),
            figure_path.relative_to(ROOT).as_posix(): _sha256(figure_path),
        },
        "individual_model_outputs": {
            "table": table_path.relative_to(ROOT).as_posix(),
            "figure": comparison_path.relative_to(ROOT).as_posix(),
            "definition": "Cocoa-area-weighted means of individual model seasonal cell changes.",
        },
        "figure": figure_path.relative_to(ROOT).as_posix(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return figure_path, outputs, manifest_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="Write all outputs to a new project directory")
    args = parser.parse_args()
    if args.output_dir is None:
        figure, rasters, manifest = _run()
    else:
        directory = (ROOT / args.output_dir).resolve()
        if not directory.is_relative_to(ROOT):
            parser.error("--output-dir must be inside the project")
        figure, rasters, manifest = _run(directory, directory / FIGURE_PATH.name)
    print(f"map: {figure.relative_to(ROOT)}")
    for scenario, raster in rasters.items():
        print(f"{scenario}: {raster.relative_to(ROOT)}")
    print(f"individual model CSV: {(manifest.parent / MODEL_COUNTRY_CSV).relative_to(ROOT)}")
    print(f"individual model figure: {figure.with_name(MODEL_COUNTRY_FIGURE).relative_to(ROOT)}")
    print(f"manifest: {manifest.relative_to(ROOT)}")
