from __future__ import annotations

# Consolidated from test_cacao_cohort_calendar.py
from pathlib import Path

import numpy as np
import yaml
from run_historical_suitability import _load_seasonal_calendar


def test_seasonal_cohort_calendar_has_two_unweighted_rainy_seasons():
    config_path = (
        Path(__file__).resolve().parents[2]
        / "03_parameters"
        / "flowering"
        / "cacao_seasonal_cohort_calendar.yml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seasons = {season["id"]: season for season in config["calendar"]["seasons"]}

    assert set(seasons) == {"major_rainy", "minor_rainy"}
    assert seasons["major_rainy"]["months"] == [4, 5, 6]
    assert seasons["minor_rainy"]["months"] == [9, 10, 11]
    assert config["calendar"]["monthly_weights"] is None
    assert config["calendar"]["annual_production_aggregation"] is False
    assert config["status"] == "regional_literature_proxy_not_locally_validated"


def test_historical_runner_uses_the_configured_cohort_calendar():
    config, season_by_month, start_day = _load_seasonal_calendar()

    assert start_day == config["calendar"]["start_day_of_month"] == 1
    assert season_by_month == {
        4: "major_rainy",
        5: "major_rainy",
        6: "major_rainy",
        9: "minor_rainy",
        10: "minor_rainy",
        11: "minor_rainy",
    }

# Consolidated from test_historical_suitability_diagnostics.py
import sys

import pytest

ANALYSIS__historical_suitability_diagnostics = Path(__file__).resolve().parents[2] / "09_analysis"
if str(ANALYSIS__historical_suitability_diagnostics) not in sys.path:
    sys.path.insert(0, str(ANALYSIS__historical_suitability_diagnostics))

from historical_suitability import _linear_slope, build_diagnostics

CALENDAR = {
    "calendar": {
        "seasons": [
            {"id": "major_rainy", "months": [4, 5, 6]},
            {"id": "minor_rainy", "months": [9, 10, 11]},
        ]
    }
}


def _records():
    rows = []
    for country, cells, source_coverage in (("GHA", 1168, 1.0), ("CIV", 1373, 0.999621)):
        for month, season in (
            (4, "major_rainy"),
            (5, "major_rainy"),
            (6, "major_rainy"),
            (9, "minor_rainy"),
            (10, "minor_rainy"),
            (11, "minor_rainy"),
        ):
            for year in range(2000, 2020):
                rows.append(
                    {
                        "country": country,
                        "cohort_start_month": str(month),
                        "cohort_start_year": str(year),
                        "season": season,
                        "mapped_cocoa_cells": str(cells),
                        "mapped_cocoa_source_cell_coverage": str(source_coverage),
                        "model_input_valid_cocoa_cell_coverage": "1.0",
                        "cocoa_fraction_weighted_relative_index": "100.0",
                        "mean_model_biomass_response_g_m2": str(200 + year - 2000),
                        "mean_heat_stress_days_per_cohort": "0.0",
                        "mean_water_stress_days_per_cohort": "0.0",
                        "mean_cohort_days_to_maturity": "90.0",
                    }
                )
    return rows


def test_linear_slope_uses_value_units_per_calendar_year():
    assert _linear_slope([2000, 2001, 2002], [1.0, 3.0, 5.0]) == pytest.approx(2.0)


def test_build_diagnostics_keeps_each_country_and_month_separate():
    diagnostics = build_diagnostics(_records(), calendar=CALENDAR)

    assert len(diagnostics) == 12
    assert {(row["country"], row["cohort_start_month"]) for row in diagnostics} == {
        (country, month)
        for country in ("GHA", "CIV")
        for month in (4, 5, 6, 9, 10, 11)
    }
    assert all(row["mean_historical_relative_index"] == pytest.approx(100.0) for row in diagnostics)
    assert all(row["linear_index_trend_points_per_year"] == pytest.approx(0.0) for row in diagnostics)
    assert all(row["linear_biomass_response_trend_g_m2_per_year"] == pytest.approx(1.0) for row in diagnostics)


def test_build_diagnostics_rejects_missing_or_duplicate_cohort_years():
    records = _records()
    records.pop()
    with pytest.raises(ValueError, match="Incomplete or duplicate years"):
        build_diagnostics(records, calendar=CALENDAR)

    records = _records()
    records.append(records[0].copy())
    with pytest.raises(ValueError, match="Incomplete or duplicate years"):
        build_diagnostics(records, calendar=CALENDAR)


def test_build_diagnostics_rejects_incomplete_model_input_coverage():
    records = _records()
    records[0]["model_input_valid_cocoa_cell_coverage"] = "0.99"

    with pytest.raises(ValueError, match="Invalid model-input coverage"):
        build_diagnostics(records, calendar=CALENDAR)

# Consolidated from test_historical_suitability_sensitivity.py
from dataclasses import replace

MODEL_DIR__historical_suitability_sensitivity = Path(__file__).resolve().parents[1]
if str(MODEL_DIR__historical_suitability_sensitivity) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR__historical_suitability_sensitivity))

from run_historical_suitability_sensitivity import (
    _add_reference_differences,
    _validate_parameters,
    build_sensitivity_cases,
)
from simple_cacao import published_cacao_parameters


def test_sensitivity_cases_follow_exploratory_manifest_without_combinations():
    config_path = (
        Path(__file__).resolve().parents[2]
        / "03_parameters"
        / "crop"
        / "cacao_suitability_sensitivity.yml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cases = build_sensitivity_cases(
        config,
        published_cacao_parameters(),
        reference_rooting_depth_mm=1500,
    )

    assert len(cases) == 19
    assert cases[0].case_id == "published_reference"
    assert cases[0].parameters.swater == 0.0
    assert cases[0].rooting_depth_mm == 1500
    assert {
        case.case_id for case in cases if case.varied_parameter == "rooting_depth_mm"
    } == {"rooting_depth_mm_low", "rooting_depth_mm_high"}
    assert {
        case.parameter_value
        for case in cases
        if case.varied_parameter == "swater"
    } == {0.5, 2.5}
    assert all(case.level in {"reference", "low", "high"} for case in cases)


def test_sensitivity_case_builder_rejects_reference_drift():
    config = {
        "parameters": {
            "tbase_c": {"reference": 11.0, "range": [9.0, 13.0]},
        }
    }

    with pytest.raises(ValueError, match="reference mismatch"):
        build_sensitivity_cases(
            config,
            published_cacao_parameters(),
            reference_rooting_depth_mm=1500,
        )


def test_parameter_guard_rejects_inconsistent_heat_thresholds():
    parameters = published_cacao_parameters()
    with pytest.raises(ValueError, match="t_heat"):
        _validate_parameters(replace(parameters, t_heat=40.0, t_extreme=40.0))


def test_reference_differences_match_country_month_and_year():
    annual_rows = [
        {
            "case_id": "published_reference",
            "country": "Ghana",
            "cohort_start_month": 4,
            "cohort_start_year": 2000,
            "mean_relative_index": 98.0,
        },
        {
            "case_id": "tbase_low",
            "country": "Ghana",
            "cohort_start_month": 4,
            "cohort_start_year": 2000,
            "mean_relative_index": 101.5,
        },
    ]
    summary_rows = [
        {
            "case_id": "published_reference",
            "country": "Ghana",
            "cohort_start_month": 4,
            "mean_historical_relative_index": 100.0,
            "linear_index_trend_points_per_year": 0.2,
        },
        {
            "case_id": "tbase_low",
            "country": "Ghana",
            "cohort_start_month": 4,
            "mean_historical_relative_index": 100.0,
            "linear_index_trend_points_per_year": 0.5,
        },
    ]

    _add_reference_differences(annual_rows, summary_rows)

    assert [row["relative_index_difference_from_published_reference"] for row in annual_rows] == [
        0.0,
        3.5,
    ]
    assert summary_rows[1][
        "mean_historical_relative_index_difference_from_published_reference"
    ] == 0.0
    assert summary_rows[1][
        "index_trend_difference_from_published_reference_points_per_year"
    ] == pytest.approx(0.3)


def test_reference_difference_rejects_missing_reference_rows():
    with pytest.raises(ValueError, match="Published-reference rows"):
        _add_reference_differences(
            [{"case_id": "tbase_low"}],
            [{"case_id": "tbase_low"}],
        )

# Consolidated from test_late_century_suitability_map.py
MODEL_DIR__late_century_suitability_map = Path(__file__).resolve().parents[1]
if str(MODEL_DIR__late_century_suitability_map) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR__late_century_suitability_map))

from run_late_century_suitability_map import (
    WEATHER_KEYS,
    _cohort_weather,
    _weather_columns,
    relative_change_percent,
    summarize_ensemble,
)


def test_weather_columns_selects_the_mapped_cocoa_cells():
    grid_codes = np.array([[0, 1, 1], [2, 2, 0]])
    cocoa_cell_ids = np.array([1, 3])

    columns = _weather_columns(grid_codes, cocoa_cell_ids)

    np.testing.assert_array_equal(columns, [0, 2])


def test_cohort_weather_joins_current_and_trailing_year():
    current = {"month": np.array([4, 4, 5])}
    following = {}
    for key in WEATHER_KEYS:
        current[key] = np.array([[0.0], [1.0], [2.0]])
        following[key] = np.array([[10.0], [11.0], [12.0]])

    weather = _cohort_weather(current, following, month=4, max_days=5)

    np.testing.assert_array_equal(weather["tmax_c"].ravel(), [0, 1, 2, 10, 11])


def test_relative_change_is_calculated_against_month_specific_reference():
    historical = np.array([[10.0, 20.0], [20.0, 40.0]])
    future = np.array([[12.0, 18.0], [18.0, 44.0]])

    changes = relative_change_percent(future, historical)

    np.testing.assert_allclose(changes, [[20.0, -10.0], [-10.0, 10.0]])


def test_ensemble_summary_returns_median_and_direction_agreement():
    historical = np.full((4, 2), 10.0)
    model_month_means = np.array(
        [
            [[11.0, 9.0], [11.0, 9.0], [50.0, 50.0], [50.0, 50.0]],
            [[12.0, 8.0], [12.0, 8.0], [50.0, 50.0], [50.0, 50.0]],
            [[9.0, 11.0], [9.0, 11.0], [50.0, 50.0], [50.0, 50.0]],
        ]
    )

    median, agreement = summarize_ensemble(
        model_month_means,
        historical,
        month_indices=[0, 1],
    )

    np.testing.assert_allclose(median, [10.0, -10.0])
    np.testing.assert_allclose(agreement, [2 / 3, 1 / 3])


@pytest.mark.parametrize("month_indices", [[], [0, 0], [-1], [2]])
def test_ensemble_summary_rejects_invalid_month_selection(month_indices):
    with pytest.raises(ValueError, match="month_indices"):
        summarize_ensemble(
            np.ones((1, 2, 1)),
            np.ones((2, 1)),
            month_indices,
        )


@pytest.mark.parametrize(
    ("future", "historical"),
    [
        (np.array([1.0]), np.array([1.0])),
        (np.array([[1.0]]), np.array([[0.0]])),
        (np.array([[np.nan]]), np.array([[1.0]])),
        (np.array([[-1.0]]), np.array([[1.0]])),
    ],
)
def test_relative_change_rejects_invalid_response_inputs(future, historical):
    with pytest.raises(ValueError):
        relative_change_percent(future, historical)
