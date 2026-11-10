from __future__ import annotations

# Consolidated from test_canopy.py
from simple_cacao import canopy_interception


def test_canopy_interception_reaches_half_max_at_i50a():
    assert abs(canopy_interception(680.0, 2016.0, 680.0, 680.0, 0.85) - 0.425) < 1e-12


def test_canopy_interception_reaches_half_max_at_senescence_i50b():
    tt = 2016.0 - 680.0
    assert abs(canopy_interception(tt, 2016.0, 680.0, 680.0, 0.85) - 0.425) < 1e-12


def test_water_stress_can_reduce_canopy_when_severe():
    normal = canopy_interception(1000.0, 2016.0, 680.0, 680.0, 0.85, 1.0)
    stressed = canopy_interception(1000.0, 2016.0, 680.0, 680.0, 0.85, 0.0)
    assert stressed < normal

# Consolidated from test_co2.py
from simple_cacao import co2_factor


def test_co2_factor_matches_original_linear_cap():
    sco2 = 0.0007
    assert co2_factor(350.0, sco2) == 1.0
    assert co2_factor(400.0, sco2) == 1.035
    assert co2_factor(550.0, sco2) == 1.14
    assert co2_factor(700.0, sco2) == 1.245
    assert co2_factor(750.0, sco2) == 1.245


def test_cacao_sco2_value_is_not_rescaled_silently():
    assert co2_factor(400.0, 0.09) == 5.5

# Consolidated from test_generic_example.py
from simple_cacao import CacaoParameters, DailyWeather, SoilParameters, run_scalar


def test_generic_simple_example_produces_positive_yield():
    params = CacaoParameters(
        tsum=160.0,
        hi=0.5,
        i50a=50.0,
        i50b=50.0,
        tbase=6.0,
        topt=26.0,
        rue=1.0,
        i50max_h=0.0,
        i50max_w=0.0,
        t_heat=32.0,
        t_extreme=45.0,
        sco2=0.0007,
        swater=0.0,
        fsolar_max=0.8,
        max_cycle_days=30,
    )
    soil = SoilParameters(awc_mm=100.0)
    weather = [DailyWeather(30.0, 22.0, 20.0, 5.0, 400.0, 4.0) for _ in range(30)]
    result = run_scalar(params, soil, weather)
    assert result.mature
    assert result.biomass_g_m2 > 0.0
    assert result.yield_kg_ha > 0.0

# Consolidated from test_heat.py
from simple_cacao import heat_factor


def test_heat_factor_piecewise_response():
    assert heat_factor(34.0, 35.0, 40.0) == 1.0
    assert heat_factor(35.0, 35.0, 40.0) == 1.0
    assert heat_factor(37.5, 35.0, 40.0) == 0.5
    assert heat_factor(40.0, 35.0, 40.0) == 0.0
    assert heat_factor(45.0, 35.0, 40.0) == 0.0

# Consolidated from test_maturity.py
from simple_cacao import published_cacao_parameters


def test_simulation_stops_at_maturity():
    params = published_cacao_parameters(tsum=32.0)
    soil = SoilParameters(awc_mm=100.0)
    weather = [DailyWeather(30.0, 22.0, 20.0, 5.0, 350.0, 4.0) for _ in range(20)]
    result = run_scalar(params, soil, weather)
    assert result.mature
    assert result.days_run == 2
    assert result.thermal_time == 32.0

# Consolidated from test_published_cacao.py
from dataclasses import replace
from pathlib import Path

import pytest
import yaml
from simple_cacao import CACAO_TABLE_CO2_RUE, relative_suitability_index


def test_suitability_baseline_config_matches_model_defaults():
    config_path = (
        Path(__file__).resolve().parents[2]
        / "03_parameters"
        / "crop"
        / "cacao_suitability_baseline.yml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    params = published_cacao_parameters()
    values = config["crop"]["parameters"]
    expected = {
        "tsum_c_days": params.tsum,
        "hi_fraction": params.hi,
        "i50a_c_days": params.i50a,
        "i50b_c_days": params.i50b,
        "tbase_c": params.tbase,
        "topt_c": params.topt,
        "rue_g_mj": params.rue,
        "i50max_h_c_days_per_day": params.i50max_h,
        "i50max_w_c_days_per_day": params.i50max_w,
        "t_heat_c": params.t_heat,
        "t_extreme_c": params.t_extreme,
        "sco2_per_ppm": params.sco2,
        "swater": params.swater,
        "fsolar_max": params.fsolar_max,
        "initial_fsolar": params.initial_fsolar,
        "max_cycle_days": params.max_cycle_days,
    }
    assert values == expected
    assert config["interpretation"]["actual_yield_prediction"] is False
    assert config["interpretation"]["locally_calibrated"] is False
    soil = config["soil"]
    assert soil["awc_band_by_root_depth_mm"] == {1000: 1, 1500: 2, 2000: 3}
    assert soil["awc_raster_units"] == "mm"
    assert soil["awc_band_by_root_depth_mm"][soil["rooting_depth_mm"]] == 2
    assert config["forcing"]["historical_forcing_years_inclusive"] == [2000, 2020]
    assert config["scenarios"]["historical_cohort_start_years_inclusive"] == [2000, 2019]
    assert config["scenarios"]["future_cohort_start_years_inclusive"] == [2060, 2079]
    assert config["scenarios"]["future_forcing_years_inclusive"] == [2060, 2080]


def test_published_cacao_anchor_values_are_explicit():
    params = published_cacao_parameters(tsum=2016.0, rue=0.6)
    assert params.tsum == 2016.0
    assert params.rue == 0.6
    assert params.hi == 0.30
    assert params.i50a == 680.0
    assert params.i50b == 680.0
    assert params.tbase == 10.0
    assert params.topt == 26.0
    assert params.t_heat == 35.0
    assert params.t_extreme == 40.0
    assert params.swater == 0.0
    assert CACAO_TABLE_CO2_RUE == 0.09
    assert params.sco2 == 0.0


@pytest.mark.parametrize("co2_ppm", [350.0, 400.0, 550.0, 700.0])
def test_approved_cacao_parameters_are_co2_neutral(co2_ppm: float):
    params = published_cacao_parameters()
    assert co2_factor(co2_ppm, params.sco2) == 1.0


def test_multiplicative_scale_parameters_cancel_from_relative_index():
    reference_weather = [
        DailyWeather(26.0, 22.0, 18.0, 4.0, 400.0, 3.0) for _ in range(30)
    ]
    period_weather = [
        DailyWeather(30.0, 24.0, 20.0, 2.0, 400.0, 4.0) for _ in range(30)
    ]
    soil = SoilParameters(awc_mm=150.0)
    reference_params = published_cacao_parameters()
    scaled_params = replace(reference_params, rue=reference_params.rue * 1.5, hi=0.45)

    reference_run = run_scalar(reference_params, soil, reference_weather)
    period_run = run_scalar(reference_params, soil, period_weather)
    scaled_reference_run = run_scalar(scaled_params, soil, reference_weather)
    scaled_period_run = run_scalar(scaled_params, soil, period_weather)

    baseline_index = relative_suitability_index(
        reference_run.yield_kg_ha,
        period_run.yield_kg_ha,
    )
    scaled_index = relative_suitability_index(
        scaled_reference_run.yield_kg_ha,
        scaled_period_run.yield_kg_ha,
    )
    assert baseline_index == pytest.approx(scaled_index)

# Consolidated from test_state_reset.py
from simple_cacao import SimpleCacaoModel


def test_repeated_runs_start_from_fresh_state():
    params = published_cacao_parameters(tsum=64.0)
    soil = SoilParameters(awc_mm=100.0)
    weather = [DailyWeather(30.0, 22.0, 20.0, 5.0, 350.0, 4.0) for _ in range(5)]
    first = run_scalar(params, soil, weather)
    second = run_scalar(params, soil, weather)
    assert first == second


def test_parameters_are_not_mutated_by_senescence_stress():
    params = published_cacao_parameters(tsum=64.0)
    stressed_params = params.__class__(**{**params.__dict__, "i50max_h": 10.0})
    soil = SoilParameters(awc_mm=100.0)
    model = SimpleCacaoModel(stressed_params, soil)
    model.step(DailyWeather(40.0, 30.0, 20.0, 0.0, 350.0, 4.0))
    assert stressed_params.i50b == params.i50b
    assert model.i50b_effective > stressed_params.i50b

# Consolidated from test_suitability.py
import numpy as np


def test_relative_suitability_index_uses_fixed_reference_scale():
    result = relative_suitability_index(
        np.array([10.0, 20.0]),
        np.array([12.0, 10.0]),
    )
    np.testing.assert_allclose(result, [120.0, 50.0])


def test_relative_suitability_index_supports_custom_base():
    assert relative_suitability_index(4.0, 2.0, base_index=1.0) == 0.5


@pytest.mark.parametrize(
    ("reference", "period"),
    [
        (0.0, 1.0),
        (-1.0, 1.0),
        (np.nan, 1.0),
        (1.0, np.nan),
        (1.0, -0.1),
    ],
)
def test_relative_suitability_index_rejects_invalid_inputs(reference, period):
    with pytest.raises(ValueError):
        relative_suitability_index(reference, period)

# Consolidated from test_temperature.py
from simple_cacao import daily_thermal_time, temperature_factor


def test_daily_thermal_time_uses_base_temperature():
    assert daily_thermal_time(26.0, 10.0) == 16.0
    assert daily_thermal_time(8.0, 10.0) == 0.0


def test_temperature_factor_piecewise_response():
    assert temperature_factor(8.0, 10.0, 26.0) == 0.0
    assert temperature_factor(10.0, 10.0, 26.0) == 0.0
    assert temperature_factor(18.0, 10.0, 26.0) == 0.5
    assert temperature_factor(26.0, 10.0, 26.0) == 1.0
    assert temperature_factor(32.0, 10.0, 26.0) == 1.0

# Consolidated from test_units.py
from simple_cacao.units import grams_per_m2_to_kg_per_ha, kg_per_ha_to_grams_per_m2


def test_biomass_yield_unit_conversion():
    assert grams_per_m2_to_kg_per_ha(1.0) == 10.0
    assert kg_per_ha_to_grams_per_m2(10.0) == 1.0

# Consolidated from test_vectorized_model.py
from simple_cacao import run_vectorized_cohort


def test_vectorized_cohorts_match_scalar_model_with_soil_and_heat_stress():
    rng = np.random.default_rng(20261006)
    n_days = 60
    n_cells = 3
    tmin = rng.uniform(18.0, 25.0, size=(n_days, n_cells))
    tmax = tmin + rng.uniform(7.0, 15.0, size=(n_days, n_cells))
    srad = rng.uniform(12.0, 24.0, size=(n_days, n_cells))
    precipitation = rng.uniform(0.0, 12.0, size=(n_days, n_cells))
    et0 = rng.uniform(2.0, 7.0, size=(n_days, n_cells))
    awc = np.array([55.0, 120.0, 240.0])
    params = published_cacao_parameters(tsum=420.0)

    vector_result = run_vectorized_cohort(
        tmax,
        tmin,
        srad,
        precipitation,
        et0,
        awc,
        params,
    )

    scalar_results = []
    for cell in range(n_cells):
        daily_weather = [
            DailyWeather(
                tmax_c=tmax[day, cell],
                tmin_c=tmin[day, cell],
                srad_mj_m2=srad[day, cell],
                precip_mm=precipitation[day, cell],
                co2_ppm=400.0,
                et0_mm=et0[day, cell],
            )
            for day in range(n_days)
        ]
        scalar_results.append(
            run_scalar(
                params,
                SoilParameters(awc_mm=awc[cell]),
                daily_weather,
            )
        )

    np.testing.assert_array_equal(
        vector_result.days_run,
        [result.days_run for result in scalar_results],
    )
    np.testing.assert_array_equal(
        vector_result.mature,
        [result.mature for result in scalar_results],
    )
    np.testing.assert_allclose(
        vector_result.thermal_time,
        [result.thermal_time for result in scalar_results],
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        vector_result.biomass_g_m2,
        [result.biomass_g_m2 for result in scalar_results],
        rtol=1e-12,
    )
    np.testing.assert_array_equal(
        vector_result.heat_stress_days,
        [
            np.count_nonzero(tmax[: result.days_run, cell] > params.t_heat)
            for cell, result in enumerate(scalar_results)
        ],
    )
    np.testing.assert_array_equal(vector_result.water_stress_days, np.zeros(n_cells))


def test_vectorized_cohorts_reject_invalid_weather_or_soil():
    params = published_cacao_parameters()
    weather = np.ones((2, 1))
    with pytest.raises(ValueError, match="nonphysical"):
        run_vectorized_cohort(
            tmax_c=weather,
            tmin_c=weather + 1,
            srad_mj_m2=weather,
            precip_mm=weather,
            et0_mm=weather,
            awc_mm=np.array([100.0]),
            params=params,
        )
    with pytest.raises(ValueError, match="positive"):
        run_vectorized_cohort(
            tmax_c=weather,
            tmin_c=weather - 1,
            srad_mj_m2=weather,
            precip_mm=weather,
            et0_mm=weather,
            awc_mm=np.array([0.0]),
            params=params,
        )


def test_zero_water_sensitivity_removes_precipitation_effect_on_response():
    params = published_cacao_parameters(tsum=500.0)
    n_days = 80
    shape = (n_days, 2)
    tmax = np.full(shape, 30.0)
    tmin = np.full(shape, 22.0)
    radiation = np.full(shape, 18.0)
    precipitation = np.column_stack(
        (np.zeros(n_days), np.full(n_days, 20.0))
    )
    et0 = np.full(shape, 5.0)
    awc = np.array([100.0, 100.0])

    reference = run_vectorized_cohort(
        tmax,
        tmin,
        radiation,
        precipitation,
        et0,
        awc,
        params,
    )
    water_sensitive = run_vectorized_cohort(
        tmax,
        tmin,
        radiation,
        precipitation,
        et0,
        awc,
        replace(params, swater=1.0),
    )

    assert reference.biomass_g_m2[0] == pytest.approx(
        reference.biomass_g_m2[1], rel=0, abs=1e-12
    )
    assert water_sensitive.biomass_g_m2[0] < water_sensitive.biomass_g_m2[1]

# Consolidated from test_water_balance.py
from simple_cacao import WaterBalanceState, arid_index, update_water_balance, water_factor


def test_arid_index_zero_when_et0_is_zero():
    assert arid_index(0.0, 0.0) == 0.0


def test_arid_index_uses_extractable_paw():
    assert arid_index(5.0, 100.0) == 0.0
    assert arid_index(5.0, 10.0) == 1.0 - 0.96 / 5.0


def test_water_factor_clips_at_zero():
    assert water_factor(0.0, 2.0) == 1.0
    assert water_factor(0.75, 2.0) == 0.0


def test_daily_water_balance_mass_closes_without_runoff_or_drainage():
    soil = SoilParameters(awc_mm=100.0, runoff_curve_number=70.0, deep_drainage_coefficient=0.5)
    new_state, fluxes = update_water_balance(
        WaterBalanceState(paw_mm=50.0),
        soil,
        precip_mm=10.0,
        irrigation_mm=0.0,
        et0_mm=4.0,
    )
    lhs = 50.0 + fluxes["input_mm"]
    rhs = (
        new_state.paw_mm + fluxes["runoff_mm"] + fluxes["et_withdrawal_mm"] + fluxes["drainage_mm"]
    )
    assert abs(lhs - rhs) < 1e-9


def test_zero_rain_drydown_increases_arid():
    soil = SoilParameters(awc_mm=100.0, runoff_curve_number=70.0, deep_drainage_coefficient=0.5)
    state = WaterBalanceState(paw_mm=20.0)
    start_arid = arid_index(5.0, state.paw_mm)

    for _ in range(5):
        state, fluxes = update_water_balance(
            state,
            soil,
            precip_mm=0.0,
            irrigation_mm=0.0,
            et0_mm=5.0,
        )

    assert state.paw_mm < 20.0
    assert fluxes["arid"] > start_arid


def test_heavy_rain_activates_runoff_and_drainage():
    soil = SoilParameters(awc_mm=100.0, runoff_curve_number=70.0, deep_drainage_coefficient=0.5)
    new_state, fluxes = update_water_balance(
        WaterBalanceState(paw_mm=90.0),
        soil,
        precip_mm=200.0,
        irrigation_mm=0.0,
        et0_mm=2.0,
    )

    lhs = 90.0 + fluxes["input_mm"]
    rhs = (
        new_state.paw_mm + fluxes["runoff_mm"] + fluxes["et_withdrawal_mm"] + fluxes["drainage_mm"]
    )
    assert fluxes["runoff_mm"] > 0.0
    assert fluxes["drainage_mm"] > 0.0
    assert abs(lhs - rhs) < 1e-9
