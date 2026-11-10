from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .parameters import CacaoParameters


@dataclass(frozen=True)
class VectorSimulationResult:
    days_run: NDArray[np.int32]
    mature: NDArray[np.bool_]
    thermal_time: NDArray[np.float64]
    biomass_g_m2: NDArray[np.float64]
    heat_stress_days: NDArray[np.int32]
    water_stress_days: NDArray[np.int32]


def run_vectorized_cohort(
    tmax_c: ArrayLike,
    tmin_c: ArrayLike,
    srad_mj_m2: ArrayLike,
    precip_mm: ArrayLike,
    et0_mm: ArrayLike,
    awc_mm: ArrayLike,
    params: CacaoParameters,
    *,
    co2_ppm: float = 400.0,
    runoff_curve_number: float = 70.0,
) -> VectorSimulationResult:
    """Run independent scalar-equivalent cohorts over a shared daily time axis.

    Weather arrays have shape ``(days, cells)``. Available water capacity is
    cell-specific, in millimetres. Only the cell dimension is vectorized; each
    cell retains the scalar model's independent soil and phenology state.
    """
    weather = [
        np.asarray(values, dtype=np.float64)
        for values in (tmax_c, tmin_c, srad_mj_m2, precip_mm, et0_mm)
    ]
    if weather[0].ndim != 2 or weather[0].shape[0] == 0:
        raise ValueError("weather arrays must have shape (days, cells)")
    if any(values.shape != weather[0].shape for values in weather):
        raise ValueError("weather arrays must share the same shape")
    if any(not np.isfinite(values).all() for values in weather):
        raise ValueError("weather arrays must contain only finite values")
    tmax, tmin, srad, precipitation, et0 = weather
    if np.any(tmin > tmax) or np.any(srad < 0) or np.any(precipitation < 0) or np.any(et0 < 0):
        raise ValueError("weather arrays contain nonphysical values")

    awc = np.asarray(awc_mm, dtype=np.float64)
    if awc.shape != (weather[0].shape[1],):
        raise ValueError("awc_mm must have one value per weather cell")
    if not np.isfinite(awc).all() or np.any(awc <= 0):
        raise ValueError("awc_mm values must be finite and positive")
    if not np.isfinite(co2_ppm):
        raise ValueError("co2_ppm must be finite")
    if not 0.0 < runoff_curve_number < 100.0:
        raise ValueError("runoff_curve_number must be between 0 and 100")

    n_cells = awc.size
    paw = awc.copy()
    thermal_time = np.zeros(n_cells, dtype=np.float64)
    biomass = np.zeros(n_cells, dtype=np.float64)
    days_run = np.zeros(n_cells, dtype=np.int32)
    heat_stress_days = np.zeros(n_cells, dtype=np.int32)
    water_stress_days = np.zeros(n_cells, dtype=np.int32)
    i50b_effective = np.full(n_cells, params.i50b, dtype=np.float64)
    active = np.ones(n_cells, dtype=bool)

    storage = 25400.0 / runoff_curve_number - 254.0
    initial_abstraction = 0.2 * storage

    for day in range(weather[0].shape[0]):
        if not active.any():
            break
        indices = np.flatnonzero(active)
        rain = precipitation[day, indices]
        water_input = np.maximum(0.0, rain)
        runoff = np.zeros_like(water_input)
        runoff_mask = water_input > initial_abstraction
        runoff[runoff_mask] = (
            (water_input[runoff_mask] - initial_abstraction) ** 2
            / (water_input[runoff_mask] + 0.8 * storage)
        )
        infiltration = np.maximum(0.0, water_input - runoff)
        available_before_et = (
            np.minimum(np.maximum(0.0, paw[indices]), awc[indices]) + infiltration
        )
        et_withdrawal = np.minimum(
            np.maximum(0.0, et0[day, indices]), 0.096 * available_before_et
        )
        after_et = available_before_et - et_withdrawal
        final_paw = np.minimum(after_et, awc[indices])

        arid = np.zeros_like(final_paw)
        positive_et = et0[day, indices] > 0.0
        arid[positive_et] = np.clip(
            1.0
            - np.minimum(
                et0[day, indices][positive_et],
                0.096 * np.maximum(0.0, final_paw[positive_et]),
            )
            / et0[day, indices][positive_et],
            0.0,
            1.0,
        )
        f_water = np.maximum(0.0, 1.0 - params.swater * arid)

        tmean = (tmax[day, indices] + tmin[day, indices]) / 2.0
        dtt = np.maximum(0.0, tmean - params.tbase)
        f_temp = np.clip((tmean - params.tbase) / (params.topt - params.tbase), 0.0, 1.0)
        f_heat = np.ones_like(tmean)
        heat_mask = tmax[day, indices] > params.t_heat
        between_heat_extreme = heat_mask & (tmax[day, indices] < params.t_extreme)
        f_heat[tmax[day, indices] >= params.t_extreme] = 0.0
        f_heat[between_heat_extreme] = (
            (params.t_extreme - tmax[day, indices][between_heat_extreme])
            / (params.t_extreme - params.t_heat)
        )
        capped_co2 = min(co2_ppm, 700.0)
        f_co2 = 1.0 + params.sco2 * max(0.0, capped_co2 - 350.0)

        current_tt = thermal_time[indices]
        f_solar = np.empty_like(current_tt)
        first_phase = current_tt <= params.tsum / 2.0
        f_solar[first_phase] = params.fsolar_max / (
            1.0 + np.exp(-0.01 * (current_tt[first_phase] - params.i50a))
        )
        f_solar[~first_phase] = params.fsolar_max / (
            1.0
            + np.exp(
                0.01 * (current_tt[~first_phase] - (params.tsum - i50b_effective[indices][~first_phase]))
            )
        )
        low_water = f_water < 0.1
        f_solar[low_water] += f_water[low_water] - 0.1
        f_solar = np.clip(f_solar, 0.0, params.fsolar_max)

        stress = np.minimum(f_heat, f_water)
        heat_stress_days[indices] += f_heat < 1.0
        water_stress_days[indices] += f_water < 1.0
        biomass[indices] += (
            np.maximum(0.0, srad[day, indices])
            * f_solar
            * params.rue
            * f_co2
            * f_temp
            * stress
        )
        i50b_effective[indices] += params.i50max_h * (1.0 - f_heat)
        i50b_effective[indices] += params.i50max_w * (1.0 - f_water)
        paw[indices] = final_paw
        thermal_time[indices] += dtt
        days_run[indices] += 1
        active[indices] = (
            (thermal_time[indices] < params.tsum)
            & (days_run[indices] < params.max_cycle_days)
        )

    return VectorSimulationResult(
        days_run=days_run,
        mature=~active,
        thermal_time=thermal_time,
        biomass_g_m2=biomass,
        heat_stress_days=heat_stress_days,
        water_stress_days=water_stress_days,
    )
