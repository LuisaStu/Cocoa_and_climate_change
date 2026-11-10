from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .canopy import canopy_interception
from .parameters import CacaoParameters, SoilParameters
from .phenology import daily_thermal_time
from .stress import co2_factor, heat_factor, temperature_factor, water_factor
from .units import grams_per_m2_to_kg_per_ha
from .water_balance import WaterBalanceState, update_water_balance


@dataclass(frozen=True)
class DailyWeather:
    tmax_c: float
    tmin_c: float
    srad_mj_m2: float
    precip_mm: float
    co2_ppm: float
    et0_mm: float
    irrigation_mm: float = 0.0

    @property
    def tmean_c(self) -> float:
        return (self.tmax_c + self.tmin_c) / 2.0


@dataclass(frozen=True)
class SimulationResult:
    days_run: int
    mature: bool
    thermal_time: float
    biomass_g_m2: float
    yield_kg_ha: float


class SimpleCacaoModel:
    """Scalar SIMPLE-Cacao cohort model with fresh state per instance."""

    def __init__(self, params: CacaoParameters, soil: SoilParameters) -> None:
        self.params = params
        self.soil = soil
        self._water = WaterBalanceState(paw_mm=soil.awc_mm)
        self._thermal_time = 0.0
        self._biomass = 0.0
        self._days = 0
        self._mature = False
        self._i50b_effective = params.i50b

    @property
    def thermal_time(self) -> float:
        return self._thermal_time

    @property
    def biomass_g_m2(self) -> float:
        return self._biomass

    @property
    def i50b_effective(self) -> float:
        return self._i50b_effective

    @property
    def mature(self) -> bool:
        return self._mature

    def step(self, weather: DailyWeather) -> None:
        if self._mature:
            return

        params = self.params
        self._water, water_fluxes = update_water_balance(
            self._water,
            self.soil,
            weather.precip_mm,
            weather.irrigation_mm,
            weather.et0_mm,
        )

        dtt = daily_thermal_time(weather.tmean_c, params.tbase)
        next_tt = self._thermal_time + dtt
        f_temp = temperature_factor(weather.tmean_c, params.tbase, params.topt)
        f_heat = heat_factor(weather.tmax_c, params.t_heat, params.t_extreme)
        f_water = water_factor(water_fluxes["arid"], params.swater)
        f_co2 = co2_factor(weather.co2_ppm, params.sco2)
        f_solar = canopy_interception(
            self._thermal_time,
            params.tsum,
            params.i50a,
            self._i50b_effective,
            params.fsolar_max,
            f_water,
        )

        stress = min(f_heat, f_water)
        self._biomass += (
            max(0.0, weather.srad_mj_m2) * f_solar * params.rue * f_co2 * f_temp * stress
        )

        self._i50b_effective += params.i50max_h * (1.0 - f_heat)
        self._i50b_effective += params.i50max_w * (1.0 - f_water)
        self._thermal_time = next_tt
        self._days += 1
        if self._thermal_time >= params.tsum or self._days >= params.max_cycle_days:
            self._mature = True

    def run(self, weather: Iterable[DailyWeather]) -> SimulationResult:
        for day in weather:
            self.step(day)
            if self._mature:
                break

        yield_g_m2 = self._biomass * self.params.hi
        return SimulationResult(
            days_run=self._days,
            mature=self._mature,
            thermal_time=self._thermal_time,
            biomass_g_m2=self._biomass,
            yield_kg_ha=grams_per_m2_to_kg_per_ha(yield_g_m2),
        )


def run_scalar(
    params: CacaoParameters,
    soil: SoilParameters,
    weather: Iterable[DailyWeather],
) -> SimulationResult:
    return SimpleCacaoModel(params, soil).run(weather)
