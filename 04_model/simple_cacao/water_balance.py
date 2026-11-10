from __future__ import annotations

from dataclasses import dataclass

from .parameters import SoilParameters


@dataclass(frozen=True)
class WaterBalanceState:
    paw_mm: float


def arid_index(et0_mm: float, paw_mm: float) -> float:
    """Woli-style ARID index used by SIMPLE."""

    if et0_mm <= 0.0:
        return 0.0
    extractable = min(et0_mm, 0.096 * max(0.0, paw_mm))
    return min(1.0, max(0.0, 1.0 - extractable / et0_mm))


def curve_number_runoff(water_input_mm: float, curve_number: float) -> float:
    """SCS curve-number runoff for a daily water input."""

    if water_input_mm <= 0.0:
        return 0.0
    if not 0.0 < curve_number < 100.0:
        raise ValueError("curve_number must be between 0 and 100")
    storage = 25400.0 / curve_number - 254.0
    initial_abstraction = 0.2 * storage
    if water_input_mm <= initial_abstraction:
        return 0.0
    return (water_input_mm - initial_abstraction) ** 2 / (water_input_mm + 0.8 * storage)


def update_water_balance(
    state: WaterBalanceState,
    soil: SoilParameters,
    precip_mm: float,
    irrigation_mm: float,
    et0_mm: float,
) -> tuple[WaterBalanceState, dict[str, float]]:
    """Advance one daily one-bucket water balance.

    The routine preserves a closed daily mass balance over plant-available water,
    runoff, evapotranspiration withdrawal, and deep drainage.
    """

    water_input = max(0.0, precip_mm) + max(0.0, irrigation_mm)
    runoff = curve_number_runoff(water_input, soil.runoff_curve_number)
    infiltration = max(0.0, water_input - runoff)
    start_paw = min(max(0.0, state.paw_mm), soil.awc_mm)
    available_before_et = start_paw + infiltration
    et_withdrawal = min(max(0.0, et0_mm), 0.096 * available_before_et)
    after_et = available_before_et - et_withdrawal
    excess = max(0.0, after_et - soil.awc_mm)
    drainage = excess
    final_paw = after_et - drainage

    fluxes = {
        "input_mm": water_input,
        "runoff_mm": runoff,
        "infiltration_mm": infiltration,
        "et_withdrawal_mm": et_withdrawal,
        "drainage_mm": drainage,
        "arid": arid_index(et0_mm, final_paw),
    }
    return WaterBalanceState(paw_mm=final_paw), fluxes
