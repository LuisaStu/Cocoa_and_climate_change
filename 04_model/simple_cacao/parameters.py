from __future__ import annotations

from dataclasses import dataclass

CACAO_TABLE_CO2_RUE = 0.09


@dataclass(frozen=True)
class CacaoParameters:
    """Crop and management parameters for one scalar SIMPLE-Cacao cohort."""

    tsum: float
    hi: float
    i50a: float
    i50b: float
    tbase: float
    topt: float
    rue: float
    i50max_h: float
    i50max_w: float
    t_heat: float
    t_extreme: float
    sco2: float
    swater: float
    fsolar_max: float = 0.85
    initial_fsolar: float = 0.01
    max_cycle_days: int = 260


@dataclass(frozen=True)
class SoilParameters:
    """One-bucket soil-water parameters used by scalar SIMPLE tests."""

    awc_mm: float
    runoff_curve_number: float = 70.0
    deep_drainage_coefficient: float = 0.5
    rooting_depth_mm: float = 1500.0


def published_cacao_parameters(tsum: float = 2016.0, rue: float = 0.6) -> CacaoParameters:
    """Return published cacao anchors with a CO2-neutral, non-published response.

    The default `tsum` and `rue` values use the Santander calibration anchor from
    the historical plan in README.md. The paper's "CO2 RUE = 0.09" entry has no established conversion
    to SIMPLE's per-ppm S_CO2, so future-facing defaults use no CO2 multiplier.
    """

    return CacaoParameters(
        tsum=tsum,
        hi=0.30,
        i50a=680.0,
        i50b=680.0,
        tbase=10.0,
        topt=26.0,
        rue=rue,
        i50max_h=0.0,
        i50max_w=0.0,
        t_heat=35.0,
        t_extreme=40.0,
        sco2=0.0,
        swater=0.0,
        fsolar_max=0.85,
        initial_fsolar=0.01,
        max_cycle_days=260,
    )
