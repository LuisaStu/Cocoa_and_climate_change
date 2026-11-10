from __future__ import annotations

from math import exp


def canopy_interception(
    thermal_time: float,
    tsum: float,
    i50a: float,
    i50b: float,
    fsolar_max: float,
    water_stress_factor: float = 1.0,
) -> float:
    """Fraction of solar radiation intercepted by the crop canopy.

    Uses the two logistic SIMPLE canopy phases: canopy closure before the midpoint
    of the cycle, and senescence afterward. Severe water stress can directly reduce
    interception following Eq. 14 from the SIMPLE paper.
    """

    if not 0.0 <= fsolar_max <= 1.0:
        raise ValueError("fsolar_max must be between 0 and 1")
    if i50a <= 0 or i50b <= 0 or tsum <= 0:
        raise ValueError("tsum, i50a, and i50b must be positive")

    if thermal_time <= tsum / 2.0:
        fsolar = fsolar_max / (1.0 + exp(-0.01 * (thermal_time - i50a)))
    else:
        fsolar = fsolar_max / (1.0 + exp(0.01 * (thermal_time - (tsum - i50b))))

    if water_stress_factor < 0.1:
        fsolar += water_stress_factor - 0.1

    return min(fsolar_max, max(0.0, fsolar))
