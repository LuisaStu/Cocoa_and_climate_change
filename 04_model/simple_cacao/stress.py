def temperature_factor(tmean_c: float, tbase_c: float, topt_c: float) -> float:
    """SIMPLE temperature impact on biomass growth."""

    if topt_c <= tbase_c:
        raise ValueError("topt_c must be greater than tbase_c")
    if tmean_c < tbase_c:
        return 0.0
    if tmean_c < topt_c:
        return (tmean_c - tbase_c) / (topt_c - tbase_c)
    return 1.0


def heat_factor(tmax_c: float, t_heat_c: float, t_extreme_c: float) -> float:
    """SIMPLE heat-stress factor."""

    if t_extreme_c <= t_heat_c:
        raise ValueError("t_extreme_c must be greater than t_heat_c")
    if tmax_c <= t_heat_c:
        return 1.0
    if tmax_c < t_extreme_c:
        return (t_extreme_c - tmax_c) / (t_extreme_c - t_heat_c)
    return 0.0


def co2_factor(co2_ppm: float, sco2_per_ppm: float) -> float:
    """SIMPLE CO2 factor using the original linear-to-700-ppm form.

    `sco2_per_ppm` is interpreted literally as the per-ppm coefficient in the
    published equation. Cacao parameter scaling is audited separately because
    the historical plan in README.md flags the published cacao value as needing equation-level review.
    """

    if co2_ppm <= 350.0:
        return 1.0
    capped_ppm = min(co2_ppm, 700.0)
    return 1.0 + sco2_per_ppm * (capped_ppm - 350.0)


def water_factor(arid: float, swater: float) -> float:
    """SIMPLE drought-stress factor, clipped to the physically useful range."""

    arid_clipped = min(1.0, max(0.0, arid))
    return max(0.0, 1.0 - swater * arid_clipped)
