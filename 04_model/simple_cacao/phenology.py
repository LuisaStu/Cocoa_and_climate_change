def daily_thermal_time(tmean_c: float, tbase_c: float) -> float:
    """Daily thermal time above the base temperature."""

    return max(0.0, tmean_c - tbase_c)
