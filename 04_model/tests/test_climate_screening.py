from __future__ import annotations

# Consolidated from test_gcm_extremes.py
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

CLIMATE__gcm_extremes = Path(__file__).resolve().parents[2] / "02_preprocessing" / "climate"
sys.path.insert(0, str(CLIMATE__gcm_extremes))
spec__gcm_extremes = importlib.util.spec_from_file_location(
    "gcm_screening_extremes", CLIMATE__gcm_extremes / "gcm_screening.py"
)
module__gcm_extremes = importlib.util.module_from_spec(spec__gcm_extremes)
spec__gcm_extremes.loader.exec_module(module__gcm_extremes)


def test_wet_threshold_and_dry_spell_reset():
    result = module__gcm_extremes.annual_indices(
        np.array([[0.0, 2.0], [0.5, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 1.0]]), "pr"
    )
    np.testing.assert_allclose(result["wet_day_fraction"], [0.2, 0.4])
    np.testing.assert_allclose(result["longest_dry_spell_within_year"], [2.0, 3.0])
    np.testing.assert_allclose(result["annual_max"], [1.0, 2.0])


def test_heat_fraction_and_missing_data():
    result = module__gcm_extremes.annual_indices(np.array([[34.0], [35.0], [36.0]]), "tasmax")
    assert result["fraction_days_ge35C"][0] == pytest.approx(2 / 3)
    with pytest.raises(ValueError, match="finite"):
        module__gcm_extremes.annual_indices(np.array([[np.nan]]), "pr")
    with pytest.raises(ValueError, match="Negative"):
        module__gcm_extremes.annual_indices(np.array([[-1.0]]), "pr")


def test_cocoa_weighting_and_spatial_rmse():
    result = module__gcm_extremes.spatial_metrics(
        np.array([1.0, 3.0, np.nan]), np.array([3.0, 3.0, np.nan]), np.array([1.0, 3.0, 0.0])
    )
    assert result["reference"] == pytest.approx(2.5)
    assert result["bias"] == pytest.approx(0.5)
    assert result["spatial_rmse"] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        module__gcm_extremes.spatial_metrics(np.ones(3), np.ones(3), np.zeros(3))

# Consolidated from test_gcm_screening.py
CLIMATE__gcm_screening = Path(__file__).resolve().parents[2] / "02_preprocessing" / "climate"
sys.path.insert(0, str(CLIMATE__gcm_screening))
spec__gcm_screening = importlib.util.spec_from_file_location(
    "gcm_screening_historical", CLIMATE__gcm_screening / "gcm_screening.py"
)
module__gcm_screening = importlib.util.module_from_spec(spec__gcm_screening)
spec__gcm_screening.loader.exec_module(module__gcm_screening)


def test_calendar_months_and_constant_flux_are_preserved():
    for calendar, february, days in [
        ("365_day", 28, 365),
        ("standard", 29, 366),
        ("360_day", 30, 360),
    ]:
        lengths = module__gcm_screening.month_lengths(2000, calendar)
        assert lengths[1] == february
        assert sum(lengths) == days
        np.testing.assert_allclose(module__gcm_screening.monthly_means(np.full((days, 2), 3.0), lengths), 3.0)


def test_month_assignment_and_missing_data_rejection():
    lengths = module__gcm_screening.month_lengths(2001, "standard")
    values = np.repeat(np.arange(1.0, 13.0), lengths)
    np.testing.assert_allclose(module__gcm_screening.monthly_means(values, lengths), np.arange(1.0, 13.0))
    values[0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        module__gcm_screening.monthly_means(values, lengths)


def test_metrics_detect_known_bias_without_weather_pairing():
    reference = np.tile(np.arange(12.0), (15, 1))
    result = module__gcm_screening.metrics(reference, reference + 2.0)
    assert result["bias"] == pytest.approx(2.0)
    assert result["seasonal_cycle_rmse"] == pytest.approx(2.0)
    assert result["seasonal_cycle_correlation"] == pytest.approx(1.0)
    assert result["monthly_distribution_wasserstein"] == pytest.approx(2.0)


def test_static_ocean_mask_allowed_but_intermittent_gaps_rejected():
    lengths = module__gcm_screening.month_lengths(2001, "standard")
    values = np.ones((365, 2))
    values[:, 1] = np.nan
    result = module__gcm_screening.monthly_means(values, lengths, allow_static_mask=True)
    assert np.isnan(result[:, 1]).all()
    np.testing.assert_allclose(result[:, 0], 1.0)
    values[0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        module__gcm_screening.monthly_means(values, lengths, allow_static_mask=True)
