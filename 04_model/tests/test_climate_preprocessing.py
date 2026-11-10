from __future__ import annotations

# Consolidated from test_agera5_downloader.py
import importlib.util
import sys
from pathlib import Path


def load_downloader():
    path = Path("02_preprocessing/climate/download_agera5.py")
    spec__agera5_downloader = importlib.util.spec_from_file_location("download_agera5", path)
    assert spec__agera5_downloader is not None
    assert spec__agera5_downloader.loader is not None
    module__agera5_downloader = importlib.util.module_from_spec(spec__agera5_downloader)
    spec__agera5_downloader.loader.exec_module(module__agera5_downloader)
    return module__agera5_downloader


def test_request_for_defaults_to_full_country_area():
    downloader = load_downloader()

    request = downloader.request_for("precipitation", 2000)

    assert request["area"] == [12, -9, 4, 4]


def test_download_defaults_to_full_extent_directory(monkeypatch):
    downloader = load_downloader()
    monkeypatch.setattr(
        sys,
        "argv",
        ["download_agera5.py", "--start-year", "2000", "--end-year", "2020"],
    )

    args = downloader.parse_args()

    assert args.output == Path("01_inputs/raw/agera5_full_extent")

# Consolidated from test_agera5_processing.py
from datetime import date

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin


def load_processor():
    path = Path("02_preprocessing/climate/prepare_agera5.py")
    spec__agera5_processing = importlib.util.spec_from_file_location("prepare_agera5", path)
    assert spec__agera5_processing is not None
    assert spec__agera5_processing.loader is not None
    module__agera5_processing = importlib.util.module_from_spec(spec__agera5_processing)
    sys.modules[spec__agera5_processing.name] = module__agera5_processing
    spec__agera5_processing.loader.exec_module(module__agera5_processing)
    return module__agera5_processing


def test_validate_member_dates_requires_complete_leap_year():
    processor = load_processor()
    members = [f"variable_{day:%Y%m%d}_final.nc" for day in processor.expected_dates(2000)]

    validated = processor.validate_member_dates(members, 2000)

    assert len(validated) == 366
    assert validated[0][0] == date(2000, 1, 1)
    assert validated[-1][0] == date(2000, 12, 31)


def test_validate_member_dates_rejects_missing_day_and_wrong_year():
    processor = load_processor()
    members = [
        f"variable_{day:%Y%m%d}_final.nc"
        for day in processor.expected_dates(2001)
        if day != date(2001, 3, 1)
    ]

    with pytest.raises(ValueError, match="incomplete"):
        processor.validate_member_dates(members, 2001)
    with pytest.raises(ValueError, match="Unexpected year"):
        processor.validate_member_dates(["variable_20000101_final.nc"], 2001)


@pytest.mark.parametrize(
    ("key", "values", "units", "expected"),
    [
        ("tmax", np.array([[300.0]], dtype=np.float32), "K", 26.85),
        ("solar_radiation", np.array([[15_000_000.0]], dtype=np.float32), "J m-2 d-1", 15.0),
        ("vpd", np.array([[25.0]], dtype=np.float32), "hPa", 2.5),
    ],
)
def test_standardize_values_converts_units(key, values, units, expected):
    processor = load_processor()
    spec__agera5_processing = next(variable for variable in processor.VARIABLES if variable.key == key)

    result = processor.standardize_values(values, spec__agera5_processing, units)

    np.testing.assert_allclose(result, [[expected]], rtol=1e-6)


def test_standardize_values_rejects_unknown_units_and_nonphysical_values():
    processor = load_processor()
    precipitation = next(
        variable for variable in processor.VARIABLES if variable.key == "precipitation"
    )
    with pytest.raises(ValueError, match="Unexpected units"):
        processor.standardize_values(np.array([[1.0]]), precipitation, "kg m-2")
    with pytest.raises(ValueError, match="below"):
        processor.standardize_values(np.array([[-1.0]]), precipitation, "mm d-1")


def test_source_transform_and_orientation_support_ascending_coordinates():
    processor = load_processor()
    latitudes = np.array([0.5, 1.5])
    longitudes = np.array([1.5, 0.5])
    values = np.array([[1, 2], [3, 4]])

    transform = processor.source_transform(latitudes, longitudes)
    oriented = processor.orient_source(values, latitudes, longitudes)

    assert transform == from_origin(0.0, 2.0, 1.0, 1.0)
    np.testing.assert_array_equal(oriented, [[4, 3], [2, 1]])


def test_reproject_daily_masks_non_country_cells(tmp_path):
    processor = load_processor()
    grid_path = tmp_path / "grid.tif"
    with rasterio.open(
        grid_path,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=1,
        dtype="uint8",
        crs="EPSG:4326",
        transform=from_origin(0.0, 2.0, 1.0, 1.0),
        nodata=255,
    ) as dataset:
        dataset.write(np.ones((2, 2), dtype=np.uint8), 1)

    with rasterio.open(grid_path) as target:
        result = processor.reproject_daily(
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            from_origin(0.0, 2.0, 1.0, 1.0),
            target,
            np.array([[True, False], [True, True]]),
        )

    np.testing.assert_allclose(result[[0, 1, 1], [0, 0, 1]], [1.0, 3.0, 4.0])
    assert np.isnan(result[0, 1])

# Consolidated from test_chirps_validation.py
from rasterio.windows import Window


def load_validation_module():
    path = Path("02_preprocessing/climate/validate_chirps.py")
    spec__chirps_validation = importlib.util.spec_from_file_location("validate_chirps", path)
    assert spec__chirps_validation is not None
    assert spec__chirps_validation.loader is not None
    module__chirps_validation = importlib.util.module_from_spec(spec__chirps_validation)
    spec__chirps_validation.loader.exec_module(module__chirps_validation)
    return module__chirps_validation


def test_area_weighted_aggregation_preserves_uniform_precipitation():
    validation = load_validation_module()
    source = np.full((4, 4), 12.5, dtype=np.float32)
    latitudes = np.array([0.15, 0.05])
    longitudes = np.array([0.05, 0.15])

    result, coverage = validation.area_weighted_chirps_to_grid(
        source,
        from_origin(0.0, 0.2, 0.05, 0.05),
        Window(0, 0, 4, 4),
        latitudes,
        longitudes,
    )

    np.testing.assert_allclose(result, 12.5)
    np.testing.assert_allclose(coverage, 1.0)


def test_area_weighted_aggregation_reports_missing_source_area():
    validation = load_validation_module()
    source = np.full((4, 4), 10.0, dtype=np.float32)
    source[:2, :2] = -9999.0
    latitudes = np.array([0.15, 0.05])
    longitudes = np.array([0.05, 0.15])

    result, coverage = validation.area_weighted_chirps_to_grid(
        source,
        from_origin(0.0, 0.2, 0.05, 0.05),
        Window(0, 0, 4, 4),
        latitudes,
        longitudes,
    )

    assert np.isnan(result[0, 0])
    assert coverage[0, 0] == 0.0
    np.testing.assert_allclose(result[1, 1], 10.0)
    np.testing.assert_allclose(coverage[1, 1], 1.0)


def test_chirps_validation_defaults_use_full_extent_paths():
    validation = load_validation_module()

    assert validation.DEFAULT_AGERA5_DIRECTORY == Path("01_inputs/raw/agera5_full_extent")
    assert validation.DEFAULT_OUTPUT_DIRECTORY == Path(
        "01_inputs/processed/validation_full_extent"
    )


def test_monthly_metrics_compare_paired_totals_without_filling_gaps():
    validation = load_validation_module()
    agera5 = np.full((36, 1, 1), np.nan, dtype=np.float32)
    chirps = np.full((36, 1, 1), np.nan, dtype=np.float32)
    agera5[[0, 12, 24], 0, 0] = [10.0, 20.0, np.nan]
    chirps[[0, 12, 24], 0, 0] = [12.0, 22.0, 40.0]
    coverage = np.ones_like(agera5)

    metrics = validation.calculate_cell_metrics(agera5, chirps, coverage, years=3)

    np.testing.assert_allclose(metrics["bias"][0, 0, 0], 2.0)
    np.testing.assert_allclose(metrics["percent_bias"][0, 0, 0], 13.3333333)
    assert metrics["valid_pairs"][0, 0, 0] == 2
    assert metrics["missing_agera5"][0, 0, 0] == 1

# Consolidated from test_model_grid.py
from rasterio.transform import array_bounds


def load_grid_module():
    path = Path("02_preprocessing/climate/create_model_grid.py")
    spec__model_grid = importlib.util.spec_from_file_location("create_model_grid", path)
    assert spec__model_grid is not None
    assert spec__model_grid.loader is not None
    module__model_grid = importlib.util.module_from_spec(spec__model_grid)
    spec__model_grid.loader.exec_module(module__model_grid)
    return module__model_grid


def test_snapped_grid_uses_fixed_outward_aligned_pixel_edges():
    grid = load_grid_module()

    transform, width, height = grid.snapped_grid((1_234.0, 2_345.0, 24_567.0, 35_678.0), 11_000)

    assert transform.c == 0.0
    assert transform.f == 44_000.0
    assert transform.a == 11_000.0
    assert transform.e == -11_000.0
    assert width == 3
    assert height == 4
    left, bottom, right, top = array_bounds(height, width, transform)
    assert left <= 1_234.0
    assert bottom <= 2_345.0
    assert right >= 24_567.0
    assert top >= 35_678.0


def test_snapped_grid_rejects_invalid_extent_and_resolution():
    grid = load_grid_module()

    try:
        grid.snapped_grid((0.0, 2.0, 1.0, 1.0), 11_000)
    except ValueError as error:
        assert "positive width and height" in str(error)
    else:
        raise AssertionError("Invalid extent should be rejected")

    try:
        grid.snapped_grid((0.0, 0.0, 1.0, 1.0), 0)
    except ValueError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("Invalid resolution should be rejected")


def test_grid_defaults_use_full_extent_paths():
    grid = load_grid_module()

    assert grid.DEFAULT_AGERA5_DIRECTORY == Path("01_inputs/raw/agera5_full_extent")
    assert grid.DEFAULT_OUTPUT_DIRECTORY == Path("01_inputs/processed/grid_full_extent")
    assert grid.DEFAULT_MANIFEST_PATH == Path(
        "01_inputs/manifests/model_grid_full_extent_manifest.yml"
    )

# Consolidated from test_nex_historical_acquisition.py
from io import BytesIO
from urllib.parse import parse_qs, urlparse

import xarray as xr

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "02_preprocessing"
    / "climate"
    / "download_nex_gddp_historical.py"
)
SPEC = importlib.util.spec_from_file_location("download_nex_gddp_historical", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Cannot load historical NEX downloader from {SCRIPT_PATH}")
NEX_DOWNLOADER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(NEX_DOWNLOADER)


def test_expected_days_preserves_source_calendar() -> None:
    assert NEX_DOWNLOADER.expected_days(2000, "proleptic_gregorian") == 366
    assert NEX_DOWNLOADER.expected_days(2000, "365_day") == 365
    assert NEX_DOWNLOADER.expected_days(2000, "360_day") == 360


def test_ncss_url_encodes_study_bounds_and_full_year() -> None:
    url = NEX_DOWNLOADER.ncss_url(
        "BCC-CSM2-MR",
        "historical",
        "r1i1p1f1",
        "pr",
        2000,
        (-9.0, 4.0, 4.0, 12.0),
    )
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    assert parsed.path.endswith(
        "/BCC-CSM2-MR/historical/r1i1p1f1/pr/pr_day_BCC-CSM2-MR_historical_r1i1p1f1_gn_2000_v2.0.nc"
    )
    assert query["west"] == ["-9.0"]
    assert query["east"] == ["4.0"]
    assert query["time_start"] == ["2000-01-01T12:00:00Z"]
    assert query["time_end"] == ["2000-12-31T12:00:00Z"]


def test_payload_validation_accepts_noleap_calendar_with_older_epoch() -> None:
    year = 2000
    bbox = (-9.0, 4.0, 4.0, 12.0)
    count = NEX_DOWNLOADER.expected_days(year, "365_day")
    first_time = (
        NEX_DOWNLOADER.calendar_ordinal(year, 1, 1, "365_day")
        - NEX_DOWNLOADER.calendar_ordinal(1850, 1, 1, "365_day")
        + 0.5
    )
    dataset = xr.Dataset(
        {
            "pr": (
                ("time", "lat", "lon"),
                np.ones((count, 2, 2), dtype=np.float32),
                {"units": "kg m-2 s-1"},
            )
        },
        coords={
            "time": (
                "time",
                first_time + np.arange(count, dtype=np.float64),
                {"units": "days since 1850-1-1", "calendar": "365_day"},
            ),
            "lat": ("lat", [4.125, 4.375]),
            "lon": ("lon", [351.0, 351.25]),
        },
    )
    payload = dataset.to_netcdf(BytesIO(), engine="scipy")
    assert payload is not None
    metadata = NEX_DOWNLOADER.validate_payload(payload, "pr", year, bbox)
    assert metadata["days"] == 365
    assert metadata["calendar"] == "365_day"
    assert metadata["time_units"] == "days since 1850-1-1"

# Consolidated from test_scenario_acquisition.py
import json

CLIMATE__scenario_acquisition = Path(__file__).resolve().parents[2] / "02_preprocessing" / "climate"
sys.path.insert(0, str(CLIMATE__scenario_acquisition))
spec__scenario_acquisition = importlib.util.spec_from_file_location(
    "download_nex_scenarios", CLIMATE__scenario_acquisition / "download_nex_scenarios.py"
)
module__scenario_acquisition = importlib.util.module_from_spec(spec__scenario_acquisition)
spec__scenario_acquisition.loader.exec_module(module__scenario_acquisition)


def test_endpoint_and_full_inventory():
    cfg = module__scenario_acquisition.load_yaml(CLIMATE__scenario_acquisition.parents[1] / "config" / "gcms.yml")
    proposal = module__scenario_acquisition.load_yaml(CLIMATE__scenario_acquisition.parents[1] / "config" / "gcm_ensemble_proposal.yml")
    jobs = module__scenario_acquisition.jobs_for(cfg, proposal, [2060, 2080, 2080])
    assert len(jobs) == 128
    assert len(set(jobs)) == 128
    assert {j[1] for j in jobs} == {"ssp245", "ssp585"}
    assert len(module__scenario_acquisition.jobs_for(cfg, proposal, list(range(2060, 2081)))) == 1344
    with pytest.raises(ValueError):
        module__scenario_acquisition.jobs_for(cfg, proposal, [2014])
    with pytest.raises(ValueError):
        module__scenario_acquisition.jobs_for(cfg, proposal, [2059])


def test_incremental_manifest_never_labels_partial_as_complete(tmp_path):
    path = tmp_path / "manifest.json"
    jobs = [("a", "ssp245", "r1i1p1f1", "pr", 2060)]
    module__scenario_acquisition.checkpoint(path, jobs, [], [], (-9, 4, 4, 12))
    assert json.loads(path.read_text())["status"] == "incomplete"
    module__scenario_acquisition.checkpoint(path, jobs, [{"path": "test.nc"}], [], (-9, 4, 4, 12))
    assert json.loads(path.read_text())["status"] == "complete_for_requested_batch"
    module__scenario_acquisition.checkpoint(path, jobs, [], [{"error": "network"}], (-9, 4, 4, 12))
    assert json.loads(path.read_text())["status"] == "incomplete"


def test_invalid_download_does_not_write_raw_file(tmp_path, monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b"server error HTML"

    monkeypatch.setattr(module__scenario_acquisition, "urlopen", lambda *args, **kwargs: Response())
    with pytest.raises(ValueError, match="NetCDF3"):
        module__scenario_acquisition.fetch_one(("ACCESS-CM2", "ssp245", "r1i1p1f1", "pr", 2060), (-9, 4, 4, 12), tmp_path)
    assert not list(tmp_path.rglob("*.nc"))


def test_resume_rejects_changed_file_before_parsing_or_network(tmp_path):
    job = ("ACCESS-CM2", "ssp245", "r1i1p1f1", "pr", 2060)
    path = (
        tmp_path / "ACCESS-CM2/ssp245/r1i1p1f1/pr/pr_day_ACCESS-CM2_ssp245_r1i1p1f1_gn_2060_v2.0.nc"
    )
    path.parent.mkdir(parents=True)
    path.write_bytes(b"changed raw data")
    with pytest.raises(ValueError, match="recorded checksum"):
        module__scenario_acquisition.fetch_one(job, (-9, 4, 4, 12), tmp_path, {path.as_posix(): "0" * 64})
    assert path.read_bytes() == b"changed raw data"


def test_checkpoint_retries_temporary_windows_file_lock(tmp_path, monkeypatch):
    original = Path.replace
    calls = []

    def locked_once(self, target):
        calls.append(target)
        if len(calls) == 1:
            raise PermissionError("Temporary sharing violation")
        return original(self, target)

    monkeypatch.setattr(Path, "replace", locked_once)
    monkeypatch.setattr(module__scenario_acquisition.time, "sleep", lambda seconds: None)
    path = tmp_path / "progress.json"
    module__scenario_acquisition.checkpoint(path, [("a", "ssp245", "member", "pr", 2060)], [], [], (-9, 4, 4, 12))
    assert len(calls) == 2
    assert json.loads(path.read_text())["status"] == "incomplete"

# Consolidated from test_weather_stream.py
import hashlib

CLIMATE__weather_stream = Path(__file__).resolve().parents[2] / "02_preprocessing/climate"
sys.path.insert(0, str(CLIMATE__weather_stream))
spec__weather_stream = importlib.util.spec_from_file_location("weather_stream", CLIMATE__weather_stream / "weather_stream.py")
module__weather_stream = importlib.util.module_from_spec(spec__weather_stream)
spec__weather_stream.loader.exec_module(module__weather_stream)


def test_chunk_boundaries_units_orientation_and_checksum(tmp_path, monkeypatch):
    entries = []
    coords = {
        "time": (
            "time",
            np.arange(365) + 0.5,
            {"units": "days since 2001-01-01", "calendar": "standard"},
        ),
        "lat": [4.125, 4.375],
        "lon": [-8.875, -8.625],
    }
    for variable, value in [("pr", 0.0001), ("tasmin", 290.0), ("tasmax", 300.0), ("rsds", 200.0)]:
        data = np.full((365, 2, 2), value, dtype="float32")
        if variable == "tasmax":
            data[:, 0, :] += 1
        path = tmp_path / f"{variable}.nc"
        dataset = xr.Dataset({variable: (("time", "lat", "lon"), data)}, coords=coords)
        if variable == "tasmin":
            dataset.time.attrs["bounds"] = "time_bnds"
        dataset.to_netcdf(path, engine="scipy")
        entries.append(
            {
                "model": "test",
                "year": 2001,
                "variable": variable,
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    monkeypatch.setattr(module__weather_stream, "load_yaml", lambda path: {"files": entries})
    grid = {
        "active": np.ones((2, 2), dtype=bool),
        "cell_ids": np.arange(4),
        "latitude": np.array([4.375, 4.375, 4.125, 4.125]),
        "shape": (2, 2),
        "transform": from_origin(-9, 4.5, 0.25, 0.25),
        "crs": "EPSG:4326",
    }
    monkeypatch.setattr(module__weather_stream, "grid_cells", lambda: grid)
    blocks = list(module__weather_stream.year_blocks("test", "historical", 2001, 31))
    assert [len(b["day_of_year"]) for b in blocks] == [31] * 11 + [24]
    np.testing.assert_array_equal(
        np.concatenate([b["day_of_year"] for b in blocks]), np.arange(1, 366)
    )
    for block in blocks:
        np.testing.assert_allclose(block["precip_mm"], 8.64, rtol=1e-6)
        np.testing.assert_allclose(block["srad_mj_m2"], 17.28, rtol=1e-6)
        np.testing.assert_allclose(block["tmax_c"][0], [26.85, 26.85, 27.85, 27.85], atol=1e-4)
        assert "et0_mm" not in block
    with pytest.raises(ValueError, match="Block size"):
        next(module__weather_stream.year_blocks("test", "historical", 2001, 32))
    Path(entries[0]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        next(module__weather_stream.year_blocks("test", "historical", 2001))
