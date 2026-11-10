from __future__ import annotations

# Consolidated from test_soilgrids_downloader.py
import importlib.util
from pathlib import Path


def load_downloader():
    path = Path("02_preprocessing/soils/soil_inputs.py")
    spec = importlib.util.spec_from_file_location("download_soilgrids", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_soilgrids_wcs_url_uses_expected_coverage_and_bbox():
    downloader = load_downloader()
    url = downloader.wcs_url("wv0033", "0-5cm", "Q0.5", (-6.0, 4.0, 4.0, 12.0))

    assert "map=%2Fmap%2Fwv0033.map" in url
    assert "COVERAGEID=wv0033_0-5cm_Q0.5" in url
    assert "SUBSET=X%28-6.0%2C4.0%29" in url
    assert "SUBSET=Y%284.0%2C12.0%29" in url


def test_soilgrids_output_path_is_stable():
    downloader = load_downloader()
    path = downloader.output_path(Path("raw"), "wv1500", "5-15cm", "Q0.5")

    assert path == Path("raw/soilgrids_wv1500_5_15cm_Q0p5.tif")


def test_soilgrids_default_bbox_covers_expanded_country_extent():
    downloader = load_downloader()

    assert downloader.DEFAULT_BBOX == (-9.0, 4.0, 4.0, 12.0)
    assert downloader.DEFAULT_OUTPUT_DIRECTORY == Path("01_inputs/raw/soilgrids_full_extent")

# Consolidated from test_soilgrids_processing.py
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin


def load_processor():
    path = Path("02_preprocessing/soils/soil_inputs.py")
    spec = importlib.util.spec_from_file_location("prepare_soilgrids", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_layer_awc_converts_soilgrids_scale_and_coarse_fragments():
    processor = load_processor()
    wv33 = np.array([[300, 250]], dtype=np.int16)
    wv1500 = np.array([[100, 100]], dtype=np.int16)
    cfvo = np.array([[250, 0]], dtype=np.int16)

    capacity, valid = processor.layer_awc_mm(wv33, wv1500, cfvo, 100.0)

    np.testing.assert_allclose(capacity, [[19.5, 15.0]], rtol=1e-6)
    np.testing.assert_array_equal(valid, [[True, True]])


def test_layer_awc_marks_missing_and_physically_invalid_values():
    processor = load_processor()
    wv33 = np.array([[0, 90, 250]], dtype=np.int16)
    wv1500 = np.array([[0, 100, 100]], dtype=np.int16)
    cfvo = np.array([[0, 0, 10001]], dtype=np.int16)

    capacity, valid = processor.layer_awc_mm(wv33, wv1500, cfvo, 100.0)

    np.testing.assert_array_equal(valid, [[False, False, False]])
    assert np.isnan(capacity).all()


def test_profile_integrates_partial_final_depth_and_sensitivity():
    processor = load_processor()
    layer_awc = {
        "0_5cm": np.array([[10.0]]),
        "5_15cm": np.array([[20.0]]),
        "15_30cm": np.array([[30.0]]),
        "30_60cm": np.array([[60.0]]),
        "60_100cm": np.array([[80.0]]),
        "100_200cm": np.array([[200.0]]),
    }

    profiles = processor.integrate_profile(layer_awc)

    assert profiles[1000][0, 0] == pytest.approx(200.0)
    assert profiles[1500][0, 0] == pytest.approx(300.0)
    assert profiles[2000][0, 0] == pytest.approx(400.0)


def test_fill_gaps_is_limited_to_two_cells_and_country_mask():
    processor = load_processor()
    values = np.array([[10.0, np.nan, np.nan, np.nan, 50.0]], dtype=np.float32)
    country = np.ones(values.shape, dtype=bool)
    country[0, 4] = False

    filled, imputed = processor.fill_small_gaps(values, country, max_distance_cells=2)

    np.testing.assert_allclose(filled[0, :3], [10.0, 10.0, 10.0])
    assert np.isnan(filled[0, 3])
    assert not imputed[0, 3]
    np.testing.assert_array_equal(imputed, [[False, True, True, False, False]])


def test_prepare_soilgrids_writes_profiles_and_provenance(tmp_path):
    processor = load_processor()
    inputs = tmp_path / "raw"
    inputs.mkdir()
    transform = from_origin(0, 250, 250, 250)
    for depth in processor.DEPTHS_CM:
        for property_name, value in (("wv0033", 300), ("wv1500", 100), ("cfvo", 250)):
            path = processor.soil_layer_path(inputs, property_name, depth)
            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                width=1,
                height=1,
                count=1,
                dtype="int16",
                crs="EPSG:6933",
                transform=transform,
            ) as dataset:
                dataset.write(np.array([[value]], dtype=np.int16), 1)

    grid_path = tmp_path / "grid.tif"
    with rasterio.open(
        grid_path,
        "w",
        driver="GTiff",
        width=1,
        height=1,
        count=1,
        dtype="uint8",
        crs="EPSG:6933",
        transform=transform,
        nodata=255,
    ) as dataset:
        dataset.write(np.array([[1]], dtype=np.uint8), 1)

    output = tmp_path / "processed" / "awc.tif"
    flags = tmp_path / "processed" / "imputed.tif"
    summary = tmp_path / "processed" / "summary.yml"
    manifest = tmp_path / "manifests" / "soil.yml"
    result = processor.prepare_soilgrids(inputs, grid_path, output, flags, summary, manifest)

    with rasterio.open(output) as dataset:
        np.testing.assert_allclose(dataset.read()[:, 0, 0], [195.0, 292.5, 390.0], rtol=1e-6)
        assert dataset.crs.to_epsg() == 6933
    with rasterio.open(flags) as dataset:
        np.testing.assert_array_equal(dataset.read()[:, 0, 0], [0, 0, 0])
    assert (
        result["country_coverage_by_root_depth_mm"]["1000"]["GHA"][
            "coverage_percent_after_imputation"
        ]
        == 100.0
    )
    assert summary.is_file()
    assert manifest.is_file()
