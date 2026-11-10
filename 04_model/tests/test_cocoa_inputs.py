from __future__ import annotations

# Consolidated from test_cocoa_map_downloader.py
import importlib.util
from io import BytesIO
from pathlib import Path

import pytest


def load_downloader():
    path = Path("02_preprocessing/cocoa_mask/cocoa_inputs.py")
    spec = importlib.util.spec_from_file_location("download_cocoa_map", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_archive_signature_accepts_rar4_and_rar5(tmp_path):
    downloader = load_downloader()
    rar4 = tmp_path / "rar4.rar"
    rar5 = tmp_path / "rar5.rar"
    rar4.write_bytes(b"Rar!\x1a\x07\x00" + b"payload")
    rar5.write_bytes(b"Rar!\x1a\x07\x01\x00" + b"payload")

    assert downloader.is_rar_archive(rar4)
    assert downloader.is_rar_archive(rar5)


def test_download_writes_valid_archive_and_hash(tmp_path, monkeypatch):
    downloader = load_downloader()
    content = b"Rar!\x1a\x07\x00" + b"small test archive"

    class Response(BytesIO):
        def __init__(self, payload: bytes):
            super().__init__(payload)
            self.headers = {"Content-Length": str(len(payload))}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    monkeypatch.setattr(downloader, "urlopen", lambda _url, timeout: Response(content))
    output = tmp_path / "download" / "dataset.rar"

    checksum = downloader.download_archive(
        output, url="https://example.invalid/dataset.rar", expected_size_bytes=len(content)
    )

    assert output.read_bytes() == content
    assert checksum == downloader.sha256_file(output)
    assert not output.with_suffix(".rar.part").exists()


def test_download_rejects_non_rar_response_and_cleans_partial_file(tmp_path, monkeypatch):
    downloader = load_downloader()

    class Response(BytesIO):
        def __init__(self, payload: bytes):
            super().__init__(payload)
            self.headers = {"Content-Length": str(len(payload))}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    monkeypatch.setattr(downloader, "urlopen", lambda _url, timeout: Response(b"not RAR"))
    output = tmp_path / "dataset.rar"

    with pytest.raises(RuntimeError, match="not a RAR archive"):
        downloader.download_archive(output, expected_size_bytes=7)

    assert not output.exists()
    assert not output.with_suffix(".rar.part").exists()

# Consolidated from test_cocoa_mask.py
import numpy as np
import rasterio
import yaml
from rasterio.transform import from_origin


def load_processor():
    path = Path("02_preprocessing/cocoa_mask/cocoa_inputs.py")
    spec = importlib.util.spec_from_file_location("prepare_cocoa_mask", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_raster(path, data, *, transform, crs="EPSG:4326", nodata=-32768):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=data.shape[1],
        height=data.shape[0],
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dataset:
        dataset.write(data, 1)


def test_aggregate_source_to_grid_averages_binary_classes(tmp_path):
    processor = load_processor()
    source_path = tmp_path / "source.tif"
    target_path = tmp_path / "target.tif"
    source_data = np.array([[1, 1], [0, 0]], dtype=np.int16)
    transform = from_origin(0, 2, 1, 1)
    write_raster(source_path, source_data, transform=transform)
    write_raster(
        target_path,
        np.array([[1]], dtype=np.uint8),
        transform=from_origin(0, 2, 2, 2),
        nodata=255,
    )
    with rasterio.open(source_path) as source, rasterio.open(target_path) as target:
        fraction = processor.aggregate_source_to_grid(source, target)
    np.testing.assert_allclose(fraction, [[0.5]], rtol=1e-6)
    source_path.unlink(missing_ok=True)
    target_path.unlink(missing_ok=True)


def test_prepare_cocoa_mask_aggregates_by_country_and_writes_manifest(tmp_path):
    processor = load_processor()
    sources = tmp_path / "sources"
    sources.mkdir()
    target_transform = from_origin(0, 2, 2, 2)
    source_transform = from_origin(0, 2, 1, 1)
    write_raster(
        sources / "GHAKakao.tif",
        np.array([[1, 1, 0, 0], [1, 1, 0, 0]], dtype=np.int16),
        transform=source_transform,
    )
    write_raster(
        sources / "CIVKakao.tif",
        np.array([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=np.int16),
        transform=source_transform,
    )
    grid = tmp_path / "grid.tif"
    write_raster(
        grid,
        np.array([[1, 2, 2]], dtype=np.uint8),
        transform=target_transform,
        nodata=255,
    )
    output = tmp_path / "processed" / "cocoa.tif"
    manifest_path = tmp_path / "manifests" / "cocoa.yml"
    archive_path = tmp_path / "Detected_Cocoa_Farms.rar"
    archive_path.write_bytes(b"archive for manifest test")

    summary = processor.prepare_cocoa_mask(sources, grid, output, manifest_path, archive_path)

    with rasterio.open(output) as dataset:
        np.testing.assert_allclose(
            dataset.read(1), [[1.0, 1.0, processor.OUTPUT_NODATA]], rtol=1e-6
        )
        assert dataset.crs.to_epsg() == 4326
        assert dataset.descriptions == ("cocoa_fraction",)
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "complete_with_source_coverage_gaps"
    assert manifest["summary"]["license"] == "CC-BY-4.0"
    assert summary["country_coverage"]["GHA"]["coverage_percent"] == 100.0
    assert summary["country_coverage"]["CIV"]["coverage_percent"] == 50.0
    assert summary["country_coverage"]["CIV"]["uncovered_grid_cells"] == 1


def test_source_class_validation_rejects_unexpected_values(tmp_path):
    processor = load_processor()
    source_path = tmp_path / "bad.tif"
    write_raster(
        source_path,
        np.array([[0, 2], [0, 1]], dtype=np.int16),
        transform=from_origin(0, 2, 1, 1),
    )

    with (
        rasterio.open(source_path) as source,
        pytest.raises(ValueError, match="Unexpected cocoa-map class"),
    ):
        processor.validate_source_classes(source)
