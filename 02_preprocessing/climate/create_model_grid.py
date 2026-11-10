"""Create the fixed EPSG:6933 analysis grid and country-code template."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any, TypedDict

import h5py  # type: ignore[import-untyped]
import numpy as np
import rasterio  # type: ignore[import-untyped]
import yaml  # type: ignore[import-untyped]
from rasterio.features import rasterize  # type: ignore[import-untyped]
from rasterio.transform import Affine  # type: ignore[import-untyped]
from rasterio.warp import transform_bounds, transform_geom  # type: ignore[import-untyped]

SOURCE_CRS = "EPSG:4326"
TARGET_CRS = "EPSG:6933"
RESOLUTION_M = 11_000
DEFAULT_AGERA5_DIRECTORY = Path("01_inputs/raw/agera5_full_extent")
DEFAULT_OUTPUT_DIRECTORY = Path("01_inputs/processed/grid_full_extent")
DEFAULT_MANIFEST_PATH = Path("01_inputs/manifests/model_grid_full_extent_manifest.yml")


class BoundaryInfo(TypedDict):
    boundary_id: str
    boundary_sha256: str
    country_code: int


BOUNDARIES: dict[str, BoundaryInfo] = {
    "GHA": {
        "boundary_id": "GHA-ADM0-71095829",
        "boundary_sha256": "6913844e68f4a679b8854bd2ce6fb71c51647d1511abf5306a2426a61a7d240c",
        "country_code": 1,
    },
    "CIV": {
        "boundary_id": "CIV-ADM0-2848817",
        "boundary_sha256": "676d4839888cf2a12896d48f7c0af1a793486995dcd074a19bd1ef4d177f8afe",
        "country_code": 2,
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapped_grid(
    bounds: tuple[float, float, float, float], resolution: int = RESOLUTION_M
) -> tuple[Affine, int, int]:
    """Snap a projected bounding box outward to a zero-anchored square grid."""
    if resolution <= 0:
        raise ValueError("Grid resolution must be positive")
    left, bottom, right, top = bounds
    if not all(math.isfinite(value) for value in bounds):
        raise ValueError("Grid bounds must be finite")
    if right <= left or top <= bottom:
        raise ValueError("Grid bounds must have positive width and height")
    snapped_left = math.floor(left / resolution) * resolution
    snapped_top = math.ceil(top / resolution) * resolution
    width = math.ceil((right - snapped_left) / resolution)
    height = math.ceil((snapped_top - bottom) / resolution)
    return Affine(resolution, 0.0, snapped_left, 0.0, -resolution, snapped_top), width, height


def load_boundary_features(directory: Path) -> dict[str, list[dict[str, Any]]]:
    features_by_country: dict[str, list[dict[str, Any]]] = {}
    for iso3, metadata in BOUNDARIES.items():
        path = directory / f"geoboundaries_{iso3}_ADM0.geojson"
        if not path.is_file():
            raise FileNotFoundError(f"Missing pinned country boundary: {path}")
        checksum = sha256_file(path)
        if checksum != metadata["boundary_sha256"]:
            raise ValueError(f"Country boundary checksum mismatch: {path}")
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("type") != "FeatureCollection" or not document.get("features"):
            raise ValueError(f"Invalid country boundary GeoJSON: {path}")
        features_by_country[iso3] = document["features"]
    return features_by_country


def make_country_raster(
    features_by_country: dict[str, list[dict[str, Any]]],
    transform: Affine,
    width: int,
    height: int,
) -> np.ndarray:
    output = np.zeros((height, width), dtype=np.uint8)
    overlapping_centers = np.zeros((height, width), dtype=bool)
    for iso3, metadata in BOUNDARIES.items():
        shapes = [
            (
                transform_geom(SOURCE_CRS, TARGET_CRS, feature["geometry"]),
                metadata["country_code"],
            )
            for feature in features_by_country[iso3]
            if feature.get("geometry")
        ]
        country = np.asarray(
            rasterize(
                shapes,
                out_shape=output.shape,
                transform=transform,
                fill=0,
                all_touched=False,
                dtype=np.uint8,
            ),
            dtype=np.uint8,
        )
        selected = country != 0
        overlapping_centers |= selected & (output != 0)
        output[selected & (output == 0)] = int(metadata["country_code"])
    output[overlapping_centers] = 3
    for metadata in BOUNDARIES.values():
        if not np.any(output == metadata["country_code"]):
            raise ValueError(
                f"No target grid cells assigned to country code {metadata['country_code']}"
            )
    return output


def create_grid(
    project_root: Path,
    resolution: int = RESOLUTION_M,
    agera5_directory: Path | None = None,
    output_directory: Path | None = None,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    raw_agera5_directory = (
        agera5_directory if agera5_directory is not None else project_root / DEFAULT_AGERA5_DIRECTORY
    )
    agera5_archive = raw_agera5_directory / "agera5_precipitation_2000.zip"
    if not agera5_archive.is_file():
        raise FileNotFoundError(f"Missing AgERA5 archive: {agera5_archive}")
    with zipfile.ZipFile(agera5_archive) as archive:
        member = next(
            (name for name in archive.namelist() if name.lower().endswith(".nc")),
            None,
        )
        if member is None:
            raise ValueError(f"No daily NetCDF member in {agera5_archive}")
        with h5py.File(BytesIO(archive.read(member)), "r") as dataset:
            latitudes = np.asarray(dataset["lat"][:], dtype=np.float64)
            longitudes = np.asarray(dataset["lon"][:], dtype=np.float64)
    if latitudes.size < 2 or longitudes.size < 2:
        raise ValueError("AgERA5 grid has insufficient coordinates to infer cell edges")
    if not np.allclose(np.diff(latitudes), np.diff(latitudes)[0], atol=1e-8):
        raise ValueError("AgERA5 latitude spacing is not regular")
    if not np.allclose(np.diff(longitudes), np.diff(longitudes)[0], atol=1e-8):
        raise ValueError("AgERA5 longitude spacing is not regular")
    source_bounds = (
        float(longitudes.min() - abs(np.diff(longitudes)[0]) / 2),
        float(latitudes.min() - abs(np.diff(latitudes)[0]) / 2),
        float(longitudes.max() + abs(np.diff(longitudes)[0]) / 2),
        float(latitudes.max() + abs(np.diff(latitudes)[0]) / 2),
    )
    projected_bounds = transform_bounds(SOURCE_CRS, TARGET_CRS, *source_bounds, densify_pts=41)
    transform, width, height = snapped_grid(projected_bounds, resolution)
    boundaries_directory = project_root / "01_inputs" / "raw" / "boundaries"
    features = load_boundary_features(boundaries_directory)
    country_codes = make_country_raster(features, transform, width, height)

    if output_directory is None:
        output_directory = project_root / DEFAULT_OUTPUT_DIRECTORY
    output_directory.mkdir(parents=True, exist_ok=True)
    raster_path = output_directory / f"model_grid_EPSG6933_{resolution}m.tif"
    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=1,
        dtype="uint8",
        crs=TARGET_CRS,
        transform=transform,
        nodata=255,
        compress="lzw",
        tiled=True,
    ) as dataset:
        dataset.write(country_codes, 1)
        dataset.set_band_description(1, "country_code")
        dataset.update_tags(
            title="SIMPLE-Cacao-WA fixed analysis grid",
            source_bounds_lonlat=",".join(f"{value:.8f}" for value in source_bounds),
            resolution_m=str(resolution),
            country_code_mapping="0=outside Ghana/Cote d'Ivoire; 1=GHA; 2=CIV; 3=ambiguous shared boundary; 255=NoData",
            country_assignment="Cell center within a single national boundary; centers rasterized into both boundaries are code 3.",
            cocoa_mask="Not applied",
            source_climate="AgERA5 precipitation grid cell edges",
        )

    country_cells = {
        iso3: int(np.count_nonzero(country_codes == metadata["country_code"]))
        for iso3, metadata in BOUNDARIES.items()
    }
    manifest = {
        "grid": {
            "crs": TARGET_CRS,
            "resolution_m": resolution,
            "alignment": "Pixel boundaries snapped outward to multiples of resolution from projected origin (0, 0).",
            "source_bounds_lonlat": [round(value, 8) for value in source_bounds],
            "projected_bounds_m": [round(value, 3) for value in projected_bounds],
            "transform": [
                transform.a,
                transform.b,
                transform.c,
                transform.d,
                transform.e,
                transform.f,
            ],
            "width": width,
            "height": height,
            "pixel_area_m2": resolution**2,
            "nodata": 255,
            "country_code_mapping": {
                "0": "Outside Ghana and Cote d'Ivoire",
                "1": "Ghana",
                "2": "Cote d'Ivoire",
                "3": "Ambiguous overlapping shared-boundary center",
                "255": "NoData",
            },
            "cell_country_assignment": "Rasterized national polygons with pixel-center rule; centers rasterized into both boundaries are coded ambiguous (3).",
            "country_grid_cells": country_cells,
            "ambiguous_shared_boundary_cells": int(np.count_nonzero(country_codes == 3)),
            "country_boundary_ids": {
                iso3: {
                    "boundary_id": metadata["boundary_id"],
                    "sha256": metadata["boundary_sha256"],
                }
                for iso3, metadata in BOUNDARIES.items()
            },
        },
        "output": {
            "path": raster_path.relative_to(project_root).as_posix(),
            "sha256": sha256_file(raster_path),
            "size_bytes": raster_path.stat().st_size,
        },
        "scope_note": (
            "The grid covers only the acquired AgERA5 geographic bounds shown above. "
            "Country summaries are limited to the intersection of those bounds with "
            "each country. No climate/soil raster reprojection or cocoa masking was performed."
        ),
    }
    if manifest_path is None:
        manifest_path = project_root / DEFAULT_MANIFEST_PATH
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return manifest


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument("--resolution-m", type=int, default=RESOLUTION_M)
    parser.add_argument("--agera5-directory", type=Path)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--manifest-path", type=Path)
    args = parser.parse_args()
    result = create_grid(
        args.project_root.resolve(),
        args.resolution_m,
        args.agera5_directory.resolve() if args.agera5_directory else None,
        args.output_directory.resolve() if args.output_directory else None,
        args.manifest_path.resolve() if args.manifest_path else None,
    )
    print(yaml.safe_dump(result, sort_keys=False), flush=True)


if __name__ == "__main__":
    main()
