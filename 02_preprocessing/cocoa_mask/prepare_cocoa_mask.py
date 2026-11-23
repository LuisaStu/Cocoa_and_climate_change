"""Aggregate Abu et al. binary cocoa maps to the full-extent model grid."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import rasterio  # type: ignore[import-untyped]
import yaml  # type: ignore[import-untyped]
from rasterio.enums import Resampling  # type: ignore[import-untyped]
from rasterio.warp import reproject  # type: ignore[import-untyped]

SOURCE_FILENAMES = {"GHA": "GHAKakao.tif", "CIV": "CIVKakao.tif"}
COUNTRY_CODES = {"GHA": 1, "CIV": 2}
SOURCE_CLASSES = {0, 1}
SOURCE_NODATA = -32768
OUTPUT_NODATA = -9999.0
DATASET_DOI = "10.1594/PANGAEA.917473"
DATASET_URL = "https://doi.pangaea.de/10.1594/PANGAEA.917473"
PAPER_DOI = "10.1016/j.ecolind.2021.107863"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_source_classes(dataset: rasterio.DatasetReader, sample_size: int = 2048) -> set[int]:
    """Validate the documented binary classes on a regular sample of the raster."""
    height = min(dataset.height, sample_size)
    width = min(dataset.width, sample_size)
    sample = dataset.read(
        1,
        out_shape=(height, width),
        resampling=Resampling.nearest,
        masked=True,
    )
    classes = {int(value) for value in np.unique(sample.compressed())}
    unexpected = classes - SOURCE_CLASSES
    if unexpected:
        raise ValueError(f"Unexpected cocoa-map class values: {sorted(unexpected)}")
    if not classes:
        raise ValueError("Cocoa source raster contains no valid class values")
    return classes


def aggregate_source_to_grid(
    dataset: rasterio.DatasetReader,
    grid: rasterio.DatasetReader,
) -> np.ndarray:
    """Calculate the mapped cocoa fraction by average resampling to target cells."""
    validate_source_classes(dataset)
    fraction = np.full((grid.height, grid.width), np.nan, dtype=np.float32)
    reproject(
        source=rasterio.band(dataset, 1),
        destination=fraction,
        src_transform=dataset.transform,
        src_crs=dataset.crs,
        src_nodata=dataset.nodata if dataset.nodata is not None else SOURCE_NODATA,
        dst_transform=grid.transform,
        dst_crs=grid.crs,
        dst_nodata=np.nan,
        resampling=Resampling.average,
        init_dest_nodata=True,
    )
    return fraction


def _country_coverage(
    fraction: np.ndarray, country_codes: np.ndarray, pixel_area_m2: float
) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for iso3, code in COUNTRY_CODES.items():
        country = country_codes == code
        valid = country & np.isfinite(fraction)
        fraction_values = fraction[valid]
        result[iso3] = {
            "grid_cells": int(np.count_nonzero(country)),
            "valid_grid_cells": int(np.count_nonzero(valid)),
            "uncovered_grid_cells": int(np.count_nonzero(country & ~np.isfinite(fraction))),
            "coverage_percent": float(
                100.0 * np.count_nonzero(valid) / np.count_nonzero(country)
                if np.any(country)
                else 0.0
            ),
            "cocoa_positive_cells": int(np.count_nonzero(fraction_values > 0)),
            "estimated_cocoa_area_ha": float(
                np.sum(fraction_values, dtype=np.float64) * pixel_area_m2 / 10_000.0
            ),
            "mean_cell_fraction": (
                float(np.mean(fraction_values)) if fraction_values.size else 0.0
            ),
        }
    return result


def prepare_cocoa_mask(
    input_directory: Path,
    grid_path: Path,
    output_path: Path,
    manifest_path: Path,
    archive_path: Path | None = None,
) -> dict[str, Any]:
    if archive_path is None:
        archive_path = input_directory.parent / "Detected_Cocoa_Farms.rar"
    source_paths = {iso3: input_directory / filename for iso3, filename in SOURCE_FILENAMES.items()}
    missing = [str(path) for path in source_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required country cocoa raster(s) are missing: {missing}")

    with rasterio.open(grid_path) as grid:
        country_codes = grid.read(1)
        fraction = np.full((grid.height, grid.width), np.nan, dtype=np.float32)
        source_metadata: dict[str, dict[str, Any]] = {}
        for iso3, source_path in source_paths.items():
            with rasterio.open(source_path) as source:
                if source.count != 1:
                    raise ValueError(f"Expected a single-band cocoa raster: {source_path}")
                if source.crs is None or grid.crs is None:
                    raise ValueError(f"Missing CRS for source or target raster: {source_path}")
                values = aggregate_source_to_grid(source, grid)
                country_mask = country_codes == COUNTRY_CODES[iso3]
                fraction[country_mask] = values[country_mask]
                source_metadata[iso3] = {
                    "path": str(source_path),
                    "sha256": sha256_file(source_path),
                    "crs": source.crs.to_string(),
                    "width": source.width,
                    "height": source.height,
                    "bounds": [float(value) for value in source.bounds],
                    "resolution_degrees": [float(value) for value in source.res],
                    "dtype": source.dtypes[0],
                    "nodata": source.nodata,
                    "sampled_class_values": sorted(validate_source_classes(source)),
                }

        country_mask = np.isin(country_codes, tuple(COUNTRY_CODES.values()))
        for iso3, code in COUNTRY_CODES.items():
            country = country_codes == code
            uncovered = country & ~np.isfinite(fraction)
            if np.any(country) and np.all(uncovered):
                raise ValueError(f"Cocoa source data do not cover any assigned {iso3} grid cells")
        if np.any((fraction[country_mask] < 0) | (fraction[country_mask] > 1)):
            raise ValueError("Aggregated cocoa fractions fall outside [0, 1]")

        profile = grid.profile.copy()
        profile.update(
            driver="GTiff",
            count=1,
            dtype="float32",
            nodata=OUTPUT_NODATA,
            compress="deflate",
            predictor=3,
            tiled=False,
        )
        profile.pop("blockxsize", None)
        profile.pop("blockysize", None)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_values = np.where(np.isfinite(fraction), fraction, OUTPUT_NODATA).astype(np.float32)
        with rasterio.open(output_path, "w", **profile) as destination:
            destination.write(output_values, 1)
            destination.set_band_description(1, "cocoa_fraction")
            destination.update_tags(
                units="fraction of 11 km analysis cell",
                source_classes="0=not classified as cocoa; 1=classified as cocoa",
                aggregation="average of source binary class to equal-area target cell",
                source_dataset_doi=DATASET_DOI,
                source_paper_doi=PAPER_DOI,
                license="CC-BY-4.0",
            )
        grid_metadata = {
            "crs": grid.crs.to_string(),
            "width": grid.width,
            "height": grid.height,
            "transform": list(grid.transform)[:6],
            "resolution_m": abs(grid.transform.a),
            "pixel_area_m2": abs(grid.transform.a * grid.transform.e),
        }

    coverage = _country_coverage(fraction, country_codes, grid_metadata["pixel_area_m2"])
    summary: dict[str, Any] = {
        "dataset": "Cocoa Map for Cote d'Ivoire and Ghana",
        "dataset_doi": DATASET_DOI,
        "dataset_url": DATASET_URL,
        "license": "CC-BY-4.0",
        "license_url": LICENSE_URL,
        "authors": [
            "Itohan-Osa Abu",
            "Zoltan Szantoi",
            "Andreas Brink",
            "M Robuchon",
            "Michael Thiel",
        ],
        "paper": {
            "citation": (
                "Abu, I-O., Szantoi, Z., Brink, A., Robuchon, M., and Thiel, M. "
                "(2021). Detecting cocoa plantations in Cote d'Ivoire and Ghana "
                "and their implications on protected areas. Ecological Indicators, "
                "129, 107863."
            ),
            "doi": PAPER_DOI,
            "reported_source_resolution_m": 10,
            "reported_producer_accuracy_percent": 82.89,
            "reported_user_accuracy_percent": 62.22,
            "source_time_note": (
                "The paper describes Sentinel imagery for 2019; this is a recent "
                "footprint, not a historical cocoa-area time series."
            ),
        },
        "source_rasters": source_metadata,
        "target_grid": grid_metadata,
        "method": {
            "source_encoding": "binary map: 1=cocoa, 0=not cocoa, -32768=nodata",
            "aggregation": (
                "Average resampling of the binary source class to the EPSG:6933 "
                "equal-area grid; output is fractional mapped cocoa area per cell."
            ),
            "country_assignment": (
                "Use Ghana or Cote d'Ivoire source raster according to the "
                "existing pixel-center country-code raster."
            ),
            "no_imputation": True,
        },
        "country_coverage": coverage,
    }
    has_coverage_gaps = any(values["uncovered_grid_cells"] > 0 for values in coverage.values())
    manifest = {
        "status": ("complete_with_source_coverage_gaps" if has_coverage_gaps else "complete"),
        "archive": {
            "path": str(archive_path),
            "size_bytes": archive_path.stat().st_size,
            "sha256": sha256_file(archive_path),
        },
        "inputs": {
            iso3: {
                "path": str(path),
                "sha256": source_metadata[iso3]["sha256"],
            }
            for iso3, path in source_paths.items()
        },
        "outputs": {str(output_path): sha256_file(output_path)},
        "summary": summary,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-directory",
        type=Path,
        default=Path("01_inputs/raw/cocoa_map_abu_2020/Detected Cocoa Farms"),
    )
    parser.add_argument(
        "--grid",
        type=Path,
        default=Path("01_inputs/processed/grid_full_extent/model_grid_EPSG6933_11000m.tif"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "01_inputs/processed/cocoa_mask_full_extent/cocoa_fraction_EPSG6933_11000m.tif"
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("01_inputs/manifests/cocoa_mask_abu_manifest.yml"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = prepare_cocoa_mask(args.input_directory, args.grid, args.output, args.manifest)
    print(yaml.safe_dump(summary, sort_keys=False), end="")


if __name__ == "__main__":
    main()
