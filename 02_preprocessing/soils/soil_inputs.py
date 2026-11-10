from __future__ import annotations

import argparse
import hashlib
from collections.abc import Mapping
from contextlib import ExitStack
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlretrieve

import numpy as np
import rasterio
import yaml
from rasterio.enums import Resampling
from rasterio.warp import reproject
from scipy.ndimage import distance_transform_edt

# SoilGrids download
BASE_URL = "https://maps.isric.org/mapserv"
DOWNLOAD_DEPTHS_CM = ("0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm", "100-200cm")
PROPERTIES = ("wv0033", "wv1500", "bdod", "cfvo", "clay", "sand", "silt")
DEFAULT_PROPERTIES = ("wv0033", "wv1500")
DEFAULT_BBOX = (-9.0, 4.0, 4.0, 12.0)
DEFAULT_CRS = "http://www.opengis.net/def/crs/EPSG/0/4326"
DEFAULT_OUTPUT_DIRECTORY = Path("01_inputs/raw/soilgrids_full_extent")


def coverage_id(property_name: str, depth_cm: str, prediction: str) -> str:
    if property_name not in PROPERTIES:
        raise ValueError(f"Unsupported SoilGrids property: {property_name}")
    if depth_cm not in DOWNLOAD_DEPTHS_CM:
        raise ValueError(f"Unsupported SoilGrids depth: {depth_cm}")
    return f"{property_name}_{depth_cm}_{prediction}"


def wcs_url(
    property_name: str,
    depth_cm: str,
    prediction: str,
    bbox: tuple[float, float, float, float] = DEFAULT_BBOX,
    subsetting_crs: str = DEFAULT_CRS,
    output_crs: str = DEFAULT_CRS,
) -> str:
    west, south, east, north = bbox
    query = {
        "map": f"/map/{property_name}.map",
        "SERVICE": "WCS",
        "VERSION": "2.0.1",
        "REQUEST": "GetCoverage",
        "COVERAGEID": coverage_id(property_name, depth_cm, prediction),
        "FORMAT": "GEOTIFF_INT16",
        "SUBSET": [f"X({west},{east})", f"Y({south},{north})"],
        "SUBSETTINGCRS": subsetting_crs,
        "OUTPUTCRS": output_crs,
    }
    return f"{BASE_URL}?{urlencode(query, doseq=True)}"


def output_path(
    output_directory: Path,
    property_name: str,
    depth_cm: str,
    prediction: str,
) -> Path:
    safe_depth = depth_cm.replace("-", "_")
    safe_prediction = prediction.replace(".", "p")
    return output_directory / f"soilgrids_{property_name}_{safe_depth}_{safe_prediction}.tif"


def looks_like_geotiff(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    with path.open("rb") as handle:
        header = handle.read(4)
    return header[:2] in {b"II", b"MM"} and header != b"<?xm"


def download_layer(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output.with_suffix(output.suffix + ".tmp")
    urlretrieve(url, tmp_output)
    if not looks_like_geotiff(tmp_output):
        tmp_output.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded response is not a GeoTIFF: {output}")
    tmp_output.replace(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--property", choices=PROPERTIES, action="append")
    parser.add_argument("--depth", choices=DOWNLOAD_DEPTHS_CM, action="append")
    parser.add_argument("--prediction", default="Q0.5")
    parser.add_argument("--bbox", type=float, nargs=4, default=DEFAULT_BBOX)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def download_main() -> None:
    args = parse_args()
    properties = args.property or list(DEFAULT_PROPERTIES)
    depths = args.depth or list(DOWNLOAD_DEPTHS_CM)
    bbox = tuple(args.bbox)
    for property_name in properties:
        for depth_cm in depths:
            url = wcs_url(property_name, depth_cm, args.prediction, bbox)
            output = output_path(args.output, property_name, depth_cm, args.prediction)
            if output.exists() and looks_like_geotiff(output):
                print(f"Skipping existing GeoTIFF: {output}", flush=True)
                continue
            if args.dry_run:
                print(f"{output}\n  {url}", flush=True)
                continue
            print(f"Downloading {property_name} {depth_cm} -> {output}", flush=True)
            download_layer(url, output)

# SoilGrids preparation
DEPTHS_CM = ("0_5cm", "5_15cm", "15_30cm", "30_60cm", "60_100cm", "100_200cm")
DEPTH_BOUNDS_MM = {
    "0_5cm": (0, 50),
    "5_15cm": (50, 150),
    "15_30cm": (150, 300),
    "30_60cm": (300, 600),
    "60_100cm": (600, 1000),
    "100_200cm": (1000, 2000),
}
ROOT_DEPTHS_MM = (1000, 1500, 2000)
SCALE_TO_FRACTION = 0.001
CFVO_SCALE_TO_FRACTION = 0.0001
NODATA_FLOAT = -9999.0


def layer_awc_mm(
    wv0033_raw: np.ndarray,
    wv1500_raw: np.ndarray,
    cfvo_raw: np.ndarray,
    thickness_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert SoilGrids integer encodings into layer AWC in mm."""
    wv0033 = np.asarray(wv0033_raw, dtype=np.float32) * SCALE_TO_FRACTION
    wv1500 = np.asarray(wv1500_raw, dtype=np.float32) * SCALE_TO_FRACTION
    coarse_fraction = np.asarray(cfvo_raw, dtype=np.float32) * CFVO_SCALE_TO_FRACTION
    valid = (
        (wv0033_raw > 0)
        & (wv1500_raw > 0)
        & (cfvo_raw >= 0)
        & (cfvo_raw <= 10000)
        & (wv0033 >= wv1500)
    )
    capacity = (wv0033 - wv1500) * thickness_mm * (1.0 - coarse_fraction)
    return np.where(valid, capacity, np.nan).astype(np.float32), valid


def integrate_profile(layer_awc: Mapping[str, np.ndarray]) -> dict[int, np.ndarray]:
    """Integrate layer AWC to each root depth, prorating the 100-200 cm layer."""
    if set(layer_awc) != set(DEPTHS_CM):
        missing = sorted(set(DEPTHS_CM) - set(layer_awc))
        extra = sorted(set(layer_awc) - set(DEPTHS_CM))
        raise ValueError(f"Expected all six depth layers; missing={missing}, extra={extra}")
    shapes = {np.asarray(values).shape for values in layer_awc.values()}
    if len(shapes) != 1:
        raise ValueError("All soil layer arrays must have the same shape")

    profiles: dict[int, np.ndarray] = {}
    for root_depth in ROOT_DEPTHS_MM:
        total = np.zeros(next(iter(shapes)), dtype=np.float32)
        valid = np.ones(total.shape, dtype=bool)
        for depth in DEPTHS_CM:
            top_mm, bottom_mm = DEPTH_BOUNDS_MM[depth]
            overlap_mm = max(0, min(bottom_mm, root_depth) - top_mm)
            if overlap_mm == 0:
                continue
            values = np.asarray(layer_awc[depth], dtype=np.float32)
            valid &= np.isfinite(values)
            total += np.nan_to_num(values, nan=0.0) * (overlap_mm / (bottom_mm - top_mm))
        profiles[root_depth] = np.where(valid, total, np.nan)
    return profiles


def fill_small_gaps(
    values: np.ndarray,
    country_mask: np.ndarray,
    max_distance_cells: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Fill country cells from the nearest valid country cell within a distance limit."""
    values = np.asarray(values, dtype=np.float32)
    country_mask = np.asarray(country_mask, dtype=bool)
    if values.shape != country_mask.shape:
        raise ValueError("Values and country mask must have the same shape")
    if max_distance_cells < 0:
        raise ValueError("Maximum fill distance cannot be negative")

    valid = country_mask & np.isfinite(values)
    missing = country_mask & ~valid
    imputed = np.zeros(values.shape, dtype=bool)
    filled = values.copy()
    if not valid.any() or not missing.any():
        return filled, imputed

    distances, nearest_indices = distance_transform_edt(~valid, return_indices=True)
    imputed = missing & (distances <= max_distance_cells)
    filled[imputed] = values[tuple(nearest_indices[:, imputed])]
    return filled, imputed


def soil_layer_path(
    input_directory: Path, property_name: str, depth: str, prediction: str = "Q0p5"
) -> Path:
    return input_directory / f"soilgrids_{property_name}_{depth}_{prediction}.tif"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_source_profile(
    input_directory: Path,
) -> tuple[dict[int, np.ndarray], dict[str, object], dict[str, int]]:
    profiles: dict[int, np.ndarray] = {}
    masks: dict[int, np.ndarray] = {}
    source_metadata: dict[str, object] | None = None
    invalid_counts: dict[str, int] = {}

    for depth in DEPTHS_CM:
        paths = {
            prop: soil_layer_path(input_directory, prop, depth)
            for prop in ("wv0033", "wv1500", "cfvo")
        }
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Required SoilGrids layer(s) are missing: {missing}")

        with ExitStack() as stack:
            datasets = {
                prop: stack.enter_context(rasterio.open(path)) for prop, path in paths.items()
            }
            reference = datasets["wv0033"]
            current_metadata: dict[str, object] = {
                "width": reference.width,
                "height": reference.height,
                "transform": reference.transform,
                "crs": reference.crs,
                "bounds": tuple(reference.bounds),
            }
            if source_metadata is None:
                source_metadata = current_metadata
                profiles = {
                    root_depth: np.zeros((reference.height, reference.width), dtype=np.float32)
                    for root_depth in ROOT_DEPTHS_MM
                }
                masks = {
                    root_depth: np.ones((reference.height, reference.width), dtype=bool)
                    for root_depth in ROOT_DEPTHS_MM
                }
            elif current_metadata != source_metadata:
                raise ValueError(f"Source grid differs between SoilGrids depth {depth}")
            for prop, dataset in datasets.items():
                if (
                    dataset.width != reference.width
                    or dataset.height != reference.height
                    or dataset.transform != reference.transform
                    or dataset.crs != reference.crs
                ):
                    raise ValueError(f"Layer {paths[prop]} is not aligned to the reference grid")

            wv0033 = datasets["wv0033"].read(1)
            wv1500 = datasets["wv1500"].read(1)
            cfvo = datasets["cfvo"].read(1)
            top_mm, bottom_mm = DEPTH_BOUNDS_MM[depth]
            capacity, valid = layer_awc_mm(wv0033, wv1500, cfvo, bottom_mm - top_mm)
            invalid_counts[depth] = int(valid.size - np.count_nonzero(valid))
            for root_depth in ROOT_DEPTHS_MM:
                overlap_mm = max(0, min(bottom_mm, root_depth) - top_mm)
                if overlap_mm == 0:
                    continue
                masks[root_depth] &= valid
                profiles[root_depth] += np.nan_to_num(
                    capacity * (overlap_mm / (bottom_mm - top_mm)), nan=0.0
                )

    if source_metadata is None:
        raise ValueError("No SoilGrids layers were processed")
    for root_depth in ROOT_DEPTHS_MM:
        profiles[root_depth][~masks[root_depth]] = np.nan
    return profiles, source_metadata, invalid_counts


def _warp_to_model_grid(
    source: np.ndarray, source_metadata: Mapping[str, object], grid: rasterio.DatasetReader
) -> np.ndarray:
    destination = np.full((grid.height, grid.width), np.nan, dtype=np.float32)
    reproject(
        source=source,
        destination=destination,
        src_transform=source_metadata["transform"],
        src_crs=source_metadata["crs"],
        src_nodata=np.nan,
        dst_transform=grid.transform,
        dst_crs=grid.crs,
        dst_nodata=np.nan,
        resampling=Resampling.average,
        init_dest_nodata=True,
    )
    return destination


def _country_coverage(
    valid: np.ndarray, imputed: np.ndarray, country_codes: np.ndarray
) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for name, code in (("GHA", 1), ("CIV", 2)):
        selected = country_codes == code
        count = int(np.count_nonzero(selected))
        valid_count = int(np.count_nonzero(valid & selected))
        imputed_count = int(np.count_nonzero(imputed & selected))
        result[name] = {
            "grid_cells": count,
            "valid_before_imputation": valid_count,
            "imputed_cells": imputed_count,
            "valid_after_imputation": valid_count + imputed_count,
            "unfilled_cells": count - valid_count - imputed_count,
            "coverage_percent_after_imputation": (
                100.0 * (valid_count + imputed_count) / count if count else 0.0
            ),
        }
    return result


def prepare_soilgrids(
    input_directory: Path,
    grid_path: Path,
    output_path: Path,
    imputation_path: Path,
    summary_path: Path,
    manifest_path: Path,
    max_fill_distance_cells: float = 2.0,
) -> dict[str, object]:
    profiles, source_metadata, invalid_counts = _read_source_profile(input_directory)
    source_bounds = source_metadata["bounds"]
    if not isinstance(source_bounds, tuple):
        raise TypeError("Source raster bounds are unavailable")
    with rasterio.open(grid_path) as grid:
        country_codes = grid.read(1)
        country_mask = np.isin(country_codes, (1, 2))
        warped: dict[int, np.ndarray] = {}
        imputed_by_depth: dict[int, np.ndarray] = {}
        coverage: dict[str, dict[str, dict[str, float | int]]] = {}
        for root_depth, profile in profiles.items():
            values = _warp_to_model_grid(profile, source_metadata, grid)
            values[~country_mask] = np.nan
            valid_before = np.isfinite(values) & country_mask
            values, imputed = fill_small_gaps(
                values, country_mask, max_distance_cells=max_fill_distance_cells
            )
            values[~country_mask] = np.nan
            warped[root_depth] = values
            imputed_by_depth[root_depth] = imputed
            coverage[str(root_depth)] = _country_coverage(valid_before, imputed, country_codes)

        profile_out = grid.profile.copy()
        profile_out.update(
            driver="GTiff",
            count=3,
            dtype="float32",
            nodata=NODATA_FLOAT,
            compress="deflate",
            predictor=3,
            tiled=False,
        )
        profile_out.pop("blockxsize", None)
        profile_out.pop("blockysize", None)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **profile_out) as destination:
            for band, root_depth in enumerate(ROOT_DEPTHS_MM, start=1):
                values = np.where(
                    np.isfinite(warped[root_depth]), warped[root_depth], NODATA_FLOAT
                ).astype(np.float32)
                destination.write(values, band)
                destination.set_band_description(band, f"awc_root_depth_{root_depth}mm")
                destination.update_tags(band, units="mm")
            destination.update_tags(
                units="mm",
                root_depths_mm="1000,1500,2000",
                resampling="area-weighted average to equal-area target grid",
                cfvo_correction="AWC multiplied by (1 - cfvo/10000)",
            )

        flag_profile = profile_out.copy()
        flag_profile.update(count=3, dtype="uint8", nodata=255, predictor=2)
        imputation_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(imputation_path, "w", **flag_profile) as destination:
            for band, root_depth in enumerate(ROOT_DEPTHS_MM, start=1):
                flags = np.where(country_mask, imputed_by_depth[root_depth], 255).astype(np.uint8)
                destination.write(flags, band)
                destination.set_band_description(band, f"imputed_root_depth_{root_depth}mm")
            destination.update_tags(
                meaning="1=nearest valid soil value imputed; 0=not imputed",
                max_fill_distance_cells=str(max_fill_distance_cells),
            )
        grid_summary = {
            "crs": grid.crs.to_string(),
            "transform": list(grid.transform)[:6],
            "width": grid.width,
            "height": grid.height,
            "resolution_m": abs(grid.transform.a),
        }

    input_hashes = {
        str(path.relative_to(input_directory)): _sha256(path)
        for path in sorted(input_directory.glob("*.tif"))
        if path.name.startswith(("soilgrids_wv0033_", "soilgrids_wv1500_", "soilgrids_cfvo_"))
    }
    summary: dict[str, object] = {
        "dataset": "SoilGrids 2.0 Q0.5",
        "input_directory": str(input_directory),
        "source_grid": {
            "crs": str(source_metadata["crs"]),
            "width": source_metadata["width"],
            "height": source_metadata["height"],
            "bounds_lonlat": [float(value) for value in source_bounds],
        },
        "grid": grid_summary,
        "metadata_sources": {
            "soilgrids_documentation": "https://docs.isric.org/globaldata/soilgrids/",
            "cfvo_wms_getcapabilities": (
                "https://maps.isric.org/mapserv?map=/map/cfvo.map"
                "&SERVICE=WMS&VERSION=1.3.0&REQUEST=GetCapabilities"
            ),
            "cfvo_unit": "ISRIC WMS describes cfvo as volumetric content in permyriad.",
        },
        "method": {
            "water_content_conversion": "SoilGrids integer value * 0.001 = m3/m3",
            "coarse_fragment_conversion": "SoilGrids cfvo integer value * 0.0001 = volume fraction",
            "awc": "(wv0033 - wv1500) * layer thickness * (1 - cfvo fraction)",
            "root_depths_mm": list(ROOT_DEPTHS_MM),
            "resampling": "rasterio/GDAL average to EPSG:6933 equal-area grid",
            "gap_fill": "nearest valid country-cell value within two target-grid cells",
            "nodata": "Zero-valued wv layers and nonphysical layer values treated as missing",
        },
        "invalid_source_pixels_by_depth": invalid_counts,
        "country_coverage_by_root_depth_mm": coverage,
        "inputs_sha256": input_hashes,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")
    complete = all(
        country_metrics["coverage_percent_after_imputation"] == 100.0
        for depths in coverage.values()
        for country_metrics in depths.values()
    )
    manifest = {
        "status": "complete" if complete else "processed_with_unfilled_gaps",
        "inputs_sha256": input_hashes,
        "outputs": {
            str(output_path): _sha256(output_path),
            str(imputation_path): _sha256(imputation_path),
            str(summary_path): _sha256(summary_path),
        },
        "summary": summary,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return summary


def prepare_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-directory", type=Path, default=Path("01_inputs/raw/soilgrids_full_extent")
    )
    parser.add_argument(
        "--grid",
        type=Path,
        default=Path("01_inputs/processed/grid_full_extent/model_grid_EPSG6933_11000m.tif"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("01_inputs/processed/soil_full_extent/soil_awc_EPSG6933_11000m.tif"),
    )
    parser.add_argument(
        "--imputation-output",
        type=Path,
        default=Path("01_inputs/processed/soil_full_extent/soil_awc_imputed_EPSG6933_11000m.tif"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("01_inputs/processed/soil_full_extent/soil_awc_summary.yml"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("01_inputs/manifests/soilgrids_full_extent_manifest.yml"),
    )
    parser.add_argument("--max-fill-distance-cells", type=float, default=2.0)
    return parser.prepare_parse_args()


def prepare_main() -> None:
    args = prepare_parse_args()
    summary = prepare_soilgrids(
        args.input_directory,
        args.grid,
        args.output,
        args.imputation_output,
        args.summary,
        args.manifest,
        args.max_fill_distance_cells,
    )
    print(yaml.safe_dump(summary, sort_keys=False), end="")


def main(argv=None):
    import argparse
    import sys

    parser = argparse.ArgumentParser(description='Download or prepare SoilGrids inputs.')
    parser.add_argument("operation", choices=('download', 'prepare'))
    args, forwarded = parser.parse_known_args(argv)
    if argv is None:
        sys.argv = [sys.argv[0], *forwarded]
    if args.operation == 'download':
        download_main()
    elif args.operation == 'prepare':
        prepare_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
