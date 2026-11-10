"""Download bounded SoilGrids WCS layers for the study region."""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlretrieve

BASE_URL = "https://maps.isric.org/mapserv"
DEPTHS_CM = ("0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm", "100-200cm")
PROPERTIES = ("wv0033", "wv1500", "bdod", "cfvo", "clay", "sand", "silt")
DEFAULT_PROPERTIES = ("wv0033", "wv1500")
DEFAULT_BBOX = (-9.0, 4.0, 4.0, 12.0)
DEFAULT_CRS = "http://www.opengis.net/def/crs/EPSG/0/4326"
DEFAULT_OUTPUT_DIRECTORY = Path("01_inputs/raw/soilgrids_full_extent")


def coverage_id(property_name: str, depth_cm: str, prediction: str) -> str:
    if property_name not in PROPERTIES:
        raise ValueError(f"Unsupported SoilGrids property: {property_name}")
    if depth_cm not in DEPTHS_CM:
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
    parser.add_argument("--depth", choices=DEPTHS_CM, action="append")
    parser.add_argument("--prediction", default="Q0.5")
    parser.add_argument("--bbox", type=float, nargs=4, default=DEFAULT_BBOX)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    properties = args.property or list(DEFAULT_PROPERTIES)
    depths = args.depth or list(DEPTHS_CM)
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


if __name__ == "__main__":
    main()
