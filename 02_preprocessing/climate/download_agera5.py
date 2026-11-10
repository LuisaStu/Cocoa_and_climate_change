"""Download the historical AgERA5 forcing for the study region."""

from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZipFile

import cdsapi  # type: ignore[import-untyped]

API_URL = "https://cds.climate.copernicus.eu/api"
DATASET = "sis-agrometeorological-indicators"
AREA = [12, -9, 4, 4]
VARIABLES = {
    "tmax": ("2m_temperature", "24_hour_maximum"),
    "tmin": ("2m_temperature", "24_hour_minimum"),
    "tmean": ("2m_temperature", "24_hour_mean"),
    "solar_radiation": ("solar_radiation_flux", None),
    "precipitation": ("precipitation_flux", None),
    "reference_et": ("reference_evapotranspiration", None),
    "vpd": ("vapour_pressure_deficit_at_maximum_temperature", None),
}


def request_for(variable: str, year: int, area: list[float] | None = None) -> dict[str, object]:
    source_variable, statistic = VARIABLES[variable]
    request: dict[str, object] = {
        "version": "2_0",
        "format": "zip",
        "variable": source_variable,
        "year": [str(year)],
        "month": [f"{month:02d}" for month in range(1, 13)],
        "day": [f"{day:02d}" for day in range(1, 32)],
        "area": AREA if area is None else area,
    }
    if statistic is not None:
        request["statistic"] = [statistic]
    return request


def valid_zip(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    with ZipFile(path) as archive:
        return archive.testzip() is None


def download(
    years: range,
    variables: list[str],
    output_directory: Path,
    area: list[float] | None = None,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client(url=API_URL, progress=True)
    for year in years:
        for variable in variables:
            output = output_directory / f"agera5_{variable}_{year}.zip"
            if valid_zip(output):
                print(f"Skipping existing archive: {output}", flush=True)
                continue
            print(f"Requesting {variable} {year}", flush=True)
            result = client.retrieve(DATASET, request_for(variable, year, area))
            result.download(str(output))
            if not valid_zip(output):
                raise RuntimeError(f"CDS returned an invalid archive: {output}")
            print(f"Completed {output}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("01_inputs/raw/agera5_full_extent")
    )
    parser.add_argument("--variable", choices=sorted(VARIABLES), action="append")
    parser.add_argument(
        "--area",
        nargs=4,
        type=float,
        metavar=("NORTH", "WEST", "SOUTH", "EAST"),
        default=AREA,
        help="CDS bounding box [north, west, south, east] in degrees",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.end_year < args.start_year:
        raise ValueError("end year must not be earlier than start year")
    north, west, south, east = args.area
    if not (-90 <= south < north <= 90 and -180 <= west < east <= 180):
        raise ValueError("area must be valid [north, west, south, east] geographic bounds")
    variables = args.variable or sorted(VARIABLES)
    download(range(args.start_year, args.end_year + 1), variables, args.output, args.area)


if __name__ == "__main__":
    main()
