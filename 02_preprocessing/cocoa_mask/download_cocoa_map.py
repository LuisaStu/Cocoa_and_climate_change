"""Download the Abu et al. cocoa map archive from PANGAEA."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen

DATASET_DOI = "10.1594/PANGAEA.917473"
DATASET_URL = "https://doi.pangaea.de/10.1594/PANGAEA.917473"
DOWNLOAD_URL = "https://download.pangaea.de/dataset/917473/files/Detected_Cocoa_Farms.rar"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
EXPECTED_ARCHIVE_SIZE_BYTES = 359_225_415
RAR_SIGNATURES = (b"Rar!\x1a\x07\x00", b"Rar!\x1a\x07\x01\x00")
CHUNK_SIZE = 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_rar_archive(path: Path) -> bool:
    if not path.is_file():
        return False
    with path.open("rb") as source:
        return source.read(8).startswith(RAR_SIGNATURES)


def download_archive(
    output_path: Path,
    *,
    url: str = DOWNLOAD_URL,
    expected_size_bytes: int | None = EXPECTED_ARCHIVE_SIZE_BYTES,
) -> str:
    if output_path.exists():
        if not is_rar_archive(output_path):
            raise RuntimeError(f"Existing file is not a RAR archive: {output_path}")
        if expected_size_bytes is not None and output_path.stat().st_size != expected_size_bytes:
            raise RuntimeError(
                f"Existing archive has size {output_path.stat().st_size}, "
                f"expected {expected_size_bytes}: {output_path}"
            )
        return sha256_file(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".part")
    digest = hashlib.sha256()
    total_bytes = 0
    try:
        with urlopen(url, timeout=60) as response, temporary_path.open("wb") as destination:
            content_length = response.headers.get("Content-Length")
            declared_size = int(content_length) if content_length is not None else None
            while chunk := response.read(CHUNK_SIZE):
                destination.write(chunk)
                digest.update(chunk)
                total_bytes += len(chunk)
                if total_bytes % (50 * CHUNK_SIZE) < CHUNK_SIZE:
                    print(f"Downloaded {total_bytes:,} bytes", flush=True)
        if expected_size_bytes is not None and total_bytes != expected_size_bytes:
            raise RuntimeError(
                f"Downloaded archive has {total_bytes} bytes; "
                f"the PANGAEA metadata lists {expected_size_bytes} bytes"
            )
        if declared_size is not None and total_bytes != declared_size:
            raise RuntimeError(f"Downloaded {total_bytes} bytes; server declared {declared_size}")
        if not is_rar_archive(temporary_path):
            raise RuntimeError(f"PANGAEA response is not a RAR archive: {url}")
        temporary_path.replace(output_path)
        return digest.hexdigest()
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("01_inputs/raw/cocoa_map_abu_2020/Detected_Cocoa_Farms.rar"),
    )
    parser.add_argument("--expected-size", type=int, default=EXPECTED_ARCHIVE_SIZE_BYTES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checksum = download_archive(args.output, expected_size_bytes=args.expected_size)
    print(f"Archive: {args.output}")
    print(f"Size: {args.output.stat().st_size:,} bytes")
    print(f"SHA-256: {checksum}")
    print(f"Dataset DOI: {DATASET_DOI}")
    print(f"License: {LICENSE_URL}")


if __name__ == "__main__":
    main()
