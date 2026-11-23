"""Measure the operational compute emissions of one project workflow."""

import argparse
import runpy
import sys
from pathlib import Path

from cli import COMMAND_ARGUMENTS, COMMANDS, ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--country", required=True, help="Compute location as an ISO alpha-3 code")
    parser.add_argument("--region", help="Optional grid region supported by CodeCarbon")
    parser.add_argument("--output", type=Path, required=True, help="New emissions CSV path")
    parser.add_argument("workflow", choices=sorted(COMMANDS.keys() - {"scenarios"}))
    parser.add_argument("workflow_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"emissions output already exists: {args.output}")

    try:
        from codecarbon import OfflineEmissionsTracker, OutputMethod
    except ImportError:
        parser.error("Install the optional carbon dependency with python -m pip install '.[carbon]'")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    script = ROOT / COMMANDS[args.workflow][0]
    tracker = OfflineEmissionsTracker(
        country_iso_code=args.country.upper(),
        region=args.region,
        project_name=f"SIMPLE-Cacao-WA {args.workflow}",
        tracking_mode="process",
        measure_power_secs=10,
        output_dir=str(args.output.parent.resolve()),
        output_file=args.output.name,
        output_methods=[OutputMethod.CSV],
    )
    sys.argv = [str(script), *COMMAND_ARGUMENTS.get(args.workflow, ()), *args.workflow_args]
    sys.path.insert(0, str(script.parent))
    tracker.start()
    try:
        runpy.run_path(str(script), run_name="__main__")
    finally:
        tracker.stop()
    if not args.output.is_file():
        raise RuntimeError(f"CodeCarbon did not write the emissions CSV: {args.output}")


if __name__ == "__main__":
    main()
