"""Single command-line entry point for project workflows."""

import argparse
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

COMMANDS: dict[str, tuple[str, str]] = {
    "map": (
        "04_model/run_late_century_suitability_map.py",
        "Generate the four-panel late-century map. Use --output-dir for a separate run.",
    ),
    "historical": (
        "04_model/run_historical_suitability.py",
        "Run the historical suitability reference.",
    ),
    "sensitivity": (
        "04_model/run_historical_suitability_sensitivity.py",
        "Run historical parameter sensitivity analysis.",
    ),
    "scenarios": (
        "workflow/acquire_scenarios.py",
        "Acquire and audit the future climate scenario archive.",
    ),
    "agera5-download": (
        "02_preprocessing/climate/download_agera5.py",
        "Download full-extent AgERA5 reference climate data.",
    ),
    "agera5-prepare": (
        "02_preprocessing/climate/prepare_agera5.py",
        "Prepare full-extent AgERA5 reference climate data.",
    ),
    "historical-climate-download": (
        "02_preprocessing/climate/download_nex_gddp_historical.py",
        "Download historical NEX-GDDP-CMIP6 climate data.",
    ),
    "scenario-download": (
        "02_preprocessing/climate/download_nex_scenarios.py",
        "Download future NEX-GDDP-CMIP6 scenario data.",
    ),
    "scenario-audit": (
        "02_preprocessing/climate/audit_nex_scenario_endpoints.py",
        "Audit NEX-GDDP-CMIP6 scenario endpoints.",
    ),
    "grid": (
        "02_preprocessing/climate/create_model_grid.py",
        "Create the full-extent model grid.",
    ),
    "rainfall-validation": (
        "02_preprocessing/climate/validate_chirps.py",
        "Validate rainfall forcing against CHIRPS.",
    ),
    "soil-download": (
        "02_preprocessing/soils/soil_inputs.py",
        "Download full-extent SoilGrids data.",
    ),
    "soil-prepare": (
        "02_preprocessing/soils/soil_inputs.py",
        "Prepare full-extent soil inputs.",
    ),
    "cocoa-map-download": (
        "02_preprocessing/cocoa_mask/cocoa_inputs.py",
        "Download the cocoa-area source map.",
    ),
    "cocoa-mask-prepare": (
        "02_preprocessing/cocoa_mask/cocoa_inputs.py",
        "Prepare the model cocoa-area mask.",
    ),
    "historical-gcm-screen": (
        "02_preprocessing/climate/gcm_screening.py",
        "Screen historical climate-model performance.",
    ),
    "scenario-extremes": (
        "02_preprocessing/climate/gcm_screening.py",
        "Screen future climate-model extremes.",
    ),
    "weather-check": (
        "02_preprocessing/climate/verify_weather_stream.py",
        "Verify the model weather stream.",
    ),
    "historical-summary": (
        "09_analysis/historical_suitability.py",
        "Summarize historical suitability results.",
    ),
    "gcm-weights": (
        "09_analysis/gcm_diagnostics.py",
        "Compare climate-model weighting approaches.",
    ),
    "gcm-screening-plot": (
        "09_analysis/gcm_diagnostics.py",
        "Plot climate-model screening results.",
    ),
}

COMMAND_ARGUMENTS: dict[str, tuple[str, ...]] = {
    "soil-download": ("download",),
    "soil-prepare": ("prepare",),
    "cocoa-map-download": ("download",),
    "cocoa-mask-prepare": ("prepare",),
    "historical-gcm-screen": ("historical",),
    "scenario-extremes": ("extremes",),
    "historical-summary": ("summary",),
    "gcm-weights": ("weights",),
    "gcm-screening-plot": ("plot",),
}


def _parser() -> argparse.ArgumentParser:
    command_help = "\n".join(
        f"  {name:<28} {description}" for name, (_, description) in COMMANDS.items()
    )
    return argparse.ArgumentParser(
        description="Run Cocoa-and-climate project workflows.",
        epilog=f"Commands:\n{command_help}\n\n"
        "Arguments after the command are passed unchanged to that workflow.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = _parser()
    if not arguments or arguments[0] in {"-h", "--help"}:
        parser.print_help()
        return 0

    command, *forwarded_arguments = arguments
    if command not in COMMANDS:
        parser.error(f"unknown command {command!r}")

    relative_script, description = COMMANDS[command]
    if forwarded_arguments and forwarded_arguments[0] in {"-h", "--help"}:
        print(f"usage: {Path(sys.argv[0]).name} {command} [workflow arguments]\n")
        print(description)
        print("\nWorkflow-specific options are passed to the underlying script.")
        return 0

    script = ROOT / relative_script
    if not script.is_file():
        raise FileNotFoundError(f"Workflow script does not exist: {script}")

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            *COMMAND_ARGUMENTS.get(command, ()),
            *forwarded_arguments,
        ],
        cwd=ROOT,
        check=False,
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
