"""Plot the archived historical hot-day bias and cocoa footprint."""

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as effects
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import Normalize, TwoSlopeNorm, to_rgb
from matplotlib.patches import Patch
from rasterio.warp import transform

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "01_inputs/processed"
DIAGNOSTICS = ROOT / "06_validation/historical_gcm/daily_spatial"
OUTPUT = ROOT / "10_figures/poster"
METRIC = "fraction_days_ge35C"
COUNTRIES = {1: ("GHA", "Ghana", -1.0, 9.7, 1168), 2: ("CIV", "Côte d’Ivoire", -5.6, 9.1, 1373)}
NEUTRAL = "#e5e7e9"
MISSING = "#766d7d"


def read_band(path, description, grid=None, nodata=-9999):
    with rasterio.open(path) as source:
        geometry = (source.shape, source.crs, source.transform)
        if source.nodata != nodata or source.descriptions.count(description) != 1:
            raise ValueError(f"Unexpected NoData or band descriptions in {path.name}")
        if grid is not None and geometry != grid:
            raise ValueError(f"Shape, CRS or affine mismatch in {path.name}")
        band = source.read(source.descriptions.index(description) + 1, masked=True)
        values = band.astype("float64").filled(np.nan)
        if not np.isfinite(values[~np.ma.getmaskarray(band)]).all():
            raise ValueError(f"Unmasked nonfinite values in {path.name}")
        if grid is not None and np.any((values < 0) | (values > 1)):
            raise ValueError(f"Fraction outside [0, 1] in {path.name}")
        return values, geometry


def frequency(values, selected, cocoa):
    valid = selected & np.isfinite(values)
    return np.average(values[valid], weights=cocoa[valid])


def cross_check(codes, cocoa, footprint, common, reference, models):
    with (DIAGNOSTICS / "spatial_diagnostics.csv").open(newline="", encoding="utf-8") as stream:
        rows = {
            (r["country"], r["model"]): r
            for r in csv.DictReader(stream)
            if (r["domain"], r["variable"], r["metric"]) == ("cocoa", "tasmax", METRIC)
        }
    for code, (iso, _, _, _, expected) in COUNTRIES.items():
        selected = footprint & (codes == code)
        count = np.count_nonzero(selected)
        common_count = np.count_nonzero(common & (codes == code))
        print(f"{iso}: {count:,} positive cocoa cells, {common_count:,} bias cells (expected {expected:,})")
        if count != expected or common_count != count:
            print(f"DISCREPANCY: {iso} footprint or bias coverage differs from the reference.")
        print(f"  Missing cocoa cover: {np.count_nonzero((codes == code) & ~np.isfinite(cocoa)):,} cells")

        ref = frequency(reference, selected, cocoa)
        frequencies = []
        for name, values in models.items():
            value = frequency(values, selected, cocoa)
            frequencies.append(value)
            row = rows[iso, name]
            print(f"  {name}: {100 * value:.1f}%")
            for key, actual in (("reference", ref), ("model_value", value)):
                if not math.isclose(actual, float(row[key]), rel_tol=0, abs_tol=1e-7):
                    print(f"  DISCREPANCY {name} {key}: raster {actual:.10f}, CSV {row[key]}")
        print(f"  AgERA5 {100 * ref:.1f}%, model range {100 * min(frequencies):.1f}–"
              f"{100 * max(frequencies):.1f}%")
    print(f"Total: {np.count_nonzero(footprint):,} positive cocoa cells; "
          f"{np.count_nonzero(common):,} bias cells")


def plot(values, codes, geometry, *, title, subtitle, legend, basename, palette, norm, key):
    shape, crs, affine = geometry
    rows, cols = np.indices(shape)
    x = affine.c + (cols + 0.5) * affine.a
    y = affine.f + (rows + 0.5) * affine.e
    extent = (affine.c, affine.c + shape[1] * affine.a, affine.f + shape[0] * affine.e, affine.f)
    figure = plt.figure(figsize=(9, 7.5))
    axis = figure.add_axes((0.04, 0.20, 0.92, 0.70))
    background = np.ones((*shape, 3))
    country = np.isin(codes, list(COUNTRIES))
    background[country] = to_rgb(key.get_facecolor())
    axis.imshow(background, extent=extent, origin="upper", interpolation="nearest")
    image = axis.imshow(
        values, extent=extent, origin="upper", interpolation="nearest", cmap=palette, norm=norm
    )
    for code, (_, name, lon, lat, _) in COUNTRIES.items():
        axis.contour(x, y, codes == code, levels=[0.5], colors="#29363e", linewidths=0.8)
        label_x, label_y = transform("EPSG:4326", crs, [lon], [lat])
        axis.text(
            label_x[0], label_y[0], name, ha="center", va="center", fontsize=15,
            weight="bold", color="#17242b",
            path_effects=[effects.withStroke(linewidth=3, foreground="white")],
        )
    axis.set(xlim=extent[:2], ylim=extent[2:], aspect="equal")
    axis.set_axis_off()
    figure.text(0.5, 0.958, title, ha="center", fontsize=22, weight="bold", color="#17242b")
    figure.text(0.5, 0.920, subtitle, ha="center", fontsize=13, color="#33454f")
    colorbar = figure.colorbar(
        image, cax=figure.add_axes((0.17, 0.132, 0.66, 0.024)), orientation="horizontal"
    )
    colorbar.set_label(legend, fontsize=13)
    colorbar.ax.tick_params(labelsize=11)
    figure.legend(handles=[key], loc="lower center", bbox_to_anchor=(0.5, 0.018),
                  frameon=False, fontsize=11)
    for extension in ("pdf", "png"):
        figure.savefig(OUTPUT / f"{basename}.{extension}", dpi=300)
    plt.close(figure)


def main():
    grid_path = INPUTS / "grid_full_extent/model_grid_EPSG6933_11000m.tif"
    codes, geometry = read_band(grid_path, "country_code", nodata=255)
    _, crs, affine = geometry
    if crs.to_epsg() != 6933 or (affine.a, affine.b, affine.d, affine.e) != (11000, 0, 0, -11000):
        raise ValueError("Expected the north-up EPSG:6933 grid with 11 km cells")
    # Code 3 marks a cell center assigned to both countries in the archived grid.
    if not np.isin(codes[np.isfinite(codes)], [0, *COUNTRIES, 3]).all():
        raise ValueError("Unexpected country grid codes")
    cocoa, _ = read_band(
        INPUTS / "cocoa_mask_full_extent/cocoa_fraction_EPSG6933_11000m.tif",
        "cocoa_fraction", geometry,
    )
    manifest = ROOT / "08_outputs/gridded/late_century_suitability_change/late_century_map_manifest.json"
    model_ids = json.loads(manifest.read_text(encoding="utf-8"))["climate_models"]
    if len(set(model_ids)) != len(model_ids) or len(model_ids) != 8:
        raise ValueError("Expected eight distinct archived models")
    reference, _ = read_band(DIAGNOSTICS / "AgERA5_tasmax_climatology.tif", METRIC, geometry)
    models = {
        name: read_band(DIAGNOSTICS / f"{name}_tasmax_climatology.tif", METRIC, geometry)[0]
        for name in model_ids
    }
    country = np.isin(codes, list(COUNTRIES))
    footprint = country & (cocoa > 0)
    stack = np.stack(list(models.values()))
    common = footprint & np.isfinite(reference) & np.isfinite(stack).all(axis=0)
    cross_check(codes, cocoa, footprint, common, reference, models)
    bias = np.full(codes.shape, np.nan)
    bias[common] = np.median(100 * (stack[:, common] - reference[common]), axis=0)
    limit = max(1, math.ceil(np.max(np.abs(bias[common])) / 5) * 5)
    print(f"Bias pixel range: {np.nanmin(bias):+.2f} to {np.nanmax(bias):+.2f} percentage points")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plot(
        bias, codes, geometry, title="Historical hot-day bias over mapped cocoa",
        subtitle="Climate models minus AgERA5, 2000–2014",
        legend="Difference in days ≥35 °C (percentage points)", basename="historical_hot_day_bias",
        palette="RdBu_r", norm=TwoSlopeNorm(0, -limit, limit),
        key=Patch(facecolor=NEUTRAL, label="Country cells outside the bias map"),
    )
    plot(
        np.where(country, 100 * cocoa, np.nan), codes, geometry,
        title="Cocoa footprint on the analysis grid", subtitle="Ghana and Côte d’Ivoire",
        legend="Share of 11 km cell mapped as cocoa (%)", basename="cocoa_footprint",
        palette="YlGn", norm=Normalize(0, 100),
        key=Patch(facecolor=MISSING, label="Missing cocoa cover"),
    )


if __name__ == "__main__":
    main()
