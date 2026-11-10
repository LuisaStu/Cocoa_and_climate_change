from __future__ import annotations

import argparse
from collections.abc import Sequence


def compare_weights() -> None:
    """Compare proposed weights using model indices, not a synthetic daily ensemble."""

    import hashlib
    import json
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import yaml

    root = Path("06_validation/historical_gcm/daily_spatial")
    source = root / "spatial_diagnostics.csv"
    config = Path("config/gcm_ensemble_proposal.yml")
    proposal = yaml.safe_load(config.read_text())
    models = proposal["primary"]["models"]
    groups = proposal["sensitivity"]["groups"]
    flat = [model for group in groups for model in group]
    if len(flat) != len(set(flat)) or set(flat) != set(models):
        raise ValueError("Grouping must include every candidate exactly once")
    weights = {
        "equal_model": {model: 1 / len(models) for model in models},
        "component_group": {model: 1 / len(groups) / len(group) for group in groups for model in group},
    }
    for scheme in weights.values():
        if not np.isclose(sum(scheme.values()), 1):
            raise ValueError("Weights do not sum to one")
    data = pd.read_csv(source)
    rows = []
    for keys, frame in data.groupby(["variable", "metric", "country", "domain", "units"]):
        if len(frame) != len(models) or set(frame.model) != set(models):
            raise ValueError("Incomplete candidate diagnostics")
        if not np.allclose(frame.reference, frame.reference.iloc[0]):
            raise ValueError("Reference differs by model")
        row = dict(zip(["variable", "metric", "country", "domain", "units"], keys, strict=True))
        for scheme, values in weights.items():
            row[f"{scheme}_bias"] = np.average(frame.bias, weights=frame.model.map(values))
        row["weighting_change"] = row["component_group_bias"] - row["equal_model_bias"]
        row["minimum_model_bias"] = frame.bias.min()
        row["maximum_model_bias"] = frame.bias.max()
        rows.append(row)
    out = root / "ensemble_weight_sensitivity.csv"
    pd.DataFrame(rows).to_csv(out, index=False)


    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()


    manifest = {
        "weights": weights,
        "sha256": {str(p): digest(p) for p in [source, config, out, Path(__file__)]},
    }
    (root / "ensemble_weight_sensitivity_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Wrote {len(rows)} weighting comparisons.")


def plot_screening() -> None:
    """Plot cocoa-weighted historical differences from the completed diagnostics."""

    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    source = Path("06_validation/historical_gcm/daily_spatial/spatial_diagnostics.csv")
    data = pd.read_csv(source)
    data = data[data.domain == "cocoa"]
    panels = [
        ("pr", "wet_day_fraction", "Wet-day fraction", "percentage points", 100),
        ("pr", "longest_dry_spell_within_year", "Longest dry spell within year", "days", 1),
        ("pr", "p95", "Annual daily rainfall 95th percentile", "mm/day", 1),
        ("tasmax", "fraction_days_ge35C", "Days at or above 35°C", "percentage points", 100),
        ("tasmax", "p95", "Annual daily Tmax 95th percentile", "°C", 1),
        ("rsds", "p95", "Annual daily radiation 95th percentile", "MJ/m²/day", 1),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    for ax, (variable, metric, title, units, scale) in zip(axes.flat, panels, strict=True):
        subset = data[(data.variable == variable) & (data.metric == metric)]
        table = subset.pivot(index="model", columns="country", values="bias")[["GHA", "CIV"]]
        values = table.to_numpy() * scale
        bound = max(abs(values).max(), 0.001)
        ax.imshow(values, cmap="RdBu_r", vmin=-bound, vmax=bound, aspect="auto")
        ax.set_xticks([0, 1], ["Ghana", "Côte d'Ivoire"])
        ax.set_yticks(np.arange(len(table)), table.index)
        ax.tick_params(axis="both", length=0, labelsize=9)
        ax.set_title(f"{title}\nModel − AgERA5 ({units})", fontsize=11, pad=12)
        for (r, c), value in np.ndenumerate(values):
            ax.text(
                c,
                r,
                f"{value:+.1f}",
                ha="center",
                va="center",
                color="white" if abs(value) > 0.55 * bound else "black",
                fontsize=10,
            )
    fig.suptitle(
        "Historical climate differences over mapped cocoa areas, 2000–2014\n"
        "Cocoa-fraction-weighted cell indices; equal-year averages; colour scale differs by panel",
        fontsize=15,
    )
    out = Path("10_figures/diagnostics")
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "gcm_cocoa_daily_bias.png", dpi=180)
    fig.savefig(out / "gcm_cocoa_daily_bias.pdf")
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run GCM diagnostic analyses.")
    parser.add_argument("operation", choices=("weights", "plot"))
    operation = parser.parse_args(argv).operation
    if operation == "weights":
        compare_weights()
    else:
        plot_screening()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
