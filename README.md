# SIMPLE-Cacao-WA

This repository contains a reproducible climate-impact analysis of cocoa in Ghana and Côte d’Ivoire. It compares a historical climate-response reference with late-century climate scenarios using the project's SIMPLE-Cacao-WA implementation and a mapped cocoa-growing footprint. The outputs are relative climatic-response proxies, not observed yields or forecasts of farm production.

## Research question and analysis

The analysis asks how the climate component of cocoa suitability may differ between the historical reference period and 2060–2079. It evaluates two separate flowering-season windows:

- **Major rainy season:** April, May, and June
- **Minor rainy season:** September, October, and November

For each window, the first day of each listed month is run as a separate cohort-start case. The three monthly relative changes are averaged equally for each climate model; the result shown is then the median across eight equally weighted climate models. The monthly cases are timing sensitivities, not flowering-frequency weights or an annual-production estimate.

The historical cohort-start reference is 2000–2019, calculated from AgERA5 daily weather for 2000–2020 (2020 supplies trailing weather for late-2019 cohorts). Future cohorts start in 2060–2079. Their weather forcing includes 2080 to complete the final cohort cycles. The comparison is cellwise relative change from the historical reference.

## Main result

![Modeled cocoa climatic-suitability change by flowering season](10_figures/maps/late_century_seasonal_suitability_change.png)

The table gives the country-level cocoa-area-weighted median of the map's modeled relative change. Each value is the median across eight GCMs; it is **not a yield change in tonnes or a prediction for individual farms**.

| Flowering window | Scenario | Ghana | Côte d’Ivoire |
|---|---|---:|---:|
| Major rainy season (Apr–Jun) | SSP2-4.5 | +6.6% | +16.4% |
| Major rainy season (Apr–Jun) | SSP5-8.5 | -4.4% | +3.8% |
| Minor rainy season (Sep–Nov) | SSP2-4.5 | -12.7% | -4.6% |
| Minor rainy season (Sep–Nov) | SSP5-8.5 | -20.5% | -16.0% |

These are summaries of the modeled climate response over the mapped cocoa area. They do not include farm management, disease, farmer adaptation, or other non-climatic drivers. Model agreement varies by country, season, and scenario; the gridded outputs and full run metadata are provided below.

The four GeoTIFF maps, checksums, ensemble details, and country summaries are in [08_outputs/gridded/late_century_suitability_change](08_outputs/gridded/late_century_suitability_change) and its [map manifest](08_outputs/gridded/late_century_suitability_change/late_century_map_manifest.json).

## Data and methods

### Climate data

- **Future forcing:** NASA NEX-GDDP-CMIP6 version 2.0 daily data, at nominal 0.25° resolution, for precipitation (`pr`), minimum and maximum temperature (`tasmin`, `tasmax`), and surface downwelling shortwave radiation (`rsds`). One member (`r1i1p1f1`) is used for each of eight models: ACCESS-CM2, BCC-CSM2-MR, CanESM5, CMCC-ESM2, MIROC6, MPI-ESM1-2-HR, NorESM2-MM, and TaiESM1. The scenarios are SSP2-4.5 (`ssp245`) and SSP5-8.5 (`ssp585`). Data are regridded to the project's approximately 11-km equal-area analysis grid; this finer grid does not add climate information beyond the source resolution.
- **Historical reference:** daily AgERA5 data from the Copernicus Climate Data Store for 2000–2020. The stored historical model response is the baseline for the published future comparison.
- **Climate-model diagnostics:** historical NEX-GDDP-CMIP6 files for 2000–2014 and CHIRPS v3 precipitation comparisons are used by separate screening and validation workflows. These are not the future scenario forcing or the source of the historical map baseline.

### Cocoa footprint and analysis grid

The cocoa mask is derived from Abu et al.'s mapped cocoa plantations in Ghana and Côte d’Ivoire. That source represents a recent, Sentinel-derived footprint (described for 2019), not a historical cocoa-area time series. Its binary source classes are averaged onto an approximately 11-km equal-area grid, producing a fractional mapped cocoa area per cell. Country-level summaries are weighted by those fractions and equal-area cell size.

### Soil and model configuration

Soil water-holding capacity is derived from ISRIC SoilGrids 2.0 median predictions (Q0.5), using field capacity, wilting point, and coarse-fragment volume for soil layers through a 1,500-mm root depth. The primary reference parameterization uses published cacao parameter anchors without calibration to Ghanaian or Côte d’Ivoire farm yields.

The primary comparison sets the model water-stress response (`S_water`) and CO₂ response (`S_CO2`) to zero. The future climate input has no reference evapotranspiration (`ET0`); therefore, the future comparison does not represent drought or excess-rainfall damage. Having soil data in the input set does not make this result drought-adjusted.

The flowering-month windows are regional literature proxies, not validated country-specific flowering calendars. See [the seasonal cohort calendar](03_parameters/flowering/cacao_seasonal_cohort_calendar.yml) and [the baseline parameter configuration](03_parameters/crop/cacao_suitability_baseline.yml) for exact assumptions.

### Dataset provenance and citations

The acquisition and processing manifests in [01_inputs/manifests](01_inputs/manifests) record source URLs, licenses where available, and checksums. Cite the source data and methods as applicable:

1. **NEX-GDDP-CMIP6:** Thrasher, B. et al. (2022). “NASA’s Global Daily Downscaled Projections, CMIP6.” *Scientific Data*, 9, 262. [https://doi.org/10.1038/s41597-022-01393-4](https://doi.org/10.1038/s41597-022-01393-4). NASA NCCS technical documentation: [https://doi.org/10.7917/OFSG3345](https://doi.org/10.7917/OFSG3345).
2. **AgERA5:** Copernicus Climate Change Service, Agrometeorological indicators from 1979 to present. [Dataset documentation and DOI](https://doi.org/10.24381/cds.6c68c9bb).
3. **Cocoa map:** Abu, I-O., Szantoi, Z., Brink, A., Robuchon, M., and Thiel, M. (2021). “Detecting cocoa plantations in Cote d'Ivoire and Ghana and their implications on protected areas.” *Ecological Indicators*, 129, 107863. [https://doi.org/10.1016/j.ecolind.2021.107863](https://doi.org/10.1016/j.ecolind.2021.107863). Dataset: [PANGAEA 917473](https://doi.org/10.1594/PANGAEA.917473).
4. **SoilGrids:** Hengl, T. et al. (2017). “SoilGrids250m: Global gridded soil information based on machine learning.” *PLOS ONE*, 12(2), e0169748. [https://doi.org/10.1371/journal.pone.0169748](https://doi.org/10.1371/journal.pone.0169748).
5. **Seasonal timing:** Claus, G. et al. (2018). “Challenges in Cocoa Pollination: The Case of Cote d'Ivoire.” [https://doi.org/10.5772/intechopen.75361](https://doi.org/10.5772/intechopen.75361); Adjaloo, M. K. et al. (2012). “Floral Phenology of Upper Amazon Cocoa Trees: Implications for Reproduction and Productivity of Cocoa.” [https://doi.org/10.5402/2012/461674](https://doi.org/10.5402/2012/461674).
6. **This software and analysis:** use the project metadata in [CITATION.cff](CITATION.cff). The implementation and parameter choices are documented in this repository; no separate DOI for this project is currently recorded there.

## Data included in this repository

The repository includes the analysis code, configuration, source-data manifests, checksums, small processed map inputs (grid, cocoa fraction, and soil AWC), historical reference outputs, future map outputs, and figures. It does **not** commit the downloaded raw datasets, the bulk daily AgERA5 rasters, the bulk CHIRPS validation input, or the retrieved reference-article PDF. These large or separately licensed assets remain excluded by [.gitignore](.gitignore); their source and processing information is retained in manifests. They are not deleted from a user's local working copy by this ignore rule.

The full future NEX-GDDP-CMIP6 time series is needed to regenerate the future map, but is intentionally acquired separately rather than stored in Git. From the project root, `python .\workflow\cli.py scenarios` downloads and audits the 1,344 regional daily NetCDF files for the eight models, two scenarios, four variables, and years 2060–2080. This is a multi-gigabyte download. The historical AgERA5 reference raster and the small static rasters needed by the map are already included.

AgERA5 source downloads require a Copernicus Climate Data Store account and accepted dataset terms. The optional historical climate screening, precipitation validation, and raw-input preparation workflows may require additional large downloads; they are not required to reproduce the included map if its archived historical reference and processed inputs are used.

## Reproduce the map

From the project root, create the environment:

```powershell
conda env create -f .\13_env\environment.yml
conda activate simple-cacao-wa
```

Download and verify the future climate forcing, then generate the figure and four GeoTIFF outputs:

```powershell
python .\workflow\cli.py scenarios
python .\workflow\cli.py map
```

The scenario acquisition takes time and disk space. The included scenario manifest and audit record identify the expected files and checksums. Run tests and lint with:

```powershell
python -m pytest
python -m ruff check .
```

## Project layout

- `01_inputs/` — source manifests, small processed map inputs, and external metadata
- `02_preprocessing/`, `03_parameters/`, `04_model/` — data preparation, parameterization, model, and tests
- `06_validation/`, `07_runs/`, `09_analysis/` — validation, run records, and supporting analysis
- `08_outputs/`, `10_figures/` — gridded results, summaries, and figures
- `11_notebooks/`, `13_env/`, `workflow/` — notebooks, environment, and workflow entry points
