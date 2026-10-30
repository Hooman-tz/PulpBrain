# Synthetic sample data

A small, **entirely synthetic** dataset for trying PulpBrain without any real measurements.
It contains no data from Canfor, UBC or any mill: every value is generated from hand-written,
textbook-plausible ranges with a fixed random seed. The numbers are *not* physically meaningful
measurements, so predictions made from them only demonstrate that the pipeline runs.

| Path | What it is |
|---|---|
| `raman/LRyy-nnn_YYMMDD-HHMMSS.csv` | 60 Wasatch WP785 spectra in the **BarrelWise API** format: headerless, one column of 1024 pixel intensities. 20 samples × 3 replicate scans (the importer averages replicates). |
| `pulpeye_sample.csv` | PulpEye-style export with 60 rows: the same 20 samples × 3 refining levels (50, 100, 150). Columns are positional, as the importer expects; row 12 holds column titles and data starts on row 13. |
| `generate_sample_data.py` | Regenerates both (`python generate_sample_data.py --samples 20 --seed 42`). Needs only `numpy`. |

The spectra use generic cellulose bands (e.g. 380, 1095, 1121, 1378 cm⁻¹) and a lignin band near 1600 cm⁻¹
on a sloping fluorescence background, mapped onto detector pixels with the WP785 calibration constants.

## Use

1. Set up MySQL and the schema as described in the main README.
2. `python pulpbrain.py`, then **Add**.
3. Raman folder: `sample_data/raman`. PulpEye file: `sample_data/pulpeye_sample.csv`. Run.
4. Set the date range to **2024-06-01 → 2024-07-31**, click **Load Data**, then open the **Prediction** tab.

Importing should yield 20 samples (60 sample/refining-level rows) and a tensile-index prediction for every row (the only model bundled).
The "Measured" columns are random dummy values in the same units as the model outputs.

Note: the file names must keep the `LRyy-nnn` prefix and the `YYMMDD-HHMMSS` timestamp, and the PulpEye
sample names (`LRyy-nnn-<level>`) must match them; that is how the importer pairs spectra with rows.
