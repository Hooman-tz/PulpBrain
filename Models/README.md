# Models

| File | What it is |
|---|---|
| `TensileIndex_Fused_BW.joblib` | The **one demo model** bundled with this repository: an `xgboost.XGBRegressor` predicting tensile index from the fused Raman + fibre features. It was trained on proprietary UBC–Canfor data (not distributed). |
| `Scaler.joblib` | A placeholder (unfitted) `RobustScaler`. The application re-fits the scaler on the selected batch every time a prediction is run, so no pre-computed statistics are stored. |
| `feature_names.txt` | The 932 input columns, **in the order the models expect**: 282 pulp/fibre features from the `selected_variables` table (`CsfAvg`, `LWeightCurl`, …, `Refining_Energy`) followed by `Spectrum_050` … `Spectrum_699` (the preprocessed Raman pixels). |

The other models of the full PulpBrain set were trained on proprietary Canfor data and are not part of this repository. The approach is described in [Poursorkh *et al.*, Anal. Chem. 2025, 97, 15808](https://doi.org/10.1021/acs.analchem.5c01768).

## Using your own models

On every **Run Prediction** the application loads **every** `*.joblib` file in this folder whose name does not start with `Scaler`, calls `.predict()` on the scaled feature table, and adds the result as a column named after the part of the file name before the first underscore (`TensileIndex_Fused_BW.joblib` → column `TensileIndex`). So to add a property:

1. Train a regressor on a table with exactly the columns of `feature_names.txt`, in that order (a pandas `DataFrame`, so XGBoost records the column names).
2. Save it as `Models/<Property>_Fused_BW.joblib` with `joblib.dump`. Use `scikit-learn==1.4.2` and `xgboost==2.0.3` (see `requirements.txt`), because the files are pickles.
3. If the database `targets` table holds measured values for that property, they are shown next to the prediction as `<Property>_Measured`.

> Only load `.joblib` files you trust: unpickling can execute arbitrary code.
