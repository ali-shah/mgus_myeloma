# MGUS to Multiple Myeloma Transformation Prediction

This project contains an exploratory machine-learning script for estimating the time from a patient's biomarker history to transformation from MGUS to multiple myeloma. The script engineers biomarker progression features and compares regression models using a patient-level train/test split.

## Project contents

- `src/predict_transformation_advanced.py` - data preparation, feature engineering, model training, evaluation, and artifact generation.
- `data/cancer_patients_89.csv` - local input used by the prediction script; it is not included in version control. It expects columns including `UniqueID`, `DateTestTaken`, `TestCode`, `Result`, and `TransformationDate`.
- `data/mgus_patients_104.csv` - additional local dataset; it is not currently loaded by this script and is not included in version control.

## Requirements

Python 3 and the following packages are required:

```bash
python3 -m pip install pandas numpy scikit-learn joblib
```

## Run

From the project root:

```bash
python3 src/predict_transformation_advanced.py
```

The script reads `data/cancer_patients_89.csv` and creates the `output/` directory if needed. Existing output files with the same names are overwritten on each run.
Before running, place an authorized copy of `cancer_patients_89.csv` in `data/`.

## Generated files

- `output/transformation_prediction_models.csv` - model comparison metrics.
- `output/transformation_predictions_detailed.csv` - per-patient prediction details.
- `output/model_advanced_*.pkl` - serialized trained models and, where applicable, scalers and feature names.

## Interpretation

This is an exploratory research project, not a clinically validated tool. Do not use its predictions for diagnosis, treatment, or patient-care decisions. Validate the methodology and results before drawing conclusions. In particular, the current detailed patient report may not align prediction values with the corresponding patient IDs, so review that output cautiously.

The datasets and generated outputs are intentionally excluded from version control because they contain patient-level clinical information. Only commit data that has been appropriately de-identified and approved for sharing.