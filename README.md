# Predicting insurance and retirement-plan subscriptions

Academic machine-learning competition project predicting two targets, `Assvie` and `Retraite`, from tabular survey features.

## Start here

Explore the [XGBoost notebook](XGB.ipynb), then the [ensemble script](boom.py). The original [French report](Rapport.docx) provides additional project context.

## Approach

- Encode categorical/ordinal variables and handle missing feature values using gradient-boosting classifiers.
- Train Random Forest, Gradient Boosting and XGBoost models.
- Explore feature selection and hyperparameter search.
- Compare calibrated variants and average model probabilities for submissions.

| File | Purpose |
| --- | --- |
| [XGB.ipynb](XGB.ipynb) | XGBoost exploration, feature selection and evaluation |
| [boom.py](boom.py) | Random Forest / Gradient Boosting probability ensemble |
| [calibrated_boom.py](calibrated_boom.py) | Calibrated ensemble variant |
| [models/](models/) | Model helpers |
| [CSV/](CSV/) | Historical submission files |

## Data and execution

The repository contains `train.csv`, `test.csv` and `sample_submission.csv`. Check the original competition's data-use terms before reusing or redistributing them.

Use the repository root as the working directory. The scripts expect the CSV files at that level and write submissions into `CSV/`. Dependencies include NumPy, pandas, scikit-learn and XGBoost; the notebook also imports matplotlib, seaborn and SciPy. Original dependency versions are not locked, so compatibility must be checked before running the older model-selection code.

Review preprocessing and model settings before running `python boom.py` or `python calibrated_boom.py`. These commands train models and write output files; notebook cells include potentially expensive searches.

## Evaluation limits

The notebook contains historical local evaluation outputs, but no verified final leaderboard placement is claimed here. Preprocessing and feature selection must be fitted within training folds to obtain leakage-resistant estimates. The scripts also fit categorical encoders separately for train and test data, a behaviour that should be reviewed before using predictions outside the original exercise.

This repository documents an academic experiment, with no production insurance or financial decision-making claim. Individual ownership of each component is not established by this README.
