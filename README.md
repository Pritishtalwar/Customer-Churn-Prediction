# Customer Churn Prediction

## Project overview

This project analyzes the supplied Cell2Cell telecom churn data, compares seven classification pipelines, saves the selected fitted pipeline, and provides a Streamlit interface for single-customer or CSV predictions.

The original IBM Telco analysis and model are preserved as `notebooks/churn_analysis_ibm_backup.ipynb` and `models/churn_model_ibm_backup.pkl`. The active notebook, model and app now use the Cell2Cell feature schema.

## Dataset

The CSV files were supplied in the project workspace. Their upstream publisher and license were not recorded in the repository, so verify and cite the source/license separately before redistributing them.

| File | Actual shape | Churn status |
|---|---:|---|
| `dataset/cell2celltrain.csv` | 51,047 rows × 58 columns | 36,336 No; 14,711 Yes |
| `dataset/cell2cellholdout.csv` | 20,000 rows × 58 columns | `Churn` exists but all 20,000 values are missing |

Both files have the same columns. Each contains zero exact duplicate rows and unique, non-missing `CustomerID` values; there are no IDs shared between train and holdout. Training churn prevalence is 28.82%, so accuracy alone is not an adequate selection metric.

The training file has 3,515 missing cells across 14 columns. The holdout has 21,304 missing cells, including all 20,000 target values and 1,304 missing predictor values. Missing data is imputed in the fitted pipelines; rows are not dropped. The notebook prints full per-column data types and missing-value counts.

The training file has 58 source columns. `CustomerID` and `Churn` are excluded from the 56 model inputs; the ID remains available in raw data and prediction results. The unlabeled holdout is reserved for inference and is not used to calculate evaluation metrics.

## Cleaning and preprocessing

- Validate that the training target contains only `Yes` and `No`, then map them to 1 and 0.
- Preserve source rows and columns; keep `CustomerID` for tracing but never pass it to the model.
- Split numerical and categorical predictors and impute missing values separately.
- Standardize numerical inputs and one-hot encode categorical inputs with unknown categories ignored. Categories with fewer than 20 training examples are grouped to control the high-cardinality `ServiceArea` feature.
- Use one stratified 80/20 split (40,837 train and 10,210 test rows). Every transform is fitted within each model Pipeline using training rows only.
- Compare all models on the same held-out test rows. Class weighting is applied where supported to address churn imbalance.

## Models and evaluation

The notebook compares Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbours, Complement Naive Bayes, a calibrated linear Support Vector Machine, and XGBoost. Complement Naive Bayes uses a non-negative numeric scaling branch. A linear SVM is used to avoid the scaling cost of a kernel SVM on 51,047 rows.

The final model is selected by churn-class F1 on the held-out test split, with ties resolved by recall, ROC-AUC, precision, and accuracy. On this run, **XGBoost** had the highest F1:

| Test metric | Result |
|---|---:|
| Accuracy | 0.6259 |
| Precision (churn) | 0.4049 |
| Recall (churn) | 0.6349 |
| F1 (churn) | 0.4944 |
| ROC-AUC | 0.6775 |

Comparison table on that same test split (rounded to three decimals):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| XGBoost | 0.626 | 0.405 | 0.635 | 0.494 | 0.677 |
| Random Forest | 0.658 | 0.418 | 0.478 | 0.446 | 0.659 |
| Decision Tree | 0.587 | 0.363 | 0.571 | 0.444 | 0.608 |
| Logistic Regression | 0.579 | 0.356 | 0.574 | 0.440 | 0.611 |
| Complement Naive Bayes | 0.568 | 0.337 | 0.516 | 0.408 | 0.576 |
| K-Nearest Neighbours | 0.693 | 0.375 | 0.099 | 0.157 | 0.571 |
| Calibrated linear SVM | 0.713 | 0.549 | 0.017 | 0.033 | 0.610 |

Confusion matrix, with rows as actual `No`/`Yes` and columns as predicted `No`/`Yes`:

| | Predicted No | Predicted Yes |
|---|---:|---:|
| Actual No | 4,522 | 2,746 |
| Actual Yes | 1,074 | 1,868 |

These are measured results for the current 80/20 split and default decision threshold, not a claim of improvement over the previous IBM model. Consider the false negatives and false positives alongside churn recall, precision, F1, ROC-AUC and accuracy. The unlabeled holdout has no evaluation metrics.

## Saved model and Streamlit app

`models/churn_model.pkl` contains the complete fitted preprocessing-plus-classifier Pipeline, with the expected 56 Cell2Cell feature names. The previous IBM model is retained separately as `models/churn_model_ibm_backup.pkl`.

The app obtains numerical defaults, categorical choices and feature names from the fitted Pipeline. It supports manual entry and CSV uploads, validates required and unexpected columns and numeric values, rejects uploads containing observed `Churn` labels to prevent target leakage, and retains an optional `CustomerID` in downloaded results. The displayed probability bands are illustrative and are not business-validated intervention thresholds.

## Setup and run

From the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the app:

```powershell
streamlit run app/app.py
```

To retrain or inspect the analysis, open `notebooks/churn_analysis.ipynb` in VS Code, select the project `.venv` Python interpreter, and run all cells from top to bottom. Training writes the fitted Pipeline to `models/churn_model.pkl`; keep the original Cell2Cell CSVs in `dataset/`.

## Project structure

```text
app/
  app.py
dataset/
  cell2celltrain.csv
  cell2cellholdout.csv
models/
  churn_model.pkl
  churn_model_ibm_backup.pkl
notebooks/
  churn_analysis.ipynb
  churn_analysis_ibm_backup.ipynb
requirements.txt
README.md
```
