# Customer Churn Prediction

## Project overview

This project analyzes the supplied Cell2Cell telecom churn data, compares seven classification pipelines, saves the selected fitted pipeline, and provides a Streamlit interface for single-customer or CSV predictions.

The original IBM Telco analysis and model are preserved as `notebooks/churn_analysis_ibm_backup.ipynb` and `models/churn_model_ibm_backup.pkl`. The active notebook, model and app now use the Cell2Cell feature schema.

The `notebooks/` folder contains only these two notebooks: the original IBM notebook and the active Cell2Cell notebook. The Cell2Cell notebook follows the original's code-first style: small numbered code cells, comments instead of long markdown sections, and a graph for each model feature. Dataset explanations, EDA observations, preprocessing decisions, and metric interpretation are documented here.

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
- Compare all original models on the same held-out test rows at their default threshold.
- Separately compare weighted and unweighted Logistic Regression, Random Forest, and XGBoost. Select F1- and accuracy-oriented thresholds using an internal validation split, then report every variant on the same untouched test rows.

## Exploratory analysis

The notebook creates 116 EDA graphs from the supplied data: churn counts and percentages; a distribution and a by-churn boxplot for each of the 34 numeric features; category counts by churn and churn rates for each of the 22 categorical features; and numeric-feature correlation views. Since `ServiceArea` has 747 categories, its two plots focus on the 15 most common areas so the charts remain readable. Boxplots hide extreme points for readability, but the data is not changed. The graphs describe observed associations and do not establish causes.

## Models and evaluation

The notebook compares Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbours, Complement Naive Bayes, a calibrated linear Support Vector Machine, and XGBoost. Complement Naive Bayes uses a non-negative numeric scaling branch. A linear SVM is used to avoid the scaling cost of a kernel SVM on 51,047 rows.

Because the current goal is higher overall accuracy, the notebook selects the candidate and decision threshold by validation accuracy, with validation F1, recall, and precision as tie-breakers. Test metrics are not used for selection. The selected candidate is **unweighted XGBoost, depth 4**, with a validation-selected threshold of **0.495**.

| Untouched test metric | Result |
|---|---:|
| Accuracy | 0.7193 |
| Precision (churn) | 0.5779 |
| Recall (churn) | 0.0959 |
| F1 (churn) | 0.1644 |
| ROC-AUC | 0.6774 |

The all-No baseline accuracy on the same test split is **0.7119**. This model improves accuracy by only **0.74 percentage points** over that baseline and detects just **9.6%** of churners (it misses 2,660 of 2,942 churners). This is an accuracy-focused trade-off, not a strong churn-detection model; use the preserved F1-focused model if detecting churners matters more than maximizing accuracy.

The table below compares the original seven model configurations at their default 0.5 threshold. The notebook also includes a paired weighted/unweighted comparison for Logistic Regression, Random Forest, and XGBoost at default and validation-tuned thresholds.

Paired comparison at each candidate's validation-F1 threshold:

| Candidate | Threshold | Test accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression (unweighted) | 0.255 | 0.5152 | 0.3388 | 0.7172 | 0.4602 | 0.6107 |
| Logistic Regression (balanced) | 0.420 | 0.4603 | 0.3247 | 0.8086 | 0.4634 | 0.6110 |
| Random Forest (unweighted) | 0.260 | 0.5219 | 0.3532 | 0.7927 | 0.4886 | 0.6605 |
| Random Forest (balanced) | 0.430 | 0.5468 | 0.3630 | 0.7587 | 0.4910 | 0.6590 |
| XGBoost (unweighted) | 0.245 | 0.5398 | 0.3641 | 0.7998 | 0.5004 | 0.6774 |
| XGBoost (weighted) | 0.460 | 0.5635 | 0.3721 | 0.7492 | 0.4972 | 0.6764 |

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| XGBoost | 0.626 | 0.405 | 0.635 | 0.494 | 0.677 |
| Random Forest | 0.658 | 0.418 | 0.478 | 0.446 | 0.659 |
| Decision Tree | 0.587 | 0.363 | 0.571 | 0.444 | 0.608 |
| Logistic Regression | 0.579 | 0.356 | 0.574 | 0.440 | 0.611 |
| Complement Naive Bayes | 0.568 | 0.337 | 0.516 | 0.408 | 0.576 |
| K-Nearest Neighbours | 0.693 | 0.375 | 0.099 | 0.157 | 0.571 |
| Calibrated linear SVM | 0.713 | 0.549 | 0.017 | 0.033 | 0.610 |

Final model confusion matrix, with rows as actual `No`/`Yes` and columns as predicted `No`/`Yes`:

| | Predicted No | Predicted Yes |
|---|---:|---:|
| Actual No | 7,062 | 206 |
| Actual Yes | 2,660 | 282 |

The validation-accuracy winner is unweighted XGBoost at threshold 0.495. Weighted XGBoost at its own validation-accuracy threshold (0.700) scored 0.7219 test accuracy, but it was not chosen: selecting it because of that test result would leak test information. Both accuracy-focused thresholds produce very low churn recall. All figures come from the same stratified test split; the unlabeled holdout has no evaluation metrics.

## Saved model and Streamlit app

`models/churn_model.pkl` contains the complete fitted unweighted XGBoost preprocessing-plus-classifier Pipeline and its validation-selected accuracy threshold, with the expected 56 Cell2Cell feature names. The F1-focused Cell2Cell model is preserved as `models/churn_model_cell2cell_f1_backup.pkl`; the accuracy model snapshot is retained as `models/churn_model_cell2cell_accuracy_backup.pkl`; the original IBM model is retained as `models/churn_model_ibm_backup.pkl`.

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
  churn_model_cell2cell_accuracy_backup.pkl
  churn_model_cell2cell_f1_backup.pkl
  churn_model_ibm_backup.pkl
notebooks/
  churn_analysis.ipynb
  churn_analysis_ibm_backup.ipynb
requirements.txt
README.md
```
