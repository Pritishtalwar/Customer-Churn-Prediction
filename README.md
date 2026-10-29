# Customer Churn Prediction System

## 1. Project Overview

Customer Churn Prediction System is a Machine Learning project that predicts whether a telecom customer is likely to churn or stay.

The project uses the IBM Telco Customer Churn dataset and compares multiple classification algorithms.

The final selected model is Logistic Regression, which is integrated into a Streamlit web application.

---

## 2. Problem Statement

Customer churn occurs when a customer stops using a company's services.

The objective of this project is to:

- Analyze customer information
- Identify patterns associated with churn
- Train machine learning classification models
- Compare model performance
- Predict the probability of customer churn
- Classify customers into different risk levels

---

## 3. Dataset

Dataset: IBM Telco Customer Churn Dataset

Original dataset:

- Rows: 7043
- Columns: 21

After data cleaning:

- Rows: 7032
- Columns used for modeling: 19 input features

Target variable:

- Churn = 1 → Customer churned
- Churn = 0 → Customer stayed

The `customerID` column was removed because it is a unique customer identifier and is not useful as a predictive feature.

---

## 4. Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Joblib
- Streamlit

---

## 5. Project Workflow

Dataset
↓
Data Cleaning
↓
Exploratory Data Analysis
↓
Feature Preparation
↓
Train-Test Split
↓
Categorical Encoding
↓
Feature Scaling
↓
Model Training
↓
Model Evaluation
↓
Model Comparison
↓
Final Model Pipeline
↓
Model Saving
↓
Streamlit Application
↓
Churn Prediction

---

## 6. Data Preprocessing

The following preprocessing steps were performed:

1. Loaded the dataset using Pandas.
2. Converted `TotalCharges` into numeric format.
3. Invalid/blank values were converted to missing values.
4. Rows containing invalid `TotalCharges` values were removed.
5. Removed the `customerID` identifier.
6. Converted the target variable `Churn`:
   - Yes → 1
   - No → 0
7. Separated features (X) and target (y).
8. Numerical features were standardized using StandardScaler.
9. Categorical features were converted using OneHotEncoder.

The 19 original input features became 45 processed features after one-hot encoding.

---

## 7. Exploratory Data Analysis

Important patterns observed during analysis:

- Month-to-month contract customers show higher churn tendency.
- Customers with lower tenure show higher churn tendency.
- Electronic check users show comparatively higher churn.
- Customers without services such as Online Security and Tech Support show higher churn tendency.
- Monthly charges are associated with differences in churn behavior.

These observations represent associations in the dataset and should not be interpreted as direct causal relationships.

---

## 8. Machine Learning Models

The following classification algorithms were trained and compared:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. K-Nearest Neighbors (KNN)
5. Naive Bayes
6. Support Vector Machine (SVM)
7. XGBoost

---

## 9. Model Evaluation

Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix

Cross-validation was also performed to check model consistency.

---

## 10. Final Model

Logistic Regression was selected as the final model based on its overall performance across the evaluation metrics.

Approximate test performance:

- Accuracy: 80.38%
- Precision: 64.85%
- Recall: 57.22%
- F1-Score: 60.80%
- ROC-AUC: 83.59%

The model was trained inside a Scikit-learn Pipeline containing preprocessing and the Logistic Regression classifier.

---

## 11. Model Validation

The dataset was divided into:

- 80% training data
- 20% testing data

Training samples: 5625

Testing samples: 1407

Five-fold cross-validation was also performed.

The Logistic Regression training and testing accuracy were approximately:

- Training Accuracy: 80.46%
- Testing Accuracy: 80.38%

The small difference indicates no obvious overfitting.

---

## 12. Streamlit Application

A Streamlit web application was created for prediction.

The application allows the user to enter customer information and provides:

- Churn probability
- Churn prediction
- Risk level

Risk levels are defined using a simple probability-based rule:

- Probability < 40% → Low Risk
- 40%–69.99% → Medium Risk
- ≥ 70% → High Risk

---

## 13. Project Structure

```text
customer churn prediction/
│
├── .venv/
│
├── app/
│   └── app.py
│
├── dataset/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── models/
│   └── churn_model.pkl
│
├── notebooks/
│   └── churn_analysis.ipynb
│
└── requirements.txt