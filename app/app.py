from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline


st.set_page_config(
    page_title="Cell2Cell Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.pkl"


@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run the Cell2Cell notebook first."
        )

    pipeline = joblib.load(MODEL_PATH)
    if not isinstance(pipeline, Pipeline):
        raise TypeError("The saved model must be a fitted scikit-learn Pipeline.")
    if not hasattr(pipeline, "feature_names_in_"):
        raise ValueError("The saved Pipeline does not expose its expected input schema.")
    if not hasattr(pipeline, "predict_proba"):
        raise ValueError("The saved Pipeline must support churn probability predictions.")
    return pipeline


model = load_model()
decision_threshold = float(getattr(model, "decision_threshold", 0.5))
if not 0 <= decision_threshold <= 1:
    raise ValueError("The saved model decision threshold must be between 0 and 1.")
feature_columns = list(model.feature_names_in_)
preprocessor = model.named_steps["preprocessor"]
numeric_features = []
categorical_features = []
for name, _, columns in preprocessor.transformers_:
    if name == "num":
        numeric_features = list(columns)
    elif name == "cat":
        categorical_features = list(columns)

numeric_imputer = preprocessor.named_transformers_["num"].named_steps["imputer"]
numeric_defaults = dict(zip(numeric_features, numeric_imputer.statistics_))
categorical_encoder = (
    preprocessor.named_transformers_["cat"].named_steps["onehot"]
)
categorical_options = {
    feature: categorical_encoder.categories_[index].tolist()
    for index, feature in enumerate(categorical_features)
}

if set(numeric_features).union(categorical_features) != set(feature_columns):
    raise ValueError("Model preprocessing schema does not match the saved feature names.")

st.title("Cell2Cell Customer Churn Prediction")
st.write(
    "Predict churn for Cell2Cell customers. The fitted model Pipeline handles "
    "imputation, scaling, and encoding."
)
st.caption(
    f"Model expects {len(feature_columns)} features. CustomerID is retained for "
    f"tracking uploaded rows, but is not a model input. The saved decision threshold "
    f"is {decision_threshold:.3f}."
)


def validate_features(frame):
    if frame.empty:
        raise ValueError("The uploaded CSV must contain at least one customer row.")

    missing = [feature for feature in feature_columns if feature not in frame.columns]
    allowed_extra = {"CustomerID", "Churn"}
    unexpected = [
        column
        for column in frame.columns
        if column not in feature_columns and column not in allowed_extra
    ]
    if missing or unexpected:
        messages = []
        if missing:
            messages.append(f"Missing required features: {', '.join(missing)}")
        if unexpected:
            messages.append(f"Unexpected columns: {', '.join(unexpected)}")
        raise ValueError("; ".join(messages))

    if "Churn" in frame.columns and frame["Churn"].notna().any():
        raise ValueError(
            "The uploaded Churn column contains labels. Remove it before prediction "
            "to prevent target leakage."
        )

    features = frame.loc[:, feature_columns].copy()
    for feature in numeric_features:
        original_values = features[feature]
        converted_values = pd.to_numeric(original_values, errors="coerce")
        invalid_values = (original_values.notna() & converted_values.isna()) | (
            converted_values.notna() & ~np.isfinite(converted_values)
        )
        if invalid_values.any():
            examples = original_values[invalid_values].astype(str).head(3).tolist()
            raise ValueError(
                f"{feature} contains non-numeric values, for example {examples}."
            )
        features[feature] = converted_values
    return features


def show_predictions(features, identifiers=None):
    positive_class_index = list(model.classes_).index(1)
    probabilities = model.predict_proba(features)[:, positive_class_index]
    predictions = (probabilities >= decision_threshold).astype(int)
    results = pd.DataFrame(
        {
            "PredictedChurn": np.where(predictions == 1, "Yes", "No"),
            "ChurnProbability": probabilities,
        }
    )
    if identifiers is not None:
        results.insert(0, "CustomerID", identifiers.to_numpy())
    return results


st.header("Predict a customer")
st.caption(
    "Enter the 56 Cell2Cell model features. Numeric defaults are training-set "
    "medians; categorical options come from the fitted encoder."
)

with st.form("cell2cell_customer_form"):
    form_columns = st.columns(3)
    manual_values = {}
    for position, feature in enumerate(feature_columns):
        with form_columns[position % len(form_columns)]:
            if feature in categorical_options:
                options = categorical_options[feature]
                manual_values[feature] = st.selectbox(
                    feature,
                    options=options,
                    format_func=str,
                    key=f"manual_{feature}",
                )
            else:
                default = float(numeric_defaults[feature])
                if not np.isfinite(default):
                    default = 0.0
                manual_values[feature] = st.number_input(
                    feature,
                    value=default,
                    step=1.0,
                    format="%.2f",
                    key=f"manual_{feature}",
                )
    submitted = st.form_submit_button("Predict churn")

if submitted:
    customer_features = pd.DataFrame(
        [manual_values], columns=feature_columns
    )
    prediction_result = show_predictions(customer_features)
    probability = float(prediction_result.loc[0, "ChurnProbability"])
    st.subheader("Prediction result")
    st.metric("Churn probability", f"{probability:.1%}")
    if prediction_result.loc[0, "PredictedChurn"] == "Yes":
        st.warning("The model predicts that this customer will churn.")
    else:
        st.success("The model predicts that this customer will stay.")

    if probability >= 0.70:
        risk = "High"
    elif probability >= 0.40:
        risk = "Medium"
    else:
        risk = "Low"
    st.info(
        f"Illustrative risk band: **{risk}**. These probability bands are "
        "heuristics, not business-validated intervention thresholds."
    )

st.header("Predict from a CSV")
st.write(
    "Upload a Cell2Cell-shaped CSV containing all model features. An optional "
    "CustomerID is kept with the results. A Churn column is accepted only when "
    "all its values are missing, as in the supplied holdout file."
)
uploaded_file = st.file_uploader("Cell2Cell CSV", type=["csv"])
if uploaded_file is not None:
    try:
        uploaded_data = pd.read_csv(uploaded_file, low_memory=False)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        st.error(f"Could not read the uploaded CSV: {exc}")
    else:
        try:
            upload_features = validate_features(uploaded_data)
        except ValueError as exc:
            st.error(str(exc))
        else:
            upload_ids = (
                uploaded_data["CustomerID"]
                if "CustomerID" in uploaded_data.columns
                else None
            )
            results = show_predictions(upload_features, upload_ids)
            st.success(f"Predictions generated for {len(results):,} rows.")
            st.dataframe(results, width="stretch")
            st.download_button(
                "Download predictions",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name="cell2cell_churn_predictions.csv",
                mime="text/csv",
            )
