"""
streamlit_app.py

Simple Streamlit demo app for predicting GDP per capita.

Run with:
streamlit run app/streamlit_app.py
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_FILE = BASE_DIR / "models" / "random_forest_gdp_model.pkl"
FEATURES_FILE = BASE_DIR / "models" / "regression_features.pkl"


FEATURE_LABELS = {
    "annual_income": "Annual Income",
    "corruption_index": "Corruption Index",
    "cost_index": "Cost Index",
    "monthly_income": "Monthly Income",
    "purchasing_power_index": "Purchasing Power Index",
    "unemployment_rate": "Unemployment Rate",
    "tourists_in_millions": "Tourists in Millions",
    "receipts_in_billions": "Tourism Receipts in Billions",
    "percentage_of_gdp": "Tourism Percentage of GDP",
    "cluster": "Economic Cluster",
}


DEFAULT_VALUES = {
    "annual_income": 10000.0,
    "corruption_index": 60.0,
    "cost_index": 60.0,
    "monthly_income": 1000.0,
    "purchasing_power_index": 50.0,
    "unemployment_rate": 5.0,
    "tourists_in_millions": 10.0,
    "receipts_in_billions": 5.0,
    "percentage_of_gdp": 2.0,
}


def load_model_and_features():
    """Load the trained model and the feature order used during training."""
    if not MODEL_FILE.exists():
        st.error(f"Model file not found: {MODEL_FILE}")
        st.stop()

    if not FEATURES_FILE.exists():
        st.error(f"Features file not found: {FEATURES_FILE}")
        st.stop()

    model = joblib.load(MODEL_FILE)
    features = joblib.load(FEATURES_FILE)
    return model, features


def create_input_form(features):
    """Create Streamlit inputs for each feature and return user values."""
    user_inputs = {}

    with st.form("prediction_form"):
        st.subheader("Enter Economic Indicators")

        for feature in features:
            label = FEATURE_LABELS.get(feature, feature.replace("_", " ").title())

            if feature == "cluster":
                user_inputs[feature] = st.selectbox(
                    label,
                    options=[0, 1, 2],
                    index=0,
                )
            else:
                user_inputs[feature] = st.number_input(
                    label,
                    min_value=0.0,
                    value=DEFAULT_VALUES.get(feature, 0.0),
                    step=1.0,
                )

        submitted = st.form_submit_button("Predict")

    return submitted, user_inputs


def predict_gdp_per_capita(model, features, user_inputs):
    """Convert user input to a DataFrame and predict GDP per capita."""
    input_df = pd.DataFrame([user_inputs])

    # Keep the exact column order used when the model was trained.
    input_df = input_df[features]

    prediction = model.predict(input_df)[0]
    return prediction


def main():
    st.set_page_config(
        page_title="GDP per Capita Prediction",
        page_icon="📊",
        layout="centered",
    )

    st.title("GDP per Capita Prediction App")
    st.write(
        "This demo uses a Random Forest model to estimate GDP per capita from "
        "country-level economic, tourism, unemployment, and corruption indicators."
    )

    model, features = load_model_and_features()
    submitted, user_inputs = create_input_form(features)

    if submitted:
        prediction = predict_gdp_per_capita(model, features, user_inputs)

        st.success(
            f"Predicted GDP per Capita: ${prediction:,.2f}"
        )

        with st.expander("Input data used for prediction"):
            st.dataframe(pd.DataFrame([user_inputs])[features])

    st.divider()
    st.subheader("Important Notes")
    st.write(
        "- This is a learning project demo for Data Analysis and Machine Learning.\n"
        "- This app should not be used for real economic forecasting or policy decisions.\n"
        "- The current Random Forest model has a low R2 Score because the dataset is small, "
        "merged from multiple sources, and contains outliers."
    )


if __name__ == "__main__":
    main()
