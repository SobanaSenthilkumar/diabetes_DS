import streamlit as st
import pandas as pd
import joblib
import os
from PIL import Image

# Set up Streamlit page
st.set_page_config(page_title="Diabetes Risk Prediction")
st.title("🩺 Diabetes Risk Prediction Dashboard")
st.markdown("This app allows both bulk and individual diabetes risk prediction using machine learning models.")

# Expected features
expected_columns = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]

# Set project paths
project_root = os.path.dirname(os.path.dirname(__file__))
model_dir = os.path.join(project_root, "src", "models")
shap_path = os.path.join(project_root, "src", "reports", "shap_summary.png")

# --- Bulk Prediction from CSV ---
st.markdown("## 📂 Upload CSV for Batch Prediction")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file:
    input_data = pd.read_csv(uploaded_file)
    st.subheader("🗂️ Uploaded Data Preview")
    st.dataframe(input_data.head())

    # Drop 'Outcome' if present
    if "Outcome" in input_data.columns:
        input_data = input_data.drop(columns=["Outcome"])

    # Validate input
    missing_cols = set(expected_columns) - set(input_data.columns)
    if missing_cols:
        st.error(f"❌ Missing columns: {', '.join(missing_cols)}")
    else:
        input_data = input_data[expected_columns]

        model_names = ["logistic", "random_forest", "xgboost", "svm"]
        results = {}
        predictions_df = pd.DataFrame()

        for name in model_names:
            model_path = os.path.join(model_dir, f"{name}.pkl")
            try:
                model = joblib.load(model_path)
                preds = model.predict_proba(input_data)[:, 1]
                avg_score = round(preds.mean(), 4)
                results[name] = avg_score
                predictions_df[name] = preds
            except FileNotFoundError:
                results[name] = None
                st.warning(f"⚠️ Model not found: {name}.pkl")

        valid_results = {k: v for k, v in results.items() if v is not None}
        if valid_results:
            sorted_models = sorted(valid_results.items(), key=lambda x: x[1], reverse=True)

            best = sorted_models[0]
            worst = sorted_models[-1]
            middle = sorted_models[len(sorted_models)//2] if len(sorted_models) > 2 else None

            st.subheader("🧠 Model Ranking (Based on Avg Risk)")
            st.markdown(f"🏆 **Best Model:** `{best[0]}` (Avg Risk: **{best[1]}**)")
            if middle:
                st.markdown(f"⚖️ **Average Model:** `{middle[0]}` (Avg Risk: **{middle[1]}**)")
            st.markdown(f"👎 **Worst Model:** `{worst[0]}` (Avg Risk: **{worst[1]}**)")

            st.subheader("🔬 Risk Scores by Model")
            st.dataframe(predictions_df)

            st.markdown("---")
            st.subheader("🔍 SHAP Feature Importance Summary")
            if os.path.exists(shap_path):
                image = Image.open(shap_path)
                st.image(image, caption="SHAP Summary Plot", use_column_width=True)
            else:
                st.warning("⚠️ SHAP summary plot not found.")
        else:
            st.error("❌ No valid models found for prediction.")

# --- Single Patient Input Section ---
st.markdown("---")
st.markdown("## 🎯 Predict Diabetes Risk for One Patient")

with st.form(key="user_form"):
    col1, col2 = st.columns(2)

    with col1:
        Pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
        Glucose = st.number_input("Glucose", min_value=0, max_value=200, value=100)
        BloodPressure = st.number_input("Blood Pressure", min_value=0, max_value=150, value=70)
        SkinThickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)

    with col2:
        Insulin = st.number_input("Insulin", min_value=0, max_value=900, value=85)
        BMI = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
        DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
        Age = st.number_input("Age", min_value=1, max_value=120, value=30)

    selected_model = st.selectbox("Select Model for Prediction", ["logistic", "random_forest", "xgboost", "svm"])
    submit_button = st.form_submit_button(label="🔍 Predict")

if submit_button:
    # Create input DataFrame
    user_input_df = pd.DataFrame([[
        Pregnancies, Glucose, BloodPressure, SkinThickness,
        Insulin, BMI, DiabetesPedigreeFunction, Age
    ]], columns=expected_columns)

    model_path = os.path.join(model_dir, f"{selected_model}.pkl")
    try:
        model = joblib.load(model_path)
        risk_score = model.predict_proba(user_input_df)[0][1]
        st.success(f"🩺 Predicted Risk Score: **{risk_score:.3f}**")

        if risk_score >= 0.5:
            st.error("🔴 High Risk: Likely to have Diabetes")
        else:
            st.success("🟢 Low Risk: Unlikely to have Diabetes")

    except Exception as e:
        st.error(f"❌ Could not load model or predict: {e}")
