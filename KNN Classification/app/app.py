import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime
from pathlib import Path

# Try to import BytesIO for Excel support
try:
    from io import BytesIO
    EXCEL_SUPPORT = True
except Exception:
    EXCEL_SUPPORT = False

# Page configuration
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    layout="wide",
    page_icon="🫀",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #c2410c;
        text-align: center;
        margin-bottom: 0;
    }
    .sub-header {
        text-align: center;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #c2410c 0%, #0ea5e9 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin: 2rem 0;
    }
    .prediction-value {
        font-size: 3rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load pre-trained model
@st.cache_resource
def load_model():
    model_path = Path(__file__).resolve().parents[1] / "models" / "knn_pipeline.joblib"
    if not model_path.exists():
        return None, None, False, False

    try:
        model = joblib.load(model_path)
    except Exception:
        return None, None, False, False

    if hasattr(model, 'named_steps'):
        return model, None, True, True

    return model, None, True, False

model, preprocessor, model_loaded, is_pipeline = load_model()

# Header
st.markdown('<p class="main-header">🫀 Heart Disease Risk Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Predict heart disease risk using clinical measurements and KNN classification</p>', unsafe_allow_html=True)

# Sidebar - Information
with st.sidebar:
    st.markdown("<h1 style='text-align: center; font-size: 100px;'>🫀</h1>", unsafe_allow_html=True)
    st.title("About This Tool")
    st.markdown("""
    This tool uses **K-Nearest Neighbors (KNN)** to estimate heart disease risk based on patient metrics.

    ### How It Works:
    1. Enter patient details
    2. Click 'Predict Risk'
    3. Get instant classification

    ### Model Highlights:
    - **Algorithm:** KNN Classifier
    - **Features Used:** 13
    - **Dataset:** UCI Heart Disease
    """)

    st.markdown("---")
    st.info("Tip: Age, chest pain type, and max heart rate are strong indicators.")

    st.markdown("---")
    st.markdown("### Model Information")
    st.markdown("""
    **Algorithm:** KNN Classifier  
    **Features Used:** 13  
    **Last Updated:** """ + datetime.now().strftime("%B %Y"))

# Check if model is loaded
if not model_loaded:
    st.error("Model could not be loaded. Please re-train and save knn_pipeline.joblib.")
    st.stop()

# Main content area
tab1, tab2, tab3 = st.tabs(["🔮 Single Prediction", "📊 Batch Predictions", "📈 Analytics"])

# Tab 1: Single Prediction
with tab1:
    st.header("Enter Patient Details")

    training_ranges = {
        'age': (29, 77),
        'trestbps': (94, 200),
        'chol': (126, 564),
        'thalach': (71, 202),
        'oldpeak': (0.0, 6.2),
        'ca': (0, 4)
    }

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Vitals & Labs")
        age = st.number_input("Age (years)", min_value=18, max_value=100, value=54, step=1)
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=220, value=130, step=1)
        chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=80, max_value=600, value=240, step=1)
        thalach = st.number_input("Max Heart Rate (bpm)", min_value=60, max_value=220, value=150, step=1)
        oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=6.5, value=1.0, step=0.1)

    with col2:
        st.subheader("Clinical Signals")
        sex = st.selectbox("Sex", options=[("Female", 0), ("Male", 1)], format_func=lambda x: x[0])[1]
        cp = st.selectbox(
            "Chest Pain Type",
            options=[("Typical Angina", 0), ("Atypical Angina", 1), ("Non-anginal Pain", 2), ("Asymptomatic", 3)],
            format_func=lambda x: x[0]
        )[1]
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
        restecg = st.selectbox(
            "Resting ECG",
            options=[("Normal", 0), ("ST-T Abnormality", 1), ("LV Hypertrophy", 2)],
            format_func=lambda x: x[0]
        )[1]
        exang = st.selectbox("Exercise-induced Angina", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
        slope = st.selectbox(
            "Slope of Peak ST",
            options=[("Upsloping", 0), ("Flat", 1), ("Downsloping", 2)],
            format_func=lambda x: x[0]
        )[1]
        ca = st.slider("Major Vessels (0-4)", min_value=0, max_value=4, value=0)
        thal = st.selectbox(
            "Thalassemia",
            options=[("Unknown", 0), ("Normal", 1), ("Fixed Defect", 2), ("Reversible Defect", 3)],
            format_func=lambda x: x[0]
        )[1]

    warnings = []
    for feature, (low, high) in training_ranges.items():
        value = locals()[feature]
        if not (low <= value <= high):
            warnings.append(f"{feature} ({value}) is outside training range ({low}-{high})")

    if warnings:
        st.warning("Some values are outside the training range. Predictions may be less accurate.")
        for warning in warnings:
            st.caption(warning)

    st.markdown("### Current Input Summary")
    input_df_display = pd.DataFrame({
        'Feature': ['Age', 'Sex', 'Chest Pain', 'Resting BP', 'Cholesterol', 'Fasting Sugar', 'Rest ECG', 'Max HR', 'Exercise Angina', 'Oldpeak', 'ST Slope', 'Major Vessels', 'Thal'],
        'Value': [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
    })
    st.dataframe(input_df_display, hide_index=True, use_container_width=True)

    st.markdown("---")
    if st.button("Predict Risk", type="primary", use_container_width=True):
        input_data = pd.DataFrame({
            'age': [age],
            'sex': [sex],
            'cp': [cp],
            'trestbps': [trestbps],
            'chol': [chol],
            'fbs': [fbs],
            'restecg': [restecg],
            'thalach': [thalach],
            'exang': [exang],
            'oldpeak': [oldpeak],
            'slope': [slope],
            'ca': [ca],
            'thal': [thal]
        })

        try:
            prediction = model.predict(input_data)[0]
            proba = model.predict_proba(input_data) if hasattr(model, 'predict_proba') else None
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            st.stop()

        risk_label = "Higher Risk" if prediction == 1 else "Lower Risk"
        confidence_pct = None
        if proba is not None:
            class_index = int(np.where(model.classes_ == 1)[0][0]) if hasattr(model, "classes_") else 1
            confidence_pct = proba[0][class_index] * 100

        st.markdown("### Prediction Result")
        if confidence_pct is not None:
            st.markdown(f"""
            <div class="prediction-box">
                <h2>Risk Classification</h2>
                <div class="prediction-value">{risk_label}</div>
                <p>Estimated probability: <strong>{confidence_pct:.1f}%</strong></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="prediction-box">
                <h2>Risk Classification</h2>
                <div class="prediction-value">{risk_label}</div>
                <p>Probability unavailable for this model configuration.</p>
            </div>
            """, unsafe_allow_html=True)

# Tab 2: Batch Predictions
with tab2:
    st.header("Batch Predictions")
    st.markdown("Upload a file with multiple patients to get predictions for all at once.")

    with st.expander("See Supported File Formats & Examples"):
        st.markdown("""
        **CSV Example**
        ```
        age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal
        54,1,2,130,240,0,1,150,0,1.0,1,0,2
        63,0,3,145,233,1,0,150,0,2.3,0,0,1
        ```
        """)

    template_df = pd.DataFrame({
        'age': [54, 63, 45],
        'sex': [1, 0, 1],
        'cp': [2, 3, 1],
        'trestbps': [130, 145, 120],
        'chol': [240, 233, 210],
        'fbs': [0, 1, 0],
        'restecg': [1, 0, 1],
        'thalach': [150, 150, 170],
        'exang': [0, 0, 1],
        'oldpeak': [1.0, 2.3, 0.5],
        'slope': [1, 0, 2],
        'ca': [0, 0, 1],
        'thal': [2, 1, 3]
    })

    with st.expander("Preview Template Data"):
        st.dataframe(template_df, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Download Template")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        csv = template_df.to_csv(index=False)
        st.download_button(label="CSV", data=csv, file_name="heart_template.csv", mime="text/csv")

    with col2:
        txt = template_df.to_csv(index=False, sep='\t')
        st.download_button(label="TXT", data=txt, file_name="heart_template.txt", mime="text/plain")

    with col3:
        if EXCEL_SUPPORT:
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                template_df.to_excel(writer, index=False, sheet_name='patients')
            st.download_button(
                label="Excel",
                data=output.getvalue(),
                file_name="heart_template.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.button("Excel", disabled=True, help="Install openpyxl")

    with col4:
        json_data = template_df.to_json(orient='records', indent=2)
        st.download_button(label="JSON", data=json_data, file_name="heart_template.json", mime="application/json")

    st.markdown("### Upload Your Data")
    st.info("Supported formats: CSV, TXT, TSV, Excel, JSON")

    file_types = ['csv', 'txt', 'tsv', 'json']
    if EXCEL_SUPPORT:
        file_types.extend(['xlsx', 'xls'])

    uploaded_file = st.file_uploader("Choose a file", type=file_types)

    if uploaded_file is not None:
        try:
            file_name = uploaded_file.name.lower()
            if file_name.endswith('.csv'):
                batch_df = pd.read_csv(uploaded_file)
            elif file_name.endswith('.txt'):
                batch_df = pd.read_csv(uploaded_file, sep=None, engine='python')
            elif file_name.endswith('.tsv'):
                batch_df = pd.read_csv(uploaded_file, sep='\t')
            elif file_name.endswith(('.xlsx', '.xls')):
                if EXCEL_SUPPORT:
                    batch_df = pd.read_excel(uploaded_file)
                else:
                    st.error("Excel support not available. Install openpyxl")
                    st.stop()
            elif file_name.endswith('.json'):
                batch_df = pd.read_json(uploaded_file)
            else:
                st.error("Unsupported file format")
                st.stop()

            st.success(f"File uploaded successfully. Found {len(batch_df)} rows.")
            st.dataframe(batch_df.head(), use_container_width=True)

            required_cols = list(template_df.columns)
            missing_cols = [col for col in required_cols if col not in batch_df.columns]

            if missing_cols:
                st.error(f"Missing required columns: {', '.join(missing_cols)}")
                st.stop()

            if st.button("Generate Predictions", type="primary"):
                features = batch_df[required_cols]
                preds = model.predict(features)
                probas = model.predict_proba(features) if hasattr(model, 'predict_proba') else None

                result_df = batch_df.copy()
                result_df['predicted_target'] = preds
                if probas is not None:
                    class_index = int(np.where(model.classes_ == 1)[0][0]) if hasattr(model, "classes_") else 1
                    result_df['risk_probability'] = probas[:, class_index].round(3)

                st.markdown("### Prediction Results")
                st.dataframe(result_df, use_container_width=True)

                result_csv = result_df.to_csv(index=False)
                st.download_button(
                    label="Download Predictions (CSV)",
                    data=result_csv,
                    file_name=f"heart_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# Tab 3: Analytics
with tab3:
    st.header("Model Analytics & Insights")

    st.info("These metrics reflect model performance during training and testing.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Model Accuracy", "~95%")
    col2.metric("Precision", "~88%")
    col3.metric("Recall", "~95%")

    st.markdown("---")

    st.markdown("### Feature Overview")
    feature_info = pd.DataFrame({
        'Feature': ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'],
        'Description': ['Age', 'Sex', 'Chest pain type', 'Resting blood pressure', 'Serum cholesterol', 'Fasting blood sugar', 'Resting ECG', 'Max heart rate', 'Exercise angina', 'ST depression', 'ST slope', 'Major vessels', 'Thalassemia']
    })
    st.dataframe(feature_info, hide_index=True, use_container_width=True)

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**Clinical Style Inputs**")
    st.caption("Aligned with UCI heart dataset")
with col2:
    st.markdown("**Fast KNN Predictions**")
    st.caption("Instant classification results")
with col3:
    st.markdown("**Responsible Use**")
    st.caption("Not a replacement for medical advice")

st.markdown("<br><center><small>Developed by Meet Bataviya | Powered by Machine Learning</small></center>", unsafe_allow_html=True)
