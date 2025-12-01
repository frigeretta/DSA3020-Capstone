import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ========================= CONFIG =========================
st.set_page_config(
    page_title="LoanGuard Pro • Instant Approval Predictor",
    page_icon="🏦",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for banking elegance
st.markdown("""
<style>
    .main {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem 0;}
    .block-container {max-width: 900px; padding-top: 2rem; padding-bottom: 2rem;}
    .header-title {font-size: 3rem !important; font-weight: 800; color: white; text-align: center; text-shadow: 0 4px 10px rgba(0,0,0,0.3);}
    .header-subtitle {color: #e0e7ff; text-align: center; font-size: 1.3rem; margin-bottom: 2rem;}
    .credit-critical {font-size: 1.4rem; font-weight: bold; padding: 1rem; border-radius: 12px; text-align: center;}
    .stButton>button {background: #4CAF50; color: white; font-weight: bold; height: 3em; border-radius: 12px;}
    .result-card {padding: 2rem; border-radius: 16px; text-align: center; box-shadow: 0 10px 30px rgba(0,0,0,0.2);}
    .gauge {font-size: 4rem; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# ========================= LOAD MODEL =========================
@st.cache_resource
def load_model_assets():
    try:
        app_dir = Path(__file__).parent
        models = {
            'Logistic Regression': joblib.load(app_dir / "logistic_regression.pkl"),
            'Decision Tree': joblib.load(app_dir / "decision_tree.pkl"),
            'Gradient Boosting': joblib.load(app_dir / "gradient_boosting.pkl")
        }
        scaler = joblib.load(app_dir / "scaler.pkl")
        return models, scaler
    except FileNotFoundError as e:
        st.error(f"Model files not found. Please ensure all model files are in the loandata directory. Error: {e}")
        st.stop()

models, scaler = load_model_assets()

# Feature order (MUST match training!)
FEATURE_COLUMNS = [
    'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
    'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
    'Loan_Amount_Term', 'Credit_History', 'Property_Area'
]

# ========================= HEADER =========================
st.markdown("<h1 class='header-title'>🏦 LoanGuard Pro</h1>", unsafe_allow_html=True)
st.markdown("<p class='header-subtitle'>Instant • Accurate • AI-Powered Loan Eligibility Prediction</p>", unsafe_allow_html=True)

# ========================= SIDEBAR INFO =========================
with st.expander("Model Insights", expanded=False):
    st.success("**Models:** Logistic Regression • Decision Tree • Gradient Boosting")
    st.info("All models are ensembled to provide diverse predictions for robustness.")
    st.markdown("""
    ### Feature Importance:
    - **Credit History** - Primary factor
    - **Married Status** - Secondary factor
    - **Education Level** - Tertiary factor
    - All other features considered for comprehensive analysis
    """)

st.markdown("---")

# ========================= INPUT FORM =========================
with st.form("loan_application_form", clear_on_submit=False):
    st.subheader("Applicant Information")

    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"], index=0)
        married = st.selectbox("Marital Status", ["Yes", "No"], index=0)
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"], index=0)
        education = st.selectbox("Education", ["Graduate", "Not Graduate"], index=0)
        self_employed = st.selectbox("Self Employed", ["No", "Yes"], index=0)

    with col2:
        applicant_income = st.number_input("Applicant Monthly Income ($)", min_value=0, value=5000, step=500)
        coapplicant_income = st.number_input("Co-Applicant Income ($)", min_value=0.0, value=0.0, step=100.0)
        loan_amount = st.number_input("Loan Amount Requested ($)", min_value=1000.0, value=150000.0, step=5000.0)
        loan_term = st.selectbox("Loan Term (Months)", [360, 240, 180, 120, 84, 60, 36], index=0)
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"], index=1)

    submitted = st.form_submit_button("Predict Eligibility Now", use_container_width=True)

# ========================= ENCODING FUNCTION =========================
def encode_input(data_dict):
    mapping = {
        'Gender': {'Male': 1, 'Female': 0},
        'Married': {'Yes': 1, 'No': 0},
        'Dependents': {'0': 0, '1': 1, '2': 2, '3+': 3},
        'Education': {'Graduate': 0, 'Not Graduate': 1},
        'Self_Employed': {'Yes': 1, 'No': 0},
        'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2},
    }

    features = [
        mapping['Gender'][data_dict['gender']],
        mapping['Married'][data_dict['married']],
        mapping['Dependents'][data_dict['dependents']],
        mapping['Education'][data_dict['education']],
        mapping['Self_Employed'][data_dict['self_employed']],
        data_dict['applicant_income'],
        data_dict['coapplicant_income'],
        data_dict['loan_amount'],
        data_dict['loan_term'],
        1.0,  # Credit history placeholder (normalized)
        mapping['Property_Area'][data_dict['property_area']]
    ]
    return np.array([features])

# ========================= REAL-TIME PREDICTION =========================
if submitted:
    with st.spinner("Analyzing application..."):
        input_data = {
            'gender': gender,
            'married': married,
            'dependents': dependents,
            'education': education,
            'self_employed': self_employed,
            'applicant_income': applicant_income,
            'coapplicant_income': coapplicant_income,
            'loan_amount': loan_amount,
            'loan_term': loan_term,
            'property_area': property_area,
        }

        X = encode_input(input_data)
        X_scaled = scaler.transform(X)
        
        # Get predictions from all models
        predictions = {}
        for model_name, model in models.items():
            pred = model.predict(X_scaled)[0]
            prob = model.predict_proba(X_scaled)[0][1]
            predictions[model_name] = {
                'prediction': pred,
                'probability': prob
            }

    st.markdown("---")

    # ========================= RESULTS DISPLAY =========================
    st.subheader("Predictions from All Models")
    
    cols = st.columns(3)
    for idx, (model_name, result) in enumerate(predictions.items()):
        with cols[idx]:
            status = "✅ APPROVED" if result['prediction'] == 1 else "❌ DECLINED"
            color = "#56ab2f" if result['prediction'] == 1 else "#ff512f"
            
            st.markdown(f"""
            <div class="result-card" style="background: linear-gradient(135deg, {color}, {'#a8e6cf' if result['prediction'] == 1 else '#dd2476'}); color: white; padding: 2rem; border-radius: 16px; text-align: center;">
                <h3>{model_name}</h3>
                <h2>{status}</h2>
                <p style="font-size: 2rem; font-weight: bold;">{result['probability']:.1%}</p>
                <p style="font-size: 0.9rem;">Approval Probability</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.caption("Powered by Synthahub  • Built by Gichangi")