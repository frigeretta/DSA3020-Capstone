import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ========================= CONFIG =========================
st.set_page_config(
    page_title="LoanGuard Pro • Instant Approval Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for banking elegance
st.markdown("""
<style>
    .main {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem 0;}
    .block-container {max-width: 1200px; padding-top: 2rem; padding-bottom: 2rem;}
    .header-title {font-size: 3rem !important; font-weight: 800; color: white; text-align: center; text-shadow: 0 4px 10px rgba(0,0,0,0.3);}
    .header-subtitle {color: #e0e7ff; text-align: center; font-size: 1.3rem; margin-bottom: 2rem;}
    .stButton>button {background: #4CAF50; color: white; font-weight: bold; height: 3em; border-radius: 12px;}
    .result-card {padding: 2rem; border-radius: 16px; text-align: center; box-shadow: 0 10px 30px rgba(0,0,0,0.2);}
    .approved {background: linear-gradient(135deg, #56ab2f, #a8e6cf) !important; color: white;}
    .declined {background: linear-gradient(135deg, #ff512f, #dd2476) !important; color: white;}
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
        st.error(f"Model files not found: {e}")
        st.stop()

models, scaler = load_model_assets()

# ========================= HEADER =========================
st.markdown("<h1 class='header-title'>🏦 LoanGuard Pro</h1>", unsafe_allow_html=True)
st.markdown("<p class='header-subtitle'>Multi-Model • Accurate • AI-Powered Loan Eligibility Prediction</p>", unsafe_allow_html=True)

# ========================= SIDEBAR INFO =========================
with st.expander("ℹ️ Model Information", expanded=False):
    st.info("""
    **Three Independent Models:**
    - **Logistic Regression**: Fast, interpretable, good baseline
    - **Decision Tree**: Captures non-linear patterns
    - **Gradient Boosting**: Ensemble method, high accuracy
    
    Compare predictions across models for robust decision-making.
    """)

st.markdown("---")

# ========================= INPUT FORM =========================
with st.form("loan_application_form", clear_on_submit=False):
    st.subheader("📋 Applicant Information")

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
        credit_history = st.selectbox("Credit History Status", ["Good", "Bad"], index=0)

    submitted = st.form_submit_button("🔍 Predict Eligibility", use_container_width=True)

# ========================= ENCODING FUNCTION =========================
def encode_input(data_dict):
    mapping = {
        'Gender': {'Male': 1, 'Female': 0},
        'Married': {'Yes': 1, 'No': 0},
        'Dependents': {'0': 0, '1': 1, '2': 2, '3+': 3},
        'Education': {'Graduate': 0, 'Not Graduate': 1},
        'Self_Employed': {'Yes': 1, 'No': 0},
        'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2},
        'Credit_History': {'Good': 1.0, 'Bad': 0.0},
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
        mapping['Credit_History'][data_dict['credit_history']],
        mapping['Property_Area'][data_dict['property_area']]
    ]
    return np.array([features])

# ========================= REAL-TIME PREDICTION =========================
if submitted:
    with st.spinner("🔄 Analyzing application across all models..."):
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
            'credit_history': credit_history
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
    st.subheader("📊 Predictions from All Models")
    
    cols = st.columns(3)
    for idx, (model_name, result) in enumerate(predictions.items()):
        with cols[idx]:
            status = "✅ APPROVED" if result['prediction'] == 1 else "❌ DECLINED"
            card_class = "approved" if result['prediction'] == 1 else "declined"
            
            st.markdown(f"""
            <div class="result-card {card_class}">
                <h3>{model_name}</h3>
                <h2 style="margin: 1rem 0;">{status}</h2>
                <p style="font-size: 2.5rem; font-weight: bold; margin: 0.5rem 0;">{result['probability']:.1%}</p>
                <p style="font-size: 0.9rem; opacity: 0.9;">Approval Probability</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ========================= SUMMARY STATISTICS =========================
    avg_approval = np.mean([p['probability'] for p in predictions.values()])
    consensus = sum([p['prediction'] for p in predictions.values()])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Average Approval Rate", f"{avg_approval:.1%}")
    with col2:
        st.metric("Models Approving", f"{consensus}/3")
    with col3:
        consensus_decision = "🟢 MAJORITY APPROVED" if consensus >= 2 else "🔴 MAJORITY DECLINED"
        st.markdown(f"### {consensus_decision}")
    
    st.markdown("---")
    st.caption("🏦 LoanGuard Pro • Multi-Model Ensemble Decision Support • Built by Gichangi")