import streamlit as st
import numpy as np
import joblib
import os

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #f6f9fc 0%, #eef2f7 100%);
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    .hero {
        background: linear-gradient(135deg, #2563eb 0%, #0ea5e9 100%);
        padding: 2rem 2.5rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(37, 99, 235, 0.25);
    }
    .hero h1 { font-size: 2.1rem; margin: 0; font-weight: 800; }
    .hero p { margin-top: 0.4rem; font-size: 1.02rem; opacity: 0.92; }

    .section-card {
        background: white;
        padding: 1.4rem 1.6rem 0.6rem 1.6rem;
        border-radius: 16px;
        border: 1px solid #e5e9f2;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04);
        margin-bottom: 1.3rem;
    }
    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.9rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    div[data-baseweb="select"] > div, .stNumberInput input {
        border-radius: 10px !important;
    }
    label { font-weight: 600 !important; color: #334155 !important; font-size: 0.9rem !important; }

    div.stButton > button {
        background: linear-gradient(135deg, #2563eb 0%, #0ea5e9 100%);
        color: white;
        font-weight: 700;
        font-size: 1.05rem;
        padding: 0.8rem 0;
        border-radius: 12px;
        border: none;
        box-shadow: 0 8px 20px rgba(37, 99, 235, 0.3);
        transition: transform 0.15s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 24px rgba(37, 99, 235, 0.4);
        color: white;
    }

    .result-box {
        border-radius: 16px;
        padding: 1.5rem 1.8rem;
        margin-top: 1rem;
        font-size: 1.15rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL
# =========================================================
MODEL_PATH = os.path.join(os.path.dirname(__file__), "random_forest_model.pkl")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# Exact feature order the model was trained on (from model.feature_names_in_)
FEATURE_ORDER = [
    "age", "gender", "education_level", "income_level", "smoking_status",
    "alcohol_consumption_per_week", "physical_activity_minutes_per_week",
    "diet_score", "sleep_hours_per_day", "screen_time_hours_per_day",
    "family_history_diabetes", "hypertension_history", "cardiovascular_history",
    "bmi", "waist_to_hip_ratio", "systolic_bp", "diastolic_bp", "heart_rate",
    "cholesterol_total", "hdl_cholesterol", "ldl_cholesterol", "triglycerides",
    "glucose_fasting", "glucose_postprandial", "insulin_level", "hba1c",
    "diabetes_risk_score",
    "ethnicity_Black", "ethnicity_Hispanic", "ethnicity_Other", "ethnicity_White",
    "diabetes_stage_No Diabetes", "diabetes_stage_Pre-Diabetes",
    "diabetes_stage_Type 1", "diabetes_stage_Type 2",
    "employment_status_Retired", "employment_status_Student", "employment_status_Unemployed"
]

# NOTE: The model was trained with these label-encodings for non-one-hot
# categorical columns. If your training script used a different LabelEncoder
# mapping, update the dictionaries below to match it exactly.
BINARY_MAP = {"No": 0, "Yes": 1}
GENDER_MAP = {"Female": 0, "Male": 1}
SMOKING_MAP = {"Current": 0, "Former": 1, "Never": 2}
EDUCATION_MAP = {"Graduate": 0, "Highschool": 1, "No formal education": 2, "Postgraduate": 3}
INCOME_MAP = {"High": 0, "Low": 1, "Lower-Middle": 2, "Middle": 3, "Upper-Middle": 4}

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("## 🩺 About")
    st.markdown(
        "This tool estimates **diabetes risk** using a trained "
        "**Random Forest** model based on personal, lifestyle, medical, "
        "and clinical data.\n\nFill in the patient details and click "
        "**Predict Diabetes Risk**."
    )
    st.markdown("---")
    st.markdown("### 📋 Sections")
    st.markdown(
        "- 👤 Personal Information\n"
        "- 🏃 Lifestyle Information\n"
        "- 🏥 Medical History\n"
        "- 🧪 Clinical Measurements\n"
        "- 🩸 Blood Test Results\n"
        "- 🧬 Glucose & Insulin"
    )
    st.markdown("---")
    st.caption("⚠️ For informational purposes only. Not a substitute for professional medical advice.")

# =========================================================
# HERO HEADER
# =========================================================
st.markdown("""
<div class="hero">
    <h1>🩺 Diabetes Risk Prediction</h1>
    <p>Enter the patient's details below to estimate their diabetes risk using our trained model.</p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# PERSONAL INFORMATION
# =========================================================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">👤 Personal Information</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=40)
    gender = st.selectbox("Gender", list(GENDER_MAP.keys()))
with col2:
    education_level = st.selectbox("Education Level", list(EDUCATION_MAP.keys()))
    income_level = st.selectbox("Income Level", list(INCOME_MAP.keys()))
with col3:
    smoking_status = st.selectbox("Smoking Status", list(SMOKING_MAP.keys()))
    employment_status = st.selectbox("Employment Status", ["Employed", "Retired", "Unemployed", "Student"])
st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# LIFESTYLE INFORMATION
# =========================================================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🏃 Lifestyle Information</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    alcohol = st.number_input("Alcohol Consumption / Week", min_value=0.0, value=2.0)
    physical_activity = st.number_input("Physical Activity (min/week)", min_value=0.0, value=150.0)
with col2:
    diet_score = st.number_input("Diet Score", min_value=0.0, value=5.0)
    sleep_hours = st.number_input("Sleep Hours / Day", min_value=0.0, max_value=24.0, value=7.0)
with col3:
    screen_time = st.number_input("Screen Time (hours/day)", min_value=0.0, value=4.0)
    ethnicity = st.selectbox("Ethnicity", ["Black", "Hispanic", "Other", "White"])
st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# MEDICAL HISTORY
# =========================================================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🏥 Medical History</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    family_history = st.selectbox("Family History of Diabetes", ["Yes", "No"])
    hypertension = st.selectbox("Hypertension History", ["Yes", "No"])
with col2:
    cardiovascular = st.selectbox("Cardiovascular History", ["Yes", "No"])
    diabetes_stage = st.selectbox("Diabetes Stage", ["No Diabetes", "Pre-Diabetes", "Type 1", "Type 2"])
with col3:
    bmi = st.number_input("BMI", min_value=0.0, value=25.0)
    waist_hip = st.number_input("Waist-to-Hip Ratio", min_value=0.0, value=0.85)
st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# CLINICAL MEASUREMENTS
# =========================================================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🧪 Clinical Measurements</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    systolic_bp = st.number_input("Systolic BP", min_value=0.0, value=120.0)
with col2:
    diastolic_bp = st.number_input("Diastolic BP", min_value=0.0, value=80.0)
with col3:
    heart_rate = st.number_input("Heart Rate", min_value=0.0, value=72.0)
with col4:
    hba1c = st.number_input("HbA1c", min_value=0.0, value=5.5)
st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# BLOOD TEST RESULTS
# =========================================================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🩸 Blood Test Results</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    cholesterol_total = st.number_input("Total Cholesterol", min_value=0.0, value=190.0)
with col2:
    hdl = st.number_input("HDL Cholesterol", min_value=0.0, value=50.0)
with col3:
    ldl = st.number_input("LDL Cholesterol", min_value=0.0, value=100.0)
with col4:
    triglycerides = st.number_input("Triglycerides", min_value=0.0, value=150.0)
st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# GLUCOSE & INSULIN
# =========================================================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🧬 Glucose & Insulin</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    glucose_fasting = st.number_input("Fasting Glucose", min_value=0.0, value=100.0)
with col2:
    glucose_post = st.number_input("Postprandial Glucose", min_value=0.0, value=130.0)
with col3:
    insulin = st.number_input("Insulin Level", min_value=0.0, value=15.0)
with col4:
    diabetes_risk_score = st.number_input("Diabetes Risk Score", min_value=0.0, value=5.0)
st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# BUILD FEATURE VECTOR (matches model.feature_names_in_ order exactly)
# =========================================================
def build_feature_vector():
    row = {
        "age": age,
        "gender": GENDER_MAP[gender],
        "education_level": EDUCATION_MAP[education_level],
        "income_level": INCOME_MAP[income_level],
        "smoking_status": SMOKING_MAP[smoking_status],
        "alcohol_consumption_per_week": alcohol,
        "physical_activity_minutes_per_week": physical_activity,
        "diet_score": diet_score,
        "sleep_hours_per_day": sleep_hours,
        "screen_time_hours_per_day": screen_time,
        "family_history_diabetes": BINARY_MAP[family_history],
        "hypertension_history": BINARY_MAP[hypertension],
        "cardiovascular_history": BINARY_MAP[cardiovascular],
        "bmi": bmi,
        "waist_to_hip_ratio": waist_hip,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "heart_rate": heart_rate,
        "cholesterol_total": cholesterol_total,
        "hdl_cholesterol": hdl,
        "ldl_cholesterol": ldl,
        "triglycerides": triglycerides,
        "glucose_fasting": glucose_fasting,
        "glucose_postprandial": glucose_post,
        "insulin_level": insulin,
        "hba1c": hba1c,
        "diabetes_risk_score": diabetes_risk_score,
        "ethnicity_Black": 1 if ethnicity == "Black" else 0,
        "ethnicity_Hispanic": 1 if ethnicity == "Hispanic" else 0,
        "ethnicity_Other": 1 if ethnicity == "Other" else 0,
        "ethnicity_White": 1 if ethnicity == "White" else 0,
        "diabetes_stage_No Diabetes": 1 if diabetes_stage == "No Diabetes" else 0,
        "diabetes_stage_Pre-Diabetes": 1 if diabetes_stage == "Pre-Diabetes" else 0,
        "diabetes_stage_Type 1": 1 if diabetes_stage == "Type 1" else 0,
        "diabetes_stage_Type 2": 1 if diabetes_stage == "Type 2" else 0,
        "employment_status_Retired": 1 if employment_status == "Retired" else 0,
        "employment_status_Student": 1 if employment_status == "Student" else 0,
        "employment_status_Unemployed": 1 if employment_status == "Unemployed" else 0,
    }
    return np.array([[row[f] for f in FEATURE_ORDER]])

# =========================================================
# PREDICT BUTTON + RESULT
# =========================================================
st.markdown("<br>", unsafe_allow_html=True)
predict_clicked = st.button("🔍 Predict Diabetes Risk", use_container_width=True)

if predict_clicked:
    with st.spinner("Analyzing patient data..."):
        X = build_feature_vector()
        proba = model.predict_proba(X)[0]
        risk_score = proba[1]  # probability of class 1 (diabetes risk)
        prediction = model.predict(X)[0]

    if risk_score < 0.33:
        color, text_color, label, icon = "#dcfce7", "#166534", "Low Risk", "✅"
    elif risk_score < 0.66:
        color, text_color, label, icon = "#fef9c3", "#854d0e", "Moderate Risk", "⚠️"
    else:
        color, text_color, label, icon = "#fee2e2", "#991b1b", "High Risk", "🚨"

    st.markdown(
        f"""
        <div class="result-box" style="background:{color}; color:{text_color};">
            {icon} Predicted Diabetes Risk: {label} ({risk_score*100:.1f}%)
        </div>
        """,
        unsafe_allow_html=True
    )
    st.progress(float(risk_score))
    st.caption(f"Model output — class: {int(prediction)} | P(no risk): {proba[0]*100:.1f}% | P(risk): {proba[1]*100:.1f}%")