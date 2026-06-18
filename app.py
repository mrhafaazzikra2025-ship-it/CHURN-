import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Page Configuration
st.set_page_config(
    page_title="Corporate Churn Diagnostics", 
    page_icon="🔮", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize Session State Variables
if "page" not in st.session_state:
    st.session_state.page = "input"

# 2. Inject External Corporate CSS Style Guide
try:
    with open("style.css", "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    # Fallback to embedded styling if file isn't written yet
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
            html, body, [data-testid="stAppViewContainer"] {
                font-family: 'Inter', sans-serif;
                background-color: #0F172A;
            }
            .main-title {
                font-size: 2.5rem;
                font-weight: 700;
                background: linear-gradient(135deg, #38BDF8 0%, #818CF8 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 0.2rem;
            }
            .sub-title { color: #94A3B8; font-size: 1rem; margin-bottom: 2rem; }
            .corp-card {
                background: #1E293B;
                border: 1px solid #334155;
                border-radius: 12px;
                padding: 24px;
                margin-bottom: -15px;
                transition: all 0.3s ease;
            }
            .corp-card:hover {
                transform: translateY(-2px);
                border-color: #4F46E5;
                box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
            }
            .card-header { font-size: 1.25rem; font-weight: 600; color: #F8FAFC; margin-bottom: 15px; }
            div.stButton > button:first-child {
                background: linear-gradient(135deg, #4F46E5 0%, #06B6D4 100%);
                color: white; border: none; padding: 12px 24px; border-radius: 8px; font-weight: 600;
            }
            div.stButton > button:first-child:hover {
                transform: translateY(-1px); box-shadow: 0 4px 12px rgba(79, 70, 229, 0.4); color: white;
            }
        </style>
    """, unsafe_allow_html=True)

# 3. Optimized Model Loading
@st.cache_resource
def load_model():
    return joblib.load("churn_model.pkl")

try:
    model = load_model()
except FileNotFoundError:
    st.error("⚠️ 'churn_model.pkl' tidak ditemukan. Pastikan file tersebut berada di folder yang sama dengan skrip ini!")

# --- PAGE 1: INPUT FORM ---
if st.session_state.page == "input":

    # Header & Corporate UI Presentation
    st.markdown('<div class="main-title">🔮 Diagnostik Churn Pelanggan Perusahaan</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Sistem Penilaian Risiko Attrisi & Paparan Pendapatan Berbasis AI</div>', unsafe_allow_html=True)
    st.markdown("---")

    # Two-Column Layout for Structural Input Balance
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<div class="corp-card"><div class="card-header">📋 Profil Akun</div></div>', unsafe_allow_html=True)
        gender = st.selectbox("Jenis Kelamin", ["Perempuan", "Laki-laki"])
        partner = st.selectbox("Memiliki Pasangan?", ["Tidak", "Ya"])
        dependents = st.selectbox("Memiliki Tanggungan?", ["Tidak", "Ya"])
        contract = st.selectbox("Jenis Kontrak", ["Bulan-ke-bulan", "Satu tahun", "Dua tahun"])
        payment_method = st.selectbox("Metode Pembayaran", [
            "Cek elektronik", "Cek lewat surat", "Transfer bank (otomatis)", "Kartu kredit (otomatis)"
        ])

    with col2:
        st.markdown('<div class="corp-card"><div class="card-header">⚡ Metrik Penggunaan & Tagihan</div></div>', unsafe_allow_html=True)
        tenure = st.slider("Masa Berlangganan (Bulan bersama perusahaan)", 0, 72, 12)
        internet_service = st.selectbox("Penyedia Layanan Internet", ["DSL", "Fiber optic", "Tidak ada"])
        online_security = st.selectbox("Fitur Keamanan Online", ["Tidak", "Ya", "Tidak ada layanan internet"])
        tech_support = st.selectbox("Fitur Dukungan Teknis (Tech Support)", ["Tidak", "Ya", "Tidak ada layanan internet"])
        
        sub_col1, sub_col2 = st.columns(2)
        with sub_col1:
            monthly_charges = st.number_input("Biaya Bulanan ($)", min_value=0.0, max_value=200.0, value=65.0)
        with sub_col2:
            total_charges = st.number_input("Total Biaya ($)", min_value=0.0, max_value=10000.0, value=750.0)

    # Feature Engineering / Categorical Mapping
    gender_encoded = 0 if gender == "Perempuan" else 1
    partner_encoded = 1 if partner == "Ya" else 0
    dependents_encoded = 1 if dependents == "Ya" else 0
    contract_encoded = 0 if contract == "Bulan-ke-bulan" else (1 if contract == "Satu tahun" else 2)

    internet_mapping = {"DSL": 0, "Fiber optic": 1, "Tidak ada": 2}
    internet_encoded = internet_mapping[internet_service]

    security_mapping = {"Tidak": 0, "Tidak ada layanan internet": 1, "Ya": 2}
    security_encoded = security_mapping[online_security]
    tech_support_encoded = security_mapping[tech_support]

    payment_mapping = {
        "Transfer bank (otomatis)": 0, "Kartu kredit (otomatis)": 1, 
        "Cek elektronik": 2, "Cek lewat surat": 3
    }
    payment_encoded = payment_mapping[payment_method]

    default_zero = 0

    # Reconstructing the Complete 19-Feature Vector
    input_data = pd.DataFrame({
        'gender': [gender_encoded],
        'SeniorCitizen': [default_zero],
        'Partner': [partner_encoded],
        'Dependents': [dependents_encoded],
        'tenure': [tenure],
        'PhoneService': [default_zero],
        'MultipleLines': [default_zero],
        'InternetService': [internet_encoded],
        'OnlineSecurity': [security_encoded],
        'OnlineBackup': [default_zero],
        'DeviceProtection': [default_zero],
        'TechSupport': [tech_support_encoded],
        'StreamingTV': [default_zero],
        'StreamingMovies': [default_zero],
        'Contract': [contract_encoded],
        'PaperlessBilling': [default_zero],
        'PaymentMethod': [payment_encoded],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges]
    })

    st.markdown("<br>", unsafe_allow_html=True) 

    # Save stable data structures to session state safely and transition
    if st.button(
        "🚀 Jalankan Diagnostik Perusahaan",
        use_container_width=True
    ):
        st.session_state.input_data = input_data
        st.session_state.contract = contract
        st.session_state.tech_support = tech_support
        st.session_state.monthly_charges = monthly_charges
        st.switch_page("pages/1_Hasil_Diagnostik.py")