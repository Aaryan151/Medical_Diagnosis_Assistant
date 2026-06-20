import streamlit as st
import numpy as np
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="MediPredict AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background: #07090f !important;
    font-family: 'Inter', sans-serif !important;
    color: #e2e8f0 !important;
}
.main .block-container {
    padding: 2rem 3rem 3rem !important;
    max-width: 1100px !important;
}
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { display:none!important; }

[data-testid="stSidebar"] {
    background: #0a0d17 !important;
    border-right: 1px solid rgba(99,102,241,0.15) !important;
}
[data-testid="stSidebar"] .stRadio label {
    display: flex !important;
    align-items: center !important;
    padding: 10px 16px !important;
    border-radius: 8px !important;
    margin: 2px 8px !important;
    font-size: 13.5px !important;
    font-weight: 500 !important;
    color: #64748b !important;
    cursor: pointer !important;
    transition: all 0.15s !important;
    border: 1px solid transparent !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(99,102,241,0.08) !important;
    color: #c7d2fe !important;
}
[data-testid="stSidebar"] [data-baseweb="radio"] { display:none !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #f1f5f9 !important; }

label[data-testid="stWidgetLabel"] p,
.stSelectbox label p,
.stNumberInput label p {
    font-size: 11px !important;
    font-weight: 700 !important;
    color: #64748b !important;
    letter-spacing: 0.6px !important;
    text-transform: uppercase !important;
}

[data-testid="stNumberInput"] input {
    background: #111827 !important;
    border: 1px solid rgba(99,102,241,0.2) !important;
    border-radius: 9px !important;
    color: #e2e8f0 !important;
    font-size: 14px !important;
}
[data-testid="stNumberInput"] input:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
    outline: none !important;
}
[data-testid="stNumberInput"] button {
    background: rgba(99,102,241,0.1) !important;
    border: 1px solid rgba(99,102,241,0.15) !important;
    color: #818cf8 !important;
    border-radius: 8px !important;
}

[data-baseweb="select"] > div {
    background: #111827 !important;
    border: 1px solid rgba(99,102,241,0.2) !important;
    border-radius: 9px !important;
    color: #e2e8f0 !important;
    font-size: 14px !important;
}
[data-baseweb="select"] span { color: #e2e8f0 !important; }
[data-baseweb="popover"] [role="listbox"] {
    background: #111827 !important;
    border: 1px solid rgba(99,102,241,0.25) !important;
    border-radius: 10px !important;
}
[data-baseweb="popover"] [role="option"] {
    color: #e2e8f0 !important;
    background: #111827 !important;
    font-size: 13px !important;
}
[data-baseweb="popover"] [role="option"]:hover,
[data-baseweb="popover"] [aria-selected="true"] {
    background: rgba(99,102,241,0.15) !important;
    color: #a5b4fc !important;
}
svg { fill: #64748b !important; }

.stButton > button {
    background: linear-gradient(135deg, #4f46e5, #6366f1) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    padding: 14px 0 !important;
    width: 100% !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.3) !important;
    transition: all 0.2s !important;
    margin-top: 8px !important;
}
.stButton > button:hover {
    filter: brightness(1.1) !important;
    box-shadow: 0 6px 28px rgba(99,102,241,0.45) !important;
    transform: translateY(-1px) !important;
}

.sec-head {
    display: flex; align-items: center; gap: 10px;
    margin: 24px 0 14px;
    padding-bottom: 10px;
    border-bottom: 1px solid rgba(99,102,241,0.12);
}
.sec-head-icon {
    width: 30px; height: 30px;
    background: rgba(99,102,241,0.1);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 7px;
    display: flex; align-items: center; justify-content: center;
    font-size: 15px;
}
.sec-head-text {
    font-size: 11px; font-weight: 700;
    color: #64748b; letter-spacing: 1.2px; text-transform: uppercase;
}

.page-hero {
    background: linear-gradient(135deg, #0d1530, #0f172a);
    border: 1px solid rgba(99,102,241,0.14);
    border-radius: 16px;
    padding: 32px 36px;
    margin-bottom: 32px;
    position: relative; overflow: hidden;
}
.page-hero::after {
    content: ''; position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(99,102,241,0.1) 0%, transparent 65%);
    border-radius: 50%;
}
.hero-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(99,102,241,0.1); border: 1px solid rgba(99,102,241,0.2);
    border-radius: 20px; padding: 4px 12px;
    font-size: 10px; font-weight: 700; color: #818cf8;
    letter-spacing: 1px; text-transform: uppercase; margin-bottom: 14px;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 30px; font-weight: 700; color: #f1f5f9;
    letter-spacing: -0.8px; line-height: 1.2; margin-bottom: 10px;
}
.hero-title span { color: #818cf8; }
.hero-sub { font-size: 13.5px; color: #64748b; line-height: 1.7; max-width: 480px; margin-bottom: 24px; }
.hero-stats { display: flex; gap: 36px; }
.hs-val { font-family: 'Space Grotesk', sans-serif; font-size: 24px; font-weight: 700; color: #f1f5f9; }
.hs-lab { font-size: 10px; color: #475569; text-transform: uppercase; letter-spacing: 0.7px; margin-top: 2px; }

.d-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 28px; }
.d-card {
    background: #0d1224; border: 1px solid rgba(99,102,241,0.1);
    border-radius: 14px; padding: 22px; transition: all 0.2s;
    position: relative; overflow: hidden;
}
.d-card:hover { border-color: rgba(99,102,241,0.3); transform: translateY(-2px); }
.d-card-bar { position: absolute; top: 0; left: 0; right: 0; height: 2.5px; }
.d-card-ico { font-size: 28px; margin-bottom: 12px; display: block; }
.d-card-title { font-family: 'Space Grotesk', sans-serif; font-size: 15px; font-weight: 700; color: #e2e8f0; margin-bottom: 6px; }
.d-card-desc { font-size: 12px; color: #475569; line-height: 1.6; margin-bottom: 14px; }
.d-card-foot { display: flex; align-items: center; justify-content: space-between; }
.d-acc { font-size: 10px; font-weight: 700; padding: 3px 9px; border-radius: 8px; background: rgba(99,102,241,0.1); color: #818cf8; border: 1px solid rgba(99,102,241,0.2); }
.d-arrow { color: #334155; font-size: 16px; }

.topbar { display: flex; align-items: center; justify-content: space-between; margin-bottom: 28px; }
.breadcrumb { font-size: 12px; color: #334155; display: flex; align-items: center; gap: 8px; }
.breadcrumb .cur { color: #a5b4fc; font-weight: 600; }
.t-badge { padding: 5px 12px; border-radius: 20px; font-size: 10.5px; font-weight: 700;
    background: rgba(99,102,241,0.1); border: 1px solid rgba(99,102,241,0.2); color: #818cf8; }

.pred-hdr { margin-bottom: 28px; }
.pred-hdr-title { font-family: 'Space Grotesk', sans-serif; font-size: 22px; font-weight: 700; color: #f1f5f9; letter-spacing: -0.4px; margin-bottom: 4px; }
.pred-hdr-sub { font-size: 13px; color: #475569; }

.result-danger {
    background: linear-gradient(135deg, #1c0909, #1f0d0d);
    border: 1px solid rgba(239,68,68,0.25); border-radius: 14px;
    padding: 26px 28px; margin-top: 24px; position: relative; overflow: hidden;
}
.result-danger::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #dc2626, #f87171);
}
.result-success {
    background: linear-gradient(135deg, #071510, #0a1a14);
    border: 1px solid rgba(34,197,94,0.25); border-radius: 14px;
    padding: 26px 28px; margin-top: 24px; position: relative; overflow: hidden;
}
.result-success::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #16a34a, #4ade80);
}
.r-icon { font-size: 32px; margin-bottom: 10px; }
.r-title { font-family: 'Space Grotesk', sans-serif; font-size: 20px; font-weight: 700; margin-bottom: 4px; }
.result-danger .r-title { color: #fca5a5; }
.result-success .r-title { color: #86efac; }
.r-sub { font-size: 12px; color: #475569; margin-bottom: 18px; }
.r-bar-label { font-size: 10px; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: 0.7px; margin-bottom: 7px; }
.r-bar-track { height: 5px; background: rgba(255,255,255,0.06); border-radius: 10px; overflow: hidden; }
.r-bar-d { height: 100%; background: linear-gradient(90deg, #dc2626, #f87171); border-radius: 10px; }
.r-bar-s { height: 100%; background: linear-gradient(90deg, #16a34a, #4ade80); border-radius: 10px; }
.r-pct { font-size: 13px; font-weight: 700; margin-top: 7px; }
.result-danger .r-pct { color: #f87171; }
.result-success .r-pct { color: #4ade80; }
.r-advice {
    background: rgba(99,102,241,0.07); border: 1px solid rgba(99,102,241,0.14);
    border-radius: 10px; padding: 14px 18px;
    font-size: 12.5px; color: #94a3b8; line-height: 1.7; margin-top: 14px;
}
.r-advice strong { color: #a5b4fc; }

.disc {
    background: rgba(234,179,8,0.05); border: 1px solid rgba(234,179,8,0.14);
    border-radius: 10px; padding: 12px 16px;
    font-size: 11.5px; color: #a16207; margin-top: 24px; line-height: 1.6;
}
.disc strong { color: #ca8a04; }

[data-testid="stForm"] {
    background: #0a0f1e !important;
    border: 1px solid rgba(99,102,241,0.12) !important;
    border-radius: 14px !important;
    padding: 16px 24px 20px !important;
}
[data-testid="column"] { padding: 0 6px !important; }
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #07090f; }
::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ── Train models fresh (no pkl needed — works on any Python version) ───────
@st.cache_resource(show_spinner=False)
def load_models():
    base = os.path.dirname(__file__)

    # ── Heart Disease ──────────────────────────────────────────────────────
    df = pd.read_csv(os.path.join(base, "data", "Dataset_Heart_Disease.csv"))
    if "Unnamed: 0" in df.columns:
        df = df.drop("Unnamed: 0", axis=1)
    X = df.drop("target", axis=1)
    y = df["target"]
    Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.2, random_state=42)
    heart = RandomForestClassifier(n_estimators=100, random_state=42)
    heart.fit(Xt, yt)

    # ── Diabetes ───────────────────────────────────────────────────────────
    df = pd.read_csv(os.path.join(base, "data", "diabetes.csv"))
    for col in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
        df[col] = df[col].replace(0, df[col].median())
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    diabetes = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42)
    diabetes.fit(Xt, yt)

    # ── Lung Cancer ────────────────────────────────────────────────────────
    df = pd.read_csv(os.path.join(base, "data", "Lung_Cancer_Dataset.csv"))
    df["GENDER"] = df["GENDER"].map({"M": 1, "F": 0})
    df["LUNG_CANCER"] = df["LUNG_CANCER"].map({"YES": 1, "NO": 0})
    X = df.drop("LUNG_CANCER", axis=1)
    y = df["LUNG_CANCER"]
    Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    lung = RandomForestClassifier(n_estimators=300, random_state=42)
    lung.fit(Xt, yt)

    # ── Thyroid ────────────────────────────────────────────────────────────
    df = pd.read_csv(os.path.join(base, "data", "thyroid_high_accuracy_dataset.csv"))
    X = df.drop("binaryClass", axis=1)
    y = df["binaryClass"]
    Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    thyroid = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_split=5, min_samples_leaf=2, random_state=42)
    thyroid.fit(Xt, yt)

    return {"heart": heart, "diabetes": diabetes, "lung": lung, "thyroid": thyroid}


# Show spinner while training
with st.spinner("⚙️ Loading MediPredict AI — training models, please wait ~15 seconds..."):
    models = load_models()


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:24px 16px 16px;">
        <div style="width:42px;height:42px;background:linear-gradient(135deg,#6366f1,#818cf8);
             border-radius:11px;display:flex;align-items:center;justify-content:center;
             font-size:20px;margin-bottom:12px;box-shadow:0 0 18px rgba(99,102,241,0.35);">🏥</div>
        <div style="font-family:'Space Grotesk',sans-serif;font-size:17px;font-weight:700;color:#f1f5f9;">MediPredict AI</div>
        <div style="font-size:10px;color:#475569;text-transform:uppercase;letter-spacing:0.6px;margin-top:2px;">Clinical Intelligence</div>
    </div>
    <hr style="border:none;border-top:1px solid rgba(99,102,241,0.12);margin:0 0 8px;">
    <div style="font-size:9.5px;font-weight:700;color:#475569;letter-spacing:1.2px;
         text-transform:uppercase;padding:8px 16px 4px;">Navigation</div>
    """, unsafe_allow_html=True)

    page = st.radio("", [
        "🏠  Dashboard",
        "❤️  Heart Disease",
        "🩸  Diabetes",
        "🫁  Lung Cancer",
        "🦋  Thyroid"
    ], label_visibility="collapsed")

    st.markdown("""
    <hr style="border:none;border-top:1px solid rgba(99,102,241,0.1);margin:12px 0 8px;">
    <div style="padding:8px 16px 16px;font-size:11px;color:#334155;line-height:1.7;">
        For educational use only.<br>Not a substitute for professional medical advice.
    </div>
    """, unsafe_allow_html=True)


# ── Result helper ──────────────────────────────────────────────────────────
def show_result(is_pos, conf, pos_title, neg_title, advice_pos, advice_neg):
    pct = conf * 100
    if is_pos:
        st.markdown(f"""
        <div class="result-danger">
            <div class="r-icon">⚠️</div>
            <div class="r-title">{pos_title}</div>
            <div class="r-sub">Model prediction based on clinical inputs</div>
            <div class="r-bar-label">Confidence Score</div>
            <div class="r-bar-track"><div class="r-bar-d" style="width:{pct:.1f}%"></div></div>
            <div class="r-pct">{pct:.1f}%</div>
        </div>
        <div class="r-advice">{advice_pos}</div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-success">
            <div class="r-icon">✅</div>
            <div class="r-title">{neg_title}</div>
            <div class="r-sub">Model prediction based on clinical inputs</div>
            <div class="r-bar-label">Confidence Score</div>
            <div class="r-bar-track"><div class="r-bar-s" style="width:{pct:.1f}%"></div></div>
            <div class="r-pct">{pct:.1f}%</div>
        </div>
        <div class="r-advice">{advice_neg}</div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════════
if "Dashboard" in page:
    st.markdown("""
    <div class="topbar">
        <div class="breadcrumb"><span>MediPredict</span><span>›</span><span class="cur">Dashboard</span></div>
        <div class="t-badge">🟢 4 Models Active</div>
    </div>
    <div class="page-hero">
        <div class="hero-badge">🧠 AI-Powered Clinical Tool</div>
        <div class="hero-title">Predict. Understand.<br><span>Act early.</span></div>
        <div class="hero-sub">Random Forest models trained on real clinical datasets. Assess risk across 4 major diseases — instantly, with confidence scoring.</div>
        <div class="hero-stats">
            <div><div class="hs-val">4</div><div class="hs-lab">Diseases</div></div>
            <div><div class="hs-val">~95%</div><div class="hs-lab">Peak Accuracy</div></div>
            <div><div class="hs-val">RF</div><div class="hs-lab">Model Type</div></div>
        </div>
    </div>
    <div class="d-grid">
        <div class="d-card">
            <div class="d-card-bar" style="background:linear-gradient(90deg,#ef4444,#f87171)"></div>
            <span class="d-card-ico">❤️</span>
            <div class="d-card-title">Heart Disease</div>
            <div class="d-card-desc">11 clinical indicators: ECG, cholesterol, ST slope, chest pain type, exercise angina, and more.</div>
            <div class="d-card-foot"><span class="d-acc">~95% Accuracy</span><span class="d-arrow">→</span></div>
        </div>
        <div class="d-card">
            <div class="d-card-bar" style="background:linear-gradient(90deg,#3b82f6,#60a5fa)"></div>
            <span class="d-card-ico">🩸</span>
            <div class="d-card-title">Diabetes</div>
            <div class="d-card-desc">8 metabolic metrics: glucose, BMI, insulin, blood pressure, skin thickness, and family history.</div>
            <div class="d-card-foot"><span class="d-acc">~80% Accuracy</span><span class="d-arrow">→</span></div>
        </div>
        <div class="d-card">
            <div class="d-card-bar" style="background:linear-gradient(90deg,#f59e0b,#fbbf24)"></div>
            <span class="d-card-ico">🫁</span>
            <div class="d-card-title">Lung Cancer</div>
            <div class="d-card-desc">15 lifestyle and symptom features: smoking, chronic disease, respiratory symptoms, and more.</div>
            <div class="d-card-foot"><span class="d-acc">~99% Accuracy</span><span class="d-arrow">→</span></div>
        </div>
        <div class="d-card">
            <div class="d-card-bar" style="background:linear-gradient(90deg,#8b5cf6,#a78bfa)"></div>
            <span class="d-card-ico">🦋</span>
            <div class="d-card-title">Thyroid Disease</div>
            <div class="d-card-desc">10 features: hormone levels TSH, TT4, FTI, T4U ratios plus medication history and demographics.</div>
            <div class="d-card-foot"><span class="d-acc">~99% Accuracy</span><span class="d-arrow">→</span></div>
        </div>
    </div>
    <div class="disc">
        <strong>⚠️ Disclaimer:</strong> MediPredict is for educational and research purposes only.
        It is <strong>not</strong> a substitute for professional medical advice, diagnosis, or treatment.
        Always consult a qualified healthcare provider.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# HEART DISEASE
# ══════════════════════════════════════════════════════════════
elif "Heart" in page:
    st.markdown("""
    <div class="topbar">
        <div class="breadcrumb"><span>MediPredict</span><span>›</span><span class="cur">Heart Disease</span></div>
        <div class="t-badge" style="background:rgba(239,68,68,0.1);border-color:rgba(239,68,68,0.2);color:#f87171;">❤️ Cardiac Module</div>
    </div>
    <div class="pred-hdr">
        <div class="pred-hdr-title">❤️ Heart Disease Risk Assessment</div>
        <div class="pred-hdr-sub">Enter the patient's clinical data below. All fields affect the prediction.</div>
    </div>
    """, unsafe_allow_html=True)

    with st.form("heart_form"):
        st.markdown('<div class="sec-head"><div class="sec-head-icon">👤</div><div class="sec-head-text">Personal Information</div></div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        age  = c1.number_input("Age (years)", 20, 100, 45)
        sex  = c2.selectbox("Sex", ["Male", "Female"])
        fbs  = c3.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])

        st.markdown('<div class="sec-head"><div class="sec-head-icon">🫀</div><div class="sec-head-text">Cardiac Indicators</div></div>', unsafe_allow_html=True)
        c4, c5, c6 = st.columns(3)
        cp   = c4.selectbox("Chest Pain Type", ["0 – Typical Angina", "1 – Atypical Angina", "2 – Non-anginal Pain", "3 – Asymptomatic"])
        rbps = c5.number_input("Resting BP (mm Hg)", 80, 220, 120)
        chol = c6.number_input("Cholesterol (mg/dl)", 100, 600, 200)

        c7, c8, c9 = st.columns(3)
        ecg   = c7.selectbox("Resting ECG", ["0 – Normal", "1 – ST-T Abnormality", "2 – LV Hypertrophy"])
        maxhr = c8.number_input("Max Heart Rate", 60, 220, 150)
        exang = c9.selectbox("Exercise-Induced Angina", ["No", "Yes"])

        c10, c11 = st.columns(2)
        oldpeak = c10.number_input("Oldpeak (ST Depression)", 0.0, 7.0, 1.0, step=0.1)
        slope   = c11.selectbox("ST Slope", ["0 – Upsloping", "1 – Flat", "2 – Downsloping"])

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🔍  Run Heart Disease Prediction")

    if submitted:
        f = np.array([[age, 1 if sex=="Male" else 0, int(cp[0]), rbps, chol,
                       1 if fbs=="Yes" else 0, int(ecg[0]), maxhr,
                       1 if exang=="Yes" else 0, oldpeak, int(slope[0])]])
        res  = models["heart"].predict(f)[0]
        prob = models["heart"].predict_proba(f)[0]
        conf = prob[1] if res == 1 else prob[0]
        show_result(res==1, conf,
            "High Risk — Heart Disease Detected",
            "Low Risk — No Heart Disease Detected",
            "<strong>Recommendation:</strong> Elevated cardiovascular risk detected. Consult a cardiologist immediately. Consider ECG, echocardiogram, and coronary angiography.",
            "<strong>Recommendation:</strong> No cardiac disease detected. Maintain a heart-healthy diet, exercise regularly, and monitor blood pressure and cholesterol."
        )


# ══════════════════════════════════════════════════════════════
# DIABETES
# ══════════════════════════════════════════════════════════════
elif "Diabetes" in page:
    st.markdown("""
    <div class="topbar">
        <div class="breadcrumb"><span>MediPredict</span><span>›</span><span class="cur">Diabetes</span></div>
        <div class="t-badge" style="background:rgba(59,130,246,0.1);border-color:rgba(59,130,246,0.2);color:#60a5fa;">🩸 Metabolic Module</div>
    </div>
    <div class="pred-hdr">
        <div class="pred-hdr-title">🩸 Diabetes Risk Assessment</div>
        <div class="pred-hdr-sub">Enter the patient's metabolic health metrics for diabetes screening.</div>
    </div>
    """, unsafe_allow_html=True)

    with st.form("diabetes_form"):
        st.markdown('<div class="sec-head"><div class="sec-head-icon">👤</div><div class="sec-head-text">Personal Information</div></div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        preg = c1.number_input("Number of Pregnancies", 0, 20, 1)
        age  = c2.number_input("Age (years)", 10, 100, 30)

        st.markdown('<div class="sec-head"><div class="sec-head-icon">🩺</div><div class="sec-head-text">Health & Metabolic Metrics</div></div>', unsafe_allow_html=True)
        c3, c4, c5 = st.columns(3)
        glucose = c3.number_input("Glucose (mg/dl)", 0, 300, 110)
        bp      = c4.number_input("Blood Pressure (mm Hg)", 0, 150, 70)
        skin    = c5.number_input("Skin Thickness (mm)", 0, 100, 20)

        c6, c7, c8 = st.columns(3)
        insulin = c6.number_input("Insulin (μU/ml)", 0, 900, 80)
        bmi     = c7.number_input("BMI", 0.0, 70.0, 25.0, step=0.1)
        dpf     = c8.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.47, step=0.01)

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🔍  Run Diabetes Prediction")

    if submitted:
        f = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
        res  = models["diabetes"].predict(f)[0]
        prob = models["diabetes"].predict_proba(f)[0]
        conf = prob[1] if res == 1 else prob[0]
        show_result(res==1, conf,
            "High Risk — Diabetic Pattern Detected",
            "Low Risk — No Diabetes Detected",
            "<strong>Recommendation:</strong> Diabetic pattern detected. Consult an endocrinologist. Get HbA1c and fasting glucose blood tests. Begin lifestyle modification immediately.",
            "<strong>Recommendation:</strong> No diabetes detected. Maintain a low-sugar balanced diet, exercise regularly, and get annual glucose screenings."
        )


# ══════════════════════════════════════════════════════════════
# LUNG CANCER
# ══════════════════════════════════════════════════════════════
elif "Lung" in page:
    st.markdown("""
    <div class="topbar">
        <div class="breadcrumb"><span>MediPredict</span><span>›</span><span class="cur">Lung Cancer</span></div>
        <div class="t-badge" style="background:rgba(245,158,11,0.1);border-color:rgba(245,158,11,0.2);color:#fbbf24;">🫁 Oncology Module</div>
    </div>
    <div class="pred-hdr">
        <div class="pred-hdr-title">🫁 Lung Cancer Risk Assessment</div>
        <div class="pred-hdr-sub">Enter the patient's lifestyle and symptom data for lung cancer screening.</div>
    </div>
    """, unsafe_allow_html=True)

    def yn(v): return 2 if "Yes" in v else 1

    with st.form("lung_form"):
        st.markdown('<div class="sec-head"><div class="sec-head-icon">👤</div><div class="sec-head-text">Personal Information</div></div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        gender = c1.selectbox("Gender", ["Male", "Female"])
        age    = c2.number_input("Age (years)", 10, 100, 45)

        st.markdown('<div class="sec-head"><div class="sec-head-icon">🚬</div><div class="sec-head-text">Lifestyle Risk Factors</div></div>', unsafe_allow_html=True)
        c3, c4, c5 = st.columns(3)
        smoking = c3.selectbox("Smoking", ["No (1)", "Yes (2)"])
        yfing   = c4.selectbox("Yellow Fingers", ["No (1)", "Yes (2)"])
        alcohol = c5.selectbox("Alcohol Consuming", ["No (1)", "Yes (2)"])
        c6, c7 = st.columns(2)
        anxiety = c6.selectbox("Anxiety", ["No (1)", "Yes (2)"])
        peer    = c7.selectbox("Peer Pressure", ["No (1)", "Yes (2)"])

        st.markdown('<div class="sec-head"><div class="sec-head-icon">🩺</div><div class="sec-head-text">Medical History & Symptoms</div></div>', unsafe_allow_html=True)
        c8, c9, c10 = st.columns(3)
        chronic = c8.selectbox("Chronic Disease", ["No (1)", "Yes (2)"])
        fatigue = c9.selectbox("Fatigue", ["No (1)", "Yes (2)"])
        allergy = c10.selectbox("Allergy", ["No (1)", "Yes (2)"])

        c11, c12, c13 = st.columns(3)
        wheeze = c11.selectbox("Wheezing", ["No (1)", "Yes (2)"])
        cough  = c12.selectbox("Coughing", ["No (1)", "Yes (2)"])
        shortb = c13.selectbox("Shortness of Breath", ["No (1)", "Yes (2)"])

        c14, c15 = st.columns(2)
        swall  = c14.selectbox("Swallowing Difficulty", ["No (1)", "Yes (2)"])
        chestp = c15.selectbox("Chest Pain", ["No (1)", "Yes (2)"])

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🔍  Run Lung Cancer Prediction")

    if submitted:
        f = np.array([[1 if gender=="Male" else 0, age,
                       yn(smoking), yn(yfing), yn(anxiety), yn(peer),
                       yn(chronic), yn(fatigue), yn(allergy), yn(wheeze),
                       yn(alcohol), yn(cough), yn(shortb), yn(swall), yn(chestp)]])
        res  = models["lung"].predict(f)[0]
        prob = models["lung"].predict_proba(f)[0]
        conf = prob[1] if res == 1 else prob[0]
        show_result(res==1, conf,
            "High Risk — Lung Cancer Indicators Found",
            "Low Risk — No Lung Cancer Detected",
            "<strong>Recommendation:</strong> Elevated risk detected. Consult a pulmonologist immediately. Request a low-dose CT scan. Begin smoking cessation if applicable.",
            "<strong>Recommendation:</strong> No lung cancer risk detected. Avoid tobacco exposure and get annual chest screenings if you smoke."
        )


# ══════════════════════════════════════════════════════════════
# THYROID
# ══════════════════════════════════════════════════════════════
elif "Thyroid" in page:
    st.markdown("""
    <div class="topbar">
        <div class="breadcrumb"><span>MediPredict</span><span>›</span><span class="cur">Thyroid</span></div>
        <div class="t-badge" style="background:rgba(139,92,246,0.1);border-color:rgba(139,92,246,0.2);color:#a78bfa;">🦋 Endocrine Module</div>
    </div>
    <div class="pred-hdr">
        <div class="pred-hdr-title">🦋 Thyroid Disease Risk Assessment</div>
        <div class="pred-hdr-sub">Enter the patient's thyroid hormone levels and clinical history.</div>
    </div>
    """, unsafe_allow_html=True)

    def p01(v): return int(v.split("(")[1][0])

    with st.form("thyroid_form"):
        st.markdown('<div class="sec-head"><div class="sec-head-icon">👤</div><div class="sec-head-text">Personal Information</div></div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        age = c1.number_input("Age (years)", 1, 100, 40)
        sex = c2.selectbox("Sex", ["Female (1)", "Male (0)"])

        st.markdown('<div class="sec-head"><div class="sec-head-icon">💊</div><div class="sec-head-text">Medication History</div></div>', unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        thyrox  = c3.selectbox("On Thyroxine", ["No (0)", "Yes (1)"])
        antithy = c4.selectbox("On Antithyroid Medication", ["No (0)", "Yes (1)"])

        st.markdown('<div class="sec-head"><div class="sec-head-icon">🔬</div><div class="sec-head-text">Hormone Levels</div></div>', unsafe_allow_html=True)
        c5, c6, c7 = st.columns(3)
        tsh = c5.number_input("TSH Level (mIU/L)", 0.0, 100.0, 2.5, step=0.1)
        tt4 = c6.number_input("Total T4 (nmol/L)", 0.0, 400.0, 100.0, step=1.0)
        t4u = c7.number_input("T4U Binding Ratio", 0.0, 3.0, 1.0, step=0.01)

        c8, c9, c10 = st.columns(3)
        fti     = c8.number_input("Free Thyroxine Index (FTI)", 0.0, 400.0, 100.0, step=1.0)
        tsh_tt4 = c9.number_input("TSH / TT4 Ratio", 0.0, 10.0, 0.025, step=0.001, format="%.4f")
        fti_tt4 = c10.number_input("FTI / TT4 Ratio", 0.0, 5.0, 1.0, step=0.01)

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🔍  Run Thyroid Prediction")

    if submitted:
        f = np.array([[age, p01(sex), p01(thyrox), p01(antithy),
                       tsh, tt4, t4u, fti, tsh_tt4, fti_tt4]])
        res  = models["thyroid"].predict(f)[0]
        prob = models["thyroid"].predict_proba(f)[0]
        conf = prob[int(res)]
        show_result(int(res)==1, conf,
            "High Risk — Thyroid Dysfunction Detected",
            "Low Risk — No Thyroid Disease Detected",
            "<strong>Recommendation:</strong> Thyroid dysfunction detected. Consult an endocrinologist. Get a full thyroid panel (TSH, Free T3, Free T4, Anti-TPO) and thyroid ultrasound.",
            "<strong>Recommendation:</strong> No thyroid abnormality detected. Monitor hormone levels regularly, especially with a family history of thyroid disorders."
        )
