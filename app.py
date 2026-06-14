import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import io
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark background */
.stApp {
    background: #0a0e1a;
    color: #e2e8f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0d1224 !important;
    border-right: 1px solid #1e2d4a;
}
[data-testid="stSidebar"] * {color: #cbd5e1 !important;}
[data-testid="stSidebar"] .stRadio label {
    padding: 8px 12px;
    border-radius: 8px;
    cursor: pointer;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #0f1a2e 0%, #1a2540 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
}
.metric-card .label {
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 8px;
}
.metric-card .value {
    font-size: 32px;
    font-weight: 700;
    color: #38bdf8;
    font-family: 'JetBrains Mono', monospace;
}
.metric-card .value.danger { color: #f87171; }
.metric-card .value.success { color: #34d399; }
.metric-card .value.warning { color: #fbbf24; }

/* Fraud alert banners */
.alert-fraud {
    background: linear-gradient(135deg, #450a0a, #7f1d1d);
    border: 1px solid #ef4444;
    border-left: 4px solid #ef4444;
    border-radius: 10px;
    padding: 20px 24px;
    color: #fecaca;
    font-size: 18px;
    font-weight: 600;
    text-align: center;
}
.alert-legit {
    background: linear-gradient(135deg, #052e16, #14532d);
    border: 1px solid #22c55e;
    border-left: 4px solid #22c55e;
    border-radius: 10px;
    padding: 20px 24px;
    color: #bbf7d0;
    font-size: 18px;
    font-weight: 600;
    text-align: center;
}

/* Section headers */
.section-header {
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #38bdf8;
    border-bottom: 1px solid #1e3a5f;
    padding-bottom: 8px;
    margin: 24px 0 16px 0;
}

/* Risk badge */
.risk-badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 13px;
    letter-spacing: 0.5px;
}
.risk-low    { background: #052e16; color: #34d399; border: 1px solid #34d399; }
.risk-medium { background: #451a03; color: #fbbf24; border: 1px solid #fbbf24; }
.risk-high   { background: #450a0a; color: #f87171; border: 1px solid #f87171; }
.risk-critical{ background: #3b0764; color: #e879f9; border: 1px solid #e879f9; }

/* Input fields */
.stNumberInput input, .stTextInput input {
    background: #0f1a2e !important;
    border: 1px solid #1e3a5f !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
}
.stNumberInput input:focus, .stTextInput input:focus {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 0 2px rgba(56,189,248,0.15) !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #0369a1, #0ea5e9) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 10px 28px !important;
    font-size: 14px !important;
    letter-spacing: 0.3px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #0284c7, #38bdf8) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 15px rgba(14,165,233,0.3) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #0d1224;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #64748b;
    border-radius: 8px;
    padding: 8px 20px;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: #1e3a5f !important;
    color: #38bdf8 !important;
}

/* Dataframe */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* Hide streamlit branding */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }

/* Page title */
.page-title {
    font-size: 28px;
    font-weight: 700;
    color: #f1f5f9;
    margin-bottom: 4px;
}
.page-subtitle {
    font-size: 14px;
    color: #475569;
    margin-bottom: 28px;
}
</style>
""", unsafe_allow_html=True)

# ── Feature list (exact training order) ────────────────────────────────────────
FEATURE_COLS = [f'V{i}' for i in range(1, 29)] + ['scaled_amount', 'scaled_time']

# ── Load models ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    errors = []
    model, amt_scaler, time_scaler = None, None, None

    for fname in ['fraud_model.pkl', 'amount_scaler.pkl', 'time_scaler.pkl']:
        if not os.path.exists(fname):
            errors.append(f"❌ Missing file: **{fname}**")

    if errors:
        return None, None, None, errors

    try:
        model = joblib.load('fraud_model.pkl')
        amt_scaler = joblib.load('amount_scaler.pkl')
        time_scaler = joblib.load('time_scaler.pkl')
        return model, amt_scaler, time_scaler, []
    except Exception as e:
        return None, None, None, [f"❌ Error loading model files: {str(e)}"]

model, amt_scaler, time_scaler, load_errors = load_models()

# ── Helpers ────────────────────────────────────────────────────────────────────
def get_risk_level(prob):
    if prob < 0.20:   return "Low Risk",      "risk-low",      "🟢"
    if prob < 0.50:   return "Medium Risk",   "risk-medium",   "🟡"
    if prob < 0.80:   return "High Risk",     "risk-high",     "🔴"
    return               "Critical Risk",  "risk-critical", "🔴"

def preprocess_row(amount, time, v_vals):
    scaled_amt  = amt_scaler.transform([[amount]])[0][0]
    scaled_time = time_scaler.transform([[time]])[0][0]
    row = v_vals + [scaled_amt, scaled_time]
    return pd.DataFrame([row], columns=FEATURE_COLS)

def preprocess_df(df_raw):
    df = df_raw.copy()
    df['scaled_amount'] = amt_scaler.transform(df[['Amount']])
    df['scaled_time']   = time_scaler.transform(df[['Time']])
    return df[FEATURE_COLS]

def gauge_chart(prob):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 2),
        number={'suffix': '%', 'font': {'size': 32, 'color': '#e2e8f0', 'family': 'JetBrains Mono'}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': '#475569', 'tickfont': {'color': '#475569'}},
            'bar': {'color': '#ef4444' if prob > 0.5 else '#38bdf8'},
            'bgcolor': '#0f1a2e',
            'bordercolor': '#1e3a5f',
            'steps': [
                {'range': [0, 20],  'color': '#052e16'},
                {'range': [20, 50], 'color': '#1c1917'},
                {'range': [50, 80], 'color': '#450a0a'},
                {'range': [80, 100],'color': '#3b0764'},
            ],
            'threshold': {'line': {'color': '#fbbf24', 'width': 3}, 'value': prob * 100}
        },
        title={'text': "Fraud Probability", 'font': {'color': '#94a3b8', 'size': 14}}
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=260,
        margin=dict(t=40, b=0, l=20, r=20),
        font={'color': '#e2e8f0'}
    )
    return fig

def dark_fig(fig):
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#0f1a2e',
        font={'color': '#cbd5e1', 'family': 'Inter'},
        xaxis={'gridcolor': '#1e3a5f', 'linecolor': '#1e3a5f'},
        yaxis={'gridcolor': '#1e3a5f', 'linecolor': '#1e3a5f'},
        margin=dict(t=40, b=40, l=40, r=20)
    )
    return fig

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 24px'>
        <div style='font-size:36px'>🛡️</div>
        <div style='font-size:16px; font-weight:700; color:#f1f5f9; margin-top:8px'>FraudGuard AI</div>
        <div style='font-size:11px; color:#475569; letter-spacing:1px; margin-top:4px'>CREDIT CARD PROTECTION</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio("Navigation", [
        "🏠  Overview",
        "🔍  Single Transaction",
        "📂  Bulk CSV Analysis",
        "📄  PDF Analysis",
        "📊  Analytics",
        "📈  Model Performance"
    ])

    st.markdown("---")

    # Model status
    if load_errors:
        st.error("Model Not Loaded")
        for e in load_errors:
            st.markdown(e)
    else:
        st.success("✅ Model Ready")
        st.markdown(f"""
        <div style='font-size:11px; color:#475569; margin-top:8px'>
        <b style='color:#64748b'>Model:</b> fraud_model.pkl<br>
        <b style='color:#64748b'>Scalers:</b> amount_scaler.pkl, time_scaler.pkl<br>
        <b style='color:#64748b'>Features:</b> 30 (V1–V28 + Amount + Time)
        </div>
        """, unsafe_allow_html=True)

# ── Guard: model not loaded ────────────────────────────────────────────────────
def model_error_banner():
    st.error("### ⚠️ Model Files Not Found")
    st.markdown("Please place the following files in the same directory as `app.py`:")
    for f in ['fraud_model.pkl', 'amount_scaler.pkl', 'time_scaler.pkl']:
        st.code(f)
    st.info("These files are generated after training your model in the Colab notebook.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if "Overview" in page:
    st.markdown('<div class="page-title">AI-Powered Credit Card Fraud Detection System</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Real-time transaction monitoring powered by machine learning</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    cards = [
        ("BEST MODEL", "Random Forest", "value"),
        ("FEATURES", "30", "value"),
        ("STATUS", "ACTIVE" if not load_errors else "OFFLINE",
         "success" if not load_errors else "danger"),
    ]
    for col, (label, val, cls) in zip([col1, col2, col3], cards):
        col.markdown(f"""
            <div class='metric-card'>
                <div class='label'>{label}</div>
                <div class='value {cls}' style='font-size:20px'>{val}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
        <div style='background:#0f1a2e; border:1px solid #1e3a5f; border-radius:12px; padding:24px 32px'>
            <div class='section-header'>Models Used in This Project</div>
            <div style='display:grid; grid-template-columns:1fr 1fr 1fr 1fr; gap:16px; margin-top:16px'>
                <div style='text-align:center; padding:16px; background:#0a0e1a; border:1px solid #1e3a5f; border-radius:10px'>
                    <div style='font-size:24px'>🌲</div>
                    <div style='font-weight:700; color:#38bdf8; margin:8px 0 4px'>Random Forest</div>
                    <div style='font-size:11px; color:#64748b; letter-spacing:1px'>SUPERVISED — ENSEMBLE</div>
                    <div style='font-size:12px; color:#94a3b8; margin-top:8px'>Best performer. Used for all dashboard predictions.</div>
                </div>
                <div style='text-align:center; padding:16px; background:#0a0e1a; border:1px solid #1e3a5f; border-radius:10px'>
                    <div style='font-size:24px'>📈</div>
                    <div style='font-weight:700; color:#34d399; margin:8px 0 4px'>Logistic Regression</div>
                    <div style='font-size:11px; color:#64748b; letter-spacing:1px'>SUPERVISED — LINEAR</div>
                    <div style='font-size:12px; color:#94a3b8; margin-top:8px'>Fast linear baseline for binary classification.</div>
                </div>
                <div style='text-align:center; padding:16px; background:#0a0e1a; border:1px solid #1e3a5f; border-radius:10px'>
                    <div style='font-size:24px'>🧠</div>
                    <div style='font-weight:700; color:#a78bfa; margin:8px 0 4px'>Neural Network</div>
                    <div style='font-size:11px; color:#64748b; letter-spacing:1px'>SUPERVISED — MLP</div>
                    <div style='font-size:12px; color:#94a3b8; margin-top:8px'>Multi-layer perceptron with ReLU activation.</div>
                </div>
                <div style='text-align:center; padding:16px; background:#0a0e1a; border:1px solid #1e3a5f; border-radius:10px'>
                    <div style='font-size:24px'>🔵</div>
                    <div style='font-weight:700; color:#fbbf24; margin:8px 0 4px'>K-Means</div>
                    <div style='font-size:11px; color:#64748b; letter-spacing:1px'>UNSUPERVISED — CLUSTERING</div>
                    <div style='font-size:12px; color:#94a3b8; margin-top:8px'>Anomaly detection without class labels.</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:#0f1a2e; border:1px solid #1e3a5f; border-radius:12px; padding:28px 32px'>
        <div class='section-header'>How It Works</div>
        <div style='display:grid; grid-template-columns:1fr 1fr 1fr; gap:20px; margin-top:16px'>
            <div style='text-align:center; padding:16px'>
                <div style='font-size:28px'>⚙️</div>
                <div style='font-weight:600; color:#e2e8f0; margin:8px 0 4px'>Pre-Processing</div>
                <div style='font-size:13px; color:#64748b'>Amount & Time are scaled using fitted StandardScalers. V1–V28 are PCA features from the dataset.</div>
            </div>
            <div style='text-align:center; padding:16px'>
                <div style='font-size:28px'>🤖</div>
                <div style='font-weight:600; color:#e2e8f0; margin:8px 0 4px'>Model Prediction</div>
                <div style='font-size:13px; color:#64748b'>model.predict() and model.predict_proba() are called on the processed 30-feature input.</div>
            </div>
            <div style='text-align:center; padding:16px'>
                <div style='font-size:28px'>🛡️</div>
                <div style='font-weight:600; color:#e2e8f0; margin:8px 0 4px'>Risk Assessment</div>
                <div style='font-size:13px; color:#64748b'>Fraud probability is mapped to Low / Medium / High / Critical risk levels with confidence scores.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Feature Overview</div>', unsafe_allow_html=True)
    feat_df = pd.DataFrame({
        "Feature": ['V1–V28', 'Amount', 'Time'],
        "Type": ['PCA Components', 'Transaction Amount (scaled)', 'Seconds since first transaction (scaled)'],
        "Description": [
            'Anonymized numerical features from PCA transformation of original transaction data',
            'Raw transaction amount — scaled using amount_scaler.pkl before prediction',
            'Raw time value — scaled using time_scaler.pkl before prediction'
        ]
    })
    st.dataframe(feat_df, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — SINGLE TRANSACTION
# ══════════════════════════════════════════════════════════════════════════════
elif "Single Transaction" in page:
    st.markdown('<div class="page-title">🔍 Single Transaction Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Enter all 30 features manually to get an instant fraud prediction</div>', unsafe_allow_html=True)

    if load_errors:
        model_error_banner()
    else:
        with st.form("single_pred_form"):
            st.markdown('<div class="section-header">Transaction Features — V1 to V28</div>', unsafe_allow_html=True)

            v_vals = []
            cols_per_row = 7
            v_list = list(range(1, 29))
            for row_start in range(0, 28, cols_per_row):
                cols = st.columns(cols_per_row)
                for j, vi in enumerate(v_list[row_start:row_start+cols_per_row]):
                    val = cols[j].number_input(f"V{vi}", value=0.0, format="%.6f", key=f"v{vi}")
                    v_vals.append(val)

            st.markdown('<div class="section-header">Transaction Details</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            amount = c1.number_input("Transaction Amount ($)", min_value=0.0, value=100.0, format="%.2f")
            time   = c2.number_input("Transaction Time (seconds)", min_value=0.0, value=50000.0, format="%.1f")

            submitted = st.form_submit_button("🔍 Predict Transaction", use_container_width=True)

        if submitted:
            input_df = preprocess_row(amount, time, v_vals)
            pred      = model.predict(input_df)[0]
            prob      = model.predict_proba(input_df)[0][1]
            risk, risk_cls, risk_icon = get_risk_level(prob)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-header">Prediction Result</div>', unsafe_allow_html=True)

            if pred == 1:
                st.markdown(f'<div class="alert-fraud">⚠️ FRAUDULENT TRANSACTION DETECTED</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="alert-legit">✅ LEGITIMATE TRANSACTION</div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            r1, r2, r3, r4 = st.columns(4)
            r1.markdown(f"""<div class='metric-card'><div class='label'>Prediction</div>
                <div class='value {"danger" if pred==1 else "success"}' style='font-size:18px'>
                {"FRAUD" if pred==1 else "NORMAL"}</div></div>""", unsafe_allow_html=True)
            r2.markdown(f"""<div class='metric-card'><div class='label'>Fraud Probability</div>
                <div class='value {"danger" if prob>0.5 else "success"}'>{prob*100:.2f}%</div></div>""", unsafe_allow_html=True)
            r3.markdown(f"""<div class='metric-card'><div class='label'>Confidence</div>
                <div class='value'>{max(prob, 1-prob)*100:.2f}%</div></div>""", unsafe_allow_html=True)
            r4.markdown(f"""<div class='metric-card'><div class='label'>Risk Level</div>
                <div style='margin-top:12px'><span class='risk-badge {risk_cls}'>{risk_icon} {risk}</span></div></div>""",
                unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            gc, ic = st.columns([1, 1])
            with gc:
                st.plotly_chart(gauge_chart(prob), use_container_width=True)
            with ic:
                st.markdown("<br><br>", unsafe_allow_html=True)
                st.markdown(f"""
                <div style='background:#0f1a2e; border:1px solid #1e3a5f; border-radius:12px; padding:24px'>
                    <div class='section-header' style='margin-top:0'>Risk Guide</div>
                    <div style='font-size:13px; line-height:2'>
                        <span class='risk-badge risk-low'>🟢 Low Risk</span>&nbsp; 0% – 20%<br>
                        <span class='risk-badge risk-medium'>🟡 Medium Risk</span>&nbsp; 20% – 50%<br>
                        <span class='risk-badge risk-high'>🔴 High Risk</span>&nbsp; 50% – 80%<br>
                        <span class='risk-badge risk-critical'>🔴 Critical Risk</span>&nbsp; 80% – 100%
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — BULK CSV
# ══════════════════════════════════════════════════════════════════════════════
elif "Bulk CSV" in page:
    st.markdown('<div class="page-title">📂 Bulk CSV Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Upload a CSV file with multiple transactions for batch prediction</div>', unsafe_allow_html=True)

    if load_errors:
        model_error_banner()
    else:
        st.info("📋 Required columns: `V1` to `V28`, `Amount`, `Time`")
        uploaded_csv = st.file_uploader("Upload CSV File", type=['csv'])

        if uploaded_csv:
            df_raw = pd.read_csv(uploaded_csv)
            st.markdown(f'<div class="section-header">Uploaded Data — {len(df_raw)} Rows</div>', unsafe_allow_html=True)
            st.dataframe(df_raw.head(5), use_container_width=True)

            required = [f'V{i}' for i in range(1,29)] + ['Amount', 'Time']
            missing  = [c for c in required if c not in df_raw.columns]

            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                with st.spinner("Running predictions on all rows..."):
                    X_proc = preprocess_df(df_raw)
                    preds  = model.predict(X_proc)
                    probs  = model.predict_proba(X_proc)[:, 1]

                df_raw['Prediction']        = ['FRAUD' if p==1 else 'NORMAL' for p in preds]
                df_raw['Fraud_Probability'] = (probs * 100).round(2)
                df_raw['Risk_Level']        = [get_risk_level(p)[0] for p in probs]

                fraud_count  = int(preds.sum())
                normal_count = len(preds) - fraud_count
                fraud_pct    = fraud_count / len(preds) * 100

                st.markdown('<div class="section-header">Summary</div>', unsafe_allow_html=True)
                m1,m2,m3,m4 = st.columns(4)
                m1.markdown(f"<div class='metric-card'><div class='label'>Total Records</div><div class='value'>{len(preds)}</div></div>", unsafe_allow_html=True)
                m2.markdown(f"<div class='metric-card'><div class='label'>Fraud Detected</div><div class='value danger'>{fraud_count}</div></div>", unsafe_allow_html=True)
                m3.markdown(f"<div class='metric-card'><div class='label'>Legitimate</div><div class='value success'>{normal_count}</div></div>", unsafe_allow_html=True)
                m4.markdown(f"<div class='metric-card'><div class='label'>Fraud Rate</div><div class='value {'danger' if fraud_pct>10 else 'warning'}'>{fraud_pct:.1f}%</div></div>", unsafe_allow_html=True)

                st.markdown('<div class="section-header">Charts</div>', unsafe_allow_html=True)
                ch1, ch2 = st.columns(2)

                with ch1:
                    fig_pie = go.Figure(go.Pie(
                        labels=['Legitimate','Fraud'],
                        values=[normal_count, fraud_count],
                        hole=0.55,
                        marker_colors=['#34d399','#f87171'],
                        textfont_color='white'
                    ))
                    fig_pie.update_layout(
                        title='Prediction Distribution',
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font={'color':'#cbd5e1'},
                        legend={'font':{'color':'#cbd5e1'}},
                        height=300, margin=dict(t=40,b=0,l=0,r=0)
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

                with ch2:
                    fig_hist = px.histogram(
                        df_raw, x='Fraud_Probability', nbins=30,
                        color_discrete_sequence=['#38bdf8'],
                        title='Fraud Probability Distribution'
                    )
                    st.plotly_chart(dark_fig(fig_hist), use_container_width=True)

                st.markdown('<div class="section-header">Full Results Table</div>', unsafe_allow_html=True)
                st.dataframe(df_raw, use_container_width=True)

                csv_out = df_raw.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Download Results CSV", csv_out,
                                   file_name="fraud_predictions.csv", mime="text/csv")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — PDF ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif "PDF Analysis" in page:
    st.markdown('<div class="page-title">📄 PDF Transaction Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Upload a PDF containing transaction table data for fraud analysis</div>', unsafe_allow_html=True)

    if load_errors:
        model_error_banner()
    else:
        try:
            import pdfplumber
        except ImportError:
            st.error("pdfplumber not installed. Run: `pip install pdfplumber`")
            st.stop()

        st.info("📋 The PDF must contain a table with columns: `V1` to `V28`, `Amount`, `Time`")
        uploaded_pdf = st.file_uploader("Upload PDF File", type=['pdf'])

        if uploaded_pdf:
            with st.spinner("Extracting table from PDF..."):
                try:
                    with pdfplumber.open(uploaded_pdf) as pdf:
                        all_tables = []
                        for page in pdf.pages:
                            tables = page.extract_tables()
                            for table in tables:
                                if table and len(table) > 1:
                                    headers = table[0]
                                    rows    = table[1:]
                                    df_page = pd.DataFrame(rows, columns=headers)
                                    all_tables.append(df_page)

                    if not all_tables:
                        st.error("No tables found in the PDF. Please ensure the PDF has tabular data.")
                        st.stop()

                    df_pdf = pd.concat(all_tables, ignore_index=True)

                    # Convert to numeric
                    for col in df_pdf.columns:
                        df_pdf[col] = pd.to_numeric(df_pdf[col], errors='coerce')
                    df_pdf.dropna(inplace=True)

                    st.success(f"✅ Extracted {len(df_pdf)} rows from PDF")
                    st.dataframe(df_pdf.head(), use_container_width=True)

                    required = [f'V{i}' for i in range(1,29)] + ['Amount','Time']
                    missing  = [c for c in required if c not in df_pdf.columns]

                    if missing:
                        st.error(f"Missing columns in PDF table: {missing}")
                    else:
                        X_proc = preprocess_df(df_pdf)
                        preds  = model.predict(X_proc)
                        probs  = model.predict_proba(X_proc)[:, 1]

                        df_pdf['Prediction']        = ['FRAUD' if p==1 else 'NORMAL' for p in preds]
                        df_pdf['Fraud_Probability'] = (probs * 100).round(2)

                        fraud_count  = int(preds.sum())
                        normal_count = len(preds) - fraud_count

                        m1,m2,m3,m4 = st.columns(4)
                        m1.markdown(f"<div class='metric-card'><div class='label'>Total Transactions</div><div class='value'>{len(preds)}</div></div>", unsafe_allow_html=True)
                        m2.markdown(f"<div class='metric-card'><div class='label'>Fraud</div><div class='value danger'>{fraud_count}</div></div>", unsafe_allow_html=True)
                        m3.markdown(f"<div class='metric-card'><div class='label'>Legitimate</div><div class='value success'>{normal_count}</div></div>", unsafe_allow_html=True)
                        m4.markdown(f"<div class='metric-card'><div class='label'>Fraud Rate</div><div class='value warning'>{fraud_count/len(preds)*100:.1f}%</div></div>", unsafe_allow_html=True)

                        st.markdown('<div class="section-header">Results</div>', unsafe_allow_html=True)
                        st.dataframe(df_pdf, use_container_width=True)

                        csv_out = df_pdf.to_csv(index=False).encode('utf-8')
                        st.download_button("⬇️ Download PDF Analysis Results",
                                           csv_out, file_name="pdf_fraud_results.csv", mime="text/csv")

                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif "Analytics" in page:
    st.markdown('<div class="page-title">📊 Analytics Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Model insights and feature analysis</div>', unsafe_allow_html=True)

    if load_errors:
        model_error_banner()
    else:
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feat_imp_df = pd.DataFrame({
                'Feature': FEATURE_COLS,
                'Importance': importances
            }).sort_values('Importance', ascending=True).tail(20)

            fig_imp = go.Figure(go.Bar(
                x=feat_imp_df['Importance'],
                y=feat_imp_df['Feature'],
                orientation='h',
                marker_color='#38bdf8',
                marker_line_color='#0ea5e9',
                marker_line_width=0.5
            ))
            fig_imp.update_layout(
                title='Top 20 Feature Importances (from model.feature_importances_)',
                xaxis_title='Importance Score',
                yaxis_title='Feature',
                height=500
            )
            st.plotly_chart(dark_fig(fig_imp), use_container_width=True)

            # Top features table
            st.markdown('<div class="section-header">Feature Importance Table</div>', unsafe_allow_html=True)
            top_feats = pd.DataFrame({
                'Feature': FEATURE_COLS,
                'Importance': importances,
                'Importance (%)': (importances / importances.sum() * 100).round(3)
            }).sort_values('Importance', ascending=False).reset_index(drop=True)
            top_feats.index += 1
            st.dataframe(top_feats, use_container_width=True)
        else:
            st.warning("Feature importances not available for this model type.")

        # Sample gauge examples
        st.markdown('<div class="section-header">Risk Level Examples</div>', unsafe_allow_html=True)
        g1, g2, g3, g4 = st.columns(4)
        for col, (label, prob) in zip([g1,g2,g3,g4],[
            ("Low Risk Example", 0.10),
            ("Medium Risk Example", 0.35),
            ("High Risk Example", 0.70),
            ("Critical Risk Example", 0.92)
        ]):
            col.plotly_chart(gauge_chart(prob), use_container_width=True)
            col.markdown(f"<div style='text-align:center; font-size:12px; color:#64748b'>{label}</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif "Model Performance" in page:
    st.markdown('<div class="page-title">📈 Model Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Upload your test dataset to evaluate model performance metrics</div>', unsafe_allow_html=True)

    if load_errors:
        model_error_banner()
    else:
        st.info("Upload a CSV with columns V1–V28, Amount, Time, and **Class** (0=Normal, 1=Fraud) to evaluate performance.")
        perf_file = st.file_uploader("Upload Test CSV with Class column", type=['csv'])

        if perf_file:
            from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                         f1_score, roc_auc_score, confusion_matrix,
                                         roc_curve, precision_recall_curve)

            df_test = pd.read_csv(perf_file)

            if 'Class' not in df_test.columns:
                st.error("Column 'Class' not found. Please include ground truth labels.")
            else:
                y_true  = df_test['Class']
                X_proc  = preprocess_df(df_test.drop('Class', axis=1))
                y_pred  = model.predict(X_proc)
                y_prob  = model.predict_proba(X_proc)[:, 1]

                acc  = accuracy_score(y_true, y_pred)
                prec = precision_score(y_true, y_pred)
                rec  = recall_score(y_true, y_pred)
                f1   = f1_score(y_true, y_pred)
                auc  = roc_auc_score(y_true, y_prob)

                st.markdown('<div class="section-header">Performance Metrics</div>', unsafe_allow_html=True)
                m1,m2,m3,m4,m5 = st.columns(5)
                for col, label, val in zip([m1,m2,m3,m4,m5],
                    ["Accuracy","Precision","Recall","F1 Score","ROC AUC"],
                    [acc, prec, rec, f1, auc]):
                    col.markdown(f"""<div class='metric-card'>
                        <div class='label'>{label}</div>
                        <div class='value' style='font-size:24px'>{val:.4f}</div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                ch1, ch2 = st.columns(2)

                # Confusion matrix
                with ch1:
                    cm = confusion_matrix(y_true, y_pred)
                    fig_cm = go.Figure(go.Heatmap(
                        z=cm, x=['Predicted Normal','Predicted Fraud'],
                        y=['Actual Normal','Actual Fraud'],
                        colorscale=[[0,'#0f1a2e'],[1,'#0ea5e9']],
                        text=cm, texttemplate="%{text}",
                        textfont={"size":18, "color":"white"}
                    ))
                    fig_cm.update_layout(title='Confusion Matrix', height=350)
                    st.plotly_chart(dark_fig(fig_cm), use_container_width=True)

                # ROC curve
                with ch2:
                    fpr, tpr, _ = roc_curve(y_true, y_prob)
                    fig_roc = go.Figure()
                    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f'ROC (AUC={auc:.4f})',
                                                  line=dict(color='#38bdf8', width=2)))
                    fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], name='Random',
                                                  line=dict(color='#475569', dash='dash')))
                    fig_roc.update_layout(title='ROC Curve', xaxis_title='False Positive Rate',
                                          yaxis_title='True Positive Rate', height=350)
                    st.plotly_chart(dark_fig(fig_roc), use_container_width=True)

                # Precision-Recall curve
                prec_c, rec_c, _ = precision_recall_curve(y_true, y_prob)
                fig_pr = go.Figure()
                fig_pr.add_trace(go.Scatter(x=rec_c, y=prec_c, fill='tozeroy',
                                             fillcolor='rgba(56,189,248,0.1)',
                                             line=dict(color='#38bdf8', width=2),
                                             name='Precision-Recall'))
                fig_pr.update_layout(title='Precision-Recall Curve',
                                      xaxis_title='Recall', yaxis_title='Precision', height=350)
                st.plotly_chart(dark_fig(fig_pr), use_container_width=True)
        else:
            st.markdown("""
            <div style='background:#0f1a2e; border:1px solid #1e3a5f; border-radius:12px; padding:28px; text-align:center; color:#475569'>
                <div style='font-size:40px; margin-bottom:12px'>📊</div>
                <div style='font-size:16px; color:#64748b'>Upload a test CSV with a <b style='color:#38bdf8'>Class</b> column to view performance metrics, confusion matrix, ROC curve and Precision-Recall curve.</div>
            </div>
            """, unsafe_allow_html=True)
