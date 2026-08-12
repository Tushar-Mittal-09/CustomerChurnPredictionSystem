"""
ChurnGuard AI — Interactive Streamlit Intelligence Dashboard.
Multi-page analytics app for Executive Overview, Real-Time Prediction, Model Performance, and SHAP Explainability.
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import PIL.Image
import seaborn as sns
import streamlit as st

# Import backend modules
from src.models.predict import load_serialized_pipeline, predict_churn_risk
from src.utils.config import get_project_root, load_config

# Streamlit Page Setup
st.set_page_config(
    page_title="ChurnGuard AI — Retention Intelligence Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #64748B;
    }
    .risk-high {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 0.4rem 0.8rem;
        border-radius: 6px;
        font-weight: 700;
    }
    .risk-medium {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 0.4rem 0.8rem;
        border-radius: 6px;
        font-weight: 700;
    }
    .risk-low {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 0.4rem 0.8rem;
        border-radius: 6px;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def get_cached_metadata():
    root = get_project_root()
    meta_path = root / "models/model_metadata.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data
def get_dataset():
    root = get_project_root()
    raw_path = root / "data/raw/Telco_Customer_Churn.csv"
    if raw_path.exists():
        return pd.read_csv(raw_path)
    return pd.DataFrame()


# Navigation Sidebar
st.sidebar.image("https://img.icons8.com/color/96/000000/shield.png", width=64)
st.sidebar.title("ChurnGuard AI")
st.sidebar.markdown("**Retention Intelligence Platform**")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation Menu",
    [
        "📊 Executive Overview",
        "🎯 Customer Predictor",
        "📈 Model Performance",
        "🔍 SHAP Explainability"
    ]
)

st.sidebar.divider()
meta = get_cached_metadata()
st.sidebar.caption(f"**Model**: {meta.get('model_name', 'ChurnGuard Model')}")
st.sidebar.caption(f"**Version**: v{meta.get('model_version', '1.0.0')}")
st.sidebar.caption(f"**Decision Threshold**: τ = {meta.get('threshold', 0.35)}")

root_dir = get_project_root()

# ==========================================
# PAGE 1: EXECUTIVE OVERVIEW
# ==========================================
if page == "📊 Executive Overview":
    st.markdown('<div class="main-header">Executive Churn Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">High-level customer churn metrics, portfolio risk distribution, and retention strategy KPIs.</div>', unsafe_allow_html=True)

    df_raw = get_dataset()
    total_customers = len(df_raw) if not df_raw.empty else 7043
    actual_churn_rate = 26.54

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{total_customers:,}</div><div class="metric-label">Total Customer Portfolio</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{actual_churn_rate:.2f}%</div><div class="metric-label">Actual Churn Rate</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-value">1,869</div><div class="metric-label">Total Churned Customers</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="metric-value">τ = 0.35</div><div class="metric-label">Optimal Risk Threshold</div></div>', unsafe_allow_html=True)

    st.markdown("### Key Business Insights")
    c1, c2 = st.columns(2)

    with c1:
        fig_target = root_dir / "reports/figures/target_distribution.png"
        if fig_target.exists():
            st.image(str(fig_target), caption="Customer Churn Class Distribution", use_container_width=True)
        else:
            st.info("Distribution plot available in reports/figures/target_distribution.png")

    with c2:
        fig_contract = root_dir / "reports/figures/contract_vs_churn.png"
        if fig_contract.exists():
            st.image(str(fig_contract), caption="Churn Rate by Contract Type", use_container_width=True)
        else:
            st.info("Contract plot available in reports/figures/contract_vs_churn.png")

# ==========================================
# PAGE 2: REAL-TIME CUSTOMER PREDICTOR
# ==========================================
elif page == "🎯 Customer Predictor":
    st.markdown('<div class="main-header">Real-Time Customer Churn Risk Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Input customer subscription details to calculate churn probability, risk level, top factors, and recommended action.</div>', unsafe_allow_html=True)

    with st.form("prediction_form"):
        st.markdown("#### Customer Subscription Profile")
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            gender = st.selectbox("Gender", ["Female", "Male"])
            SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
            Partner = st.selectbox("Partner", ["No", "Yes"])
            Dependents = st.selectbox("Dependents", ["No", "Yes"])
            tenure = st.slider("Tenure (Months)", 0, 72, 8)
            PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
            MultipleLines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])

        with col_b:
            InternetService = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
            OnlineSecurity = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            OnlineBackup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            DeviceProtection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            TechSupport = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            StreamingTV = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            StreamingMovies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

        with col_c:
            Contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
            PaymentMethod = st.selectbox("Payment Method", [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ])
            MonthlyCharges = st.number_input("Monthly Charges ($)", 18.0, 150.0, 89.5)
            TotalCharges = st.number_input("Total Charges ($)", 0.0, 10000.0, float(tenure * MonthlyCharges))

        submit_btn = st.form_submit_button("🔍 Calculate Churn Risk Score", use_container_width=True)

    if submit_btn:
        input_data = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": SeniorCitizen,
            "Partner": Partner,
            "Dependents": Dependents,
            "tenure": tenure,
            "PhoneService": PhoneService,
            "MultipleLines": MultipleLines,
            "InternetService": InternetService,
            "OnlineSecurity": OnlineSecurity,
            "OnlineBackup": OnlineBackup,
            "DeviceProtection": DeviceProtection,
            "TechSupport": TechSupport,
            "StreamingTV": StreamingTV,
            "StreamingMovies": StreamingMovies,
            "Contract": Contract,
            "PaperlessBilling": PaperlessBilling,
            "PaymentMethod": PaymentMethod,
            "MonthlyCharges": MonthlyCharges,
            "TotalCharges": TotalCharges
        }])

        res = predict_churn_risk(input_data)[0]

        st.divider()
        st.markdown("### Prediction Results")

        res_col1, res_col2, res_col3 = st.columns(3)

        with res_col1:
            st.metric(
                label="Churn Probability",
                value=f"{res['churn_probability'] * 100:.1f}%",
                delta="HIGH RISK" if res["prediction"] == 1 else "LOW RISK",
                delta_color="inverse"
            )

        with res_col2:
            badge_class = f"risk-{res['risk_level'].lower()}"
            st.markdown(f"**Risk Level Tier**")
            st.markdown(f'<span class="{badge_class}">{res["risk_level"]} RISK</span>', unsafe_allow_html=True)

        with res_col3:
            st.metric(label="Decision Threshold", value=f"τ = {res['threshold']}")

        st.markdown("#### Key Risk Factor Attribution")
        for factor in res["top_risk_factors"]:
            st.markdown(f"- ⚠️ {factor}")

        st.markdown("#### Recommended Retention Action")
        st.info(f"💡 **Business Action**: {res['recommended_action']}")

# ==========================================
# PAGE 3: MODEL PERFORMANCE ANALYTICS
# ==========================================
elif page == "📈 Model Performance":
    st.markdown('<div class="main-header">Model Performance & Evaluation Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Comparative evaluation metrics, confusion matrix, ROC/PR curves, and business loss threshold curve.</div>', unsafe_allow_html=True)

    metrics_path = root_dir / "reports/metrics/model_comparison.json"
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            comp_metrics = json.load(f)
        df_comp = pd.DataFrame(comp_metrics).T
        st.markdown("### Model Metric Comparison Table")
        st.dataframe(df_comp.style.highlight_max(axis=0, color="#D1FAE5"), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        roc_path = root_dir / "reports/figures/model_comparison_roc.png"
        if roc_path.exists():
            st.image(str(roc_path), caption="Model Comparison — ROC Curves", use_container_width=True)
    with c2:
        pr_path = root_dir / "reports/figures/model_comparison_pr.png"
        if pr_path.exists():
            st.image(str(pr_path), caption="Model Comparison — Precision-Recall Curves", use_container_width=True)

    st.markdown("### Business Cost Threshold Optimization Curve")
    thresh_path = root_dir / "reports/figures/threshold_optimization.png"
    if thresh_path.exists():
        st.image(str(thresh_path), caption="Threshold vs Total Business Cost Curve", use_container_width=True)

# ==========================================
# PAGE 4: SHAP EXPLAINABILITY
# ==========================================
elif page == "🔍 SHAP Explainability":
    st.markdown('<div class="main-header">SHAP Model Explainability</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Global SHAP feature attribution plots explaining model predictions across the entire customer base.</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        shap_bar = root_dir / "reports/figures/shap_summary_bar.png"
        if shap_bar.exists():
            st.image(str(shap_bar), caption="Global SHAP Feature Importance (Bar Plot)", use_container_width=True)
    with c2:
        shap_bee = root_dir / "reports/figures/shap_beeswarm.png"
        if shap_bee.exists():
            st.image(str(shap_bee), caption="Global SHAP Feature Attribution (Beeswarm)", use_container_width=True)
