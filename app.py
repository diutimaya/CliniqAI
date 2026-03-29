"""
CliniqAI — Production-Ready AI Healthcare Dashboard
Single-service Streamlit application. No separate backend.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Page config (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="CliniqAI",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Lazy imports with spinner ──────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training risk model…")
def load_risk_model():
    from models.risk_model import get_model
    return get_model()

@st.cache_resource(show_spinner=False)
def load_nlp():
    from models.nlp_model import extract_entities, get_entity_summary, SAMPLE_NOTES
    return extract_entities, get_entity_summary, SAMPLE_NOTES

@st.cache_resource(show_spinner=False)
def load_vitals():
    from models.vitals_model import detect_anomalies, generate_sample_vitals, VITAL_RANGES, SAMPLE_SCENARIOS
    return detect_anomalies, generate_sample_vitals, VITAL_RANGES, SAMPLE_SCENARIOS

# Pre-warm
load_risk_model()

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Dark base */
.stApp {
    background: #0b0f1a;
    color: #e2e8f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0d1321 !important;
    border-right: 1px solid #1e2a42;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #111827;
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
    border: 1px solid #1e2a42;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #94a3b8;
    font-weight: 600;
    font-size: 14px;
    padding: 10px 24px;
}
.stTabs [aria-selected="true"] {
    background: #1d4ed8 !important;
    color: #fff !important;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #111827 0%, #0f172a 100%);
    border: 1px solid #1e2a42;
    border-radius: 16px;
    padding: 20px 24px;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #1d4ed8; }
.metric-card .label {
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #64748b;
    margin-bottom: 6px;
}
.metric-card .value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 28px;
    font-weight: 600;
    color: #f1f5f9;
}
.metric-card .sub {
    font-size: 12px;
    color: #475569;
    margin-top: 4px;
}

/* Risk badge */
.risk-badge {
    display: inline-block;
    padding: 8px 28px;
    border-radius: 50px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 22px;
    font-weight: 700;
    letter-spacing: 2px;
}
.risk-LOW    { background: #052e16; color: #22c55e; border: 2px solid #22c55e33; }
.risk-MODERATE { background: #422006; color: #f59e0b; border: 2px solid #f59e0b33; }
.risk-HIGH   { background: #450a0a; color: #ef4444; border: 2px solid #ef444433; }

/* Entity chips */
.entity-chip {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 13px;
    margin: 3px;
    font-family: 'IBM Plex Mono', monospace;
}
.chip-symptoms   { background: #1e1b4b; color: #a5b4fc; border: 1px solid #3730a3; }
.chip-conditions { background: #450a0a; color: #fca5a5; border: 1px solid #7f1d1d; }
.chip-medications{ background: #052e16; color: #86efac; border: 1px solid #14532d; }
.chip-lab_values { background: #0c1a2e; color: #93c5fd; border: 1px solid #1e3a5f; }

/* Status badge */
.status-NORMAL   { color: #22c55e; }
.status-WARNING  { color: #f59e0b; }
.status-CRITICAL { color: #ef4444; }

/* Section header */
.section-header {
    font-size: 13px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #475569;
    margin: 20px 0 10px;
    border-bottom: 1px solid #1e2a42;
    padding-bottom: 6px;
}

/* Inputs */
.stSlider > div > div > div > div { background: #1d4ed8 !important; }
.stSelectbox > div > div { background: #111827 !important; border-color: #1e2a42 !important; }
.stTextArea > div > div > textarea {
    background: #111827 !important;
    border-color: #1e2a42 !important;
    color: #e2e8f0 !important;
    font-family: 'IBM Plex Mono', monospace;
}
.stButton > button {
    background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 28px;
    font-weight: 600;
    font-size: 15px;
    width: 100%;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.9; }

/* Logo */
.logo {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 24px;
    font-weight: 700;
    background: linear-gradient(90deg, #60a5fa, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
}
.logo-sub {
    font-size: 11px;
    color: #475569;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: -4px;
}

/* Info box */
.info-box {
    background: #0f172a;
    border-left: 3px solid #1d4ed8;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    font-size: 13px;
    color: #94a3b8;
    margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="logo">CliniqAI</div>', unsafe_allow_html=True)
    st.markdown('<div class="logo-sub">Clinical Intelligence Platform</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div class="info-box">
    AI-powered clinical decision support. For demonstration purposes only. 
    Not for clinical use.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    # Navigation hint
    st.markdown("#### Navigation")
    st.markdown("""
    - **Tab 1** — Patient Risk Prediction  
    - **Tab 2** — Clinical Note NLP  
    - **Tab 3** — Vitals Anomaly Detection  
    """)
    st.markdown("---")
    st.markdown('<div style="font-size:11px;color:#334155;text-align:center">CliniqAI v1.0 · Built with Streamlit</div>', unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "Patient Risk Prediction",
    "Clinical Note NLP",
    "Vitals Anomaly Detection",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Patient Risk Prediction
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    from models.risk_model import predict_risk, RISK_COLORS

    st.markdown("## Patient Risk Stratification")
    st.markdown("Enter patient vitals and lab values to predict clinical deterioration risk.")

    st.markdown("---")

    # Input form
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="section-header">Demographics</div>', unsafe_allow_html=True)
        age = st.slider("Age (years)", 18, 100, 50)
        gender = st.selectbox("Gender", ["Female", "Male"])
        los = st.slider("Length of Stay (days)", 0, 30, 3)

    with col2:
        st.markdown('<div class="section-header">Lab Values</div>', unsafe_allow_html=True)
        glucose    = st.slider("Glucose (mg/dL)", 50, 500, 110)
        wbc        = st.slider("WBC (K/μL)", 1.0, 40.0, 8.5, step=0.1)
        creatinine = st.slider("Creatinine (mg/dL)", 0.4, 15.0, 1.0, step=0.1)
        diag_count = st.slider("Diagnoses Count", 1, 15, 2)
        lab_tests  = st.slider("Lab Tests Ordered", 1, 25, 5)

    with col3:
        st.markdown('<div class="section-header">Vitals</div>', unsafe_allow_html=True)
        heart_rate   = st.slider("Heart Rate (bpm)", 30, 180, 78)
        systolic_bp  = st.slider("Systolic BP (mmHg)", 60, 240, 120)
        spo2         = st.slider("SpO2 (%)", 70, 100, 97)

    st.markdown("")
    run_btn = st.button("Run Risk Assessment", key="risk_btn")

    if run_btn:
        with st.spinner("Analyzing patient data…"):
            label, prob, all_probs, importances = predict_risk(
                age, gender, glucose, wbc, creatinine,
                heart_rate, systolic_bp, spo2,
                los, diag_count, lab_tests
            )

        st.markdown("---")
        st.markdown("### Assessment Results")

        # Risk badge
        r1, r2, r3 = st.columns([1, 1.2, 1])
        with r2:
            st.markdown(
                f'<div style="text-align:center;padding:24px 0">'
                f'<div style="font-size:12px;color:#64748b;letter-spacing:2px;margin-bottom:10px">RISK CLASSIFICATION</div>'
                f'<span class="risk-badge risk-{label}">{label}</span>'
                f'<div style="margin-top:12px;font-family:IBM Plex Mono,monospace;font-size:14px;color:#94a3b8">'
                f'Confidence: {prob:.1%}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Probability breakdown
        m1, m2, m3 = st.columns(3)
        for col, lbl in zip([m1, m2, m3], ["LOW", "MODERATE", "HIGH"]):
            c = RISK_COLORS[lbl]
            with col:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="label">{lbl} Risk</div>'
                    f'<div class="value" style="color:{c}">{all_probs[lbl]:.1%}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("")

        # Charts
        ch1, ch2 = st.columns(2)

        with ch1:
            # Probability gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(all_probs["HIGH"] * 100, 1),
                title={"text": "High-Risk Probability", "font": {"color": "#94a3b8", "size": 14}},
                number={"suffix": "%", "font": {"color": "#f1f5f9", "size": 32}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#475569"},
                    "bar": {"color": RISK_COLORS["HIGH"]},
                    "bgcolor": "#111827",
                    "bordercolor": "#1e2a42",
                    "steps": [
                        {"range": [0, 33], "color": "#052e16"},
                        {"range": [33, 66], "color": "#422006"},
                        {"range": [66, 100], "color": "#450a0a"},
                    ],
                    "threshold": {
                        "line": {"color": "#ef4444", "width": 3},
                        "thickness": 0.75,
                        "value": 66,
                    },
                },
            ))
            fig_gauge.update_layout(
                height=260, paper_bgcolor="#0b0f1a", plot_bgcolor="#0b0f1a",
                font_color="#94a3b8", margin=dict(t=40, b=10, l=20, r=20),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with ch2:
            # Feature importance
            top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:8]
            feat_names  = [f[0].replace("_", " ").title() for f in top_features]
            feat_values = [f[1] for f in top_features]

            fig_imp = go.Figure(go.Bar(
                x=feat_values[::-1], y=feat_names[::-1],
                orientation="h",
                marker_color="#1d4ed8",
                marker_line_color="#1e40af",
            ))
            fig_imp.update_layout(
                title={"text": "Feature Importance", "font": {"color": "#94a3b8", "size": 14}},
                height=260, paper_bgcolor="#0b0f1a", plot_bgcolor="#111827",
                font_color="#94a3b8", xaxis=dict(gridcolor="#1e2a42"),
                yaxis=dict(gridcolor="#1e2a42"),
                margin=dict(t=40, b=10, l=10, r=10),
            )
            st.plotly_chart(fig_imp, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Clinical Note NLP
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    extract_entities, get_entity_summary, _ = load_nlp()

    st.markdown("## Clinical Note Entity Extraction")
    st.markdown("Paste a clinical note to extract structured entities: symptoms, conditions, medications, and lab values.")

    note_text = st.text_area(
        "Clinical Note",
        value="",
        height=200,
        placeholder="e.g. 65-year-old male with type 2 diabetes presenting with fever and shortness of breath…",
        key="note_input",
    )

    nlp_btn = st.button("Extract Entities", key="nlp_btn")

    if nlp_btn:
        if not note_text.strip():
            st.warning("Please paste a clinical note to analyze.")
        else:
            with st.spinner("Analyzing clinical note…"):
                entities = extract_entities(note_text)

            st.markdown("---")
            st.markdown("### Extracted Entities")
            st.markdown(f'<div class="info-box">{get_entity_summary(entities)}</div>', unsafe_allow_html=True)

            CATEGORIES = {
                "symptoms":    ("Symptoms",    "chip-symptoms"),
                "conditions":  ("Conditions",  "chip-conditions"),
                "medications": ("Medications", "chip-medications"),
                "lab_values":  ("Lab Values",  "chip-lab_values"),
            }

            ent_cols = st.columns(2)
            col_idx  = 0

            for key, (title, chip_class) in CATEGORIES.items():
                items = entities[key]
                with ent_cols[col_idx % 2]:
                    st.markdown(f"#### {title}")
                    if items:
                        chips = "".join(
                            f'<span class="entity-chip {chip_class}">{item}</span>'
                            for item in items
                        )
                        st.markdown(chips, unsafe_allow_html=True)
                    else:
                        st.markdown('<span style="color:#475569;font-size:13px">None detected</span>', unsafe_allow_html=True)
                    st.markdown("")
                col_idx += 1

            # Summary bar chart
            st.markdown("---")
            counts = {CATEGORIES[k][0]: len(v) for k, v in entities.items()}
            fig_bar = go.Figure(go.Bar(
                x=list(counts.keys()),
                y=list(counts.values()),
                marker_color=["#818cf8", "#f87171", "#86efac", "#93c5fd"],
                text=list(counts.values()),
                textposition="outside",
                textfont=dict(color="#94a3b8"),
            ))
            fig_bar.update_layout(
                title={"text": "Entity Count by Category", "font": {"color": "#94a3b8"}},
                height=280, paper_bgcolor="#0b0f1a", plot_bgcolor="#111827",
                font_color="#94a3b8",
                xaxis=dict(gridcolor="#1e2a42"),
                yaxis=dict(gridcolor="#1e2a42"),
                margin=dict(t=50, b=10, l=10, r=10),
                showlegend=False,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Vitals Anomaly Detection
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    detect_anomalies, generate_sample_vitals, VITAL_RANGES, SAMPLE_SCENARIOS = load_vitals()

    st.markdown("## Vitals Anomaly Detection")
    st.markdown("Upload or generate a time-series vitals dataset to detect anomalies using rolling z-score and clinical thresholds.")

    v1, v2 = st.columns([1, 2])
    with v1:
        data_source = st.radio("Data Source", ["Use Sample Data", "Upload CSV"], key="vitals_src")
    with v2:
        if data_source == "Use Sample Data":
            scenario = st.selectbox(
                "Select Scenario",
                list(SAMPLE_SCENARIOS.keys()),
                key="vitals_scenario"
            )
            n_hours = st.slider("Observation Window (hours)", 6, 72, 24, key="vitals_hours")

    # Load data
    df_vitals = None
    if data_source == "Use Sample Data":
        df_vitals = generate_sample_vitals(SAMPLE_SCENARIOS[scenario], n_hours)
    else:
        uploaded = st.file_uploader(
            "Upload CSV (columns: Time, Heart Rate (bpm), Systolic BP (mmHg), SpO2 (%), Temperature (°C))",
            type=["csv"]
        )
        if uploaded:
            try:
                df_vitals = pd.read_csv(uploaded, parse_dates=["Time"])
            except Exception as e:
                st.error(f"Error reading file: {e}")

    analyze_btn = st.button("Analyze Vitals", key="vitals_btn")

    if analyze_btn and df_vitals is not None:
        with st.spinner("Detecting anomalies…"):
            results = detect_anomalies(df_vitals)

        st.markdown("---")
        st.markdown("### Anomaly Detection Results")

        # Status cards
        stat_cols = st.columns(len(results))
        STATUS_ICON = {"NORMAL": "", "WARNING": "!", "CRITICAL": "!!"}
        STATUS_COLOR = {"NORMAL": "#22c55e", "WARNING": "#f59e0b", "CRITICAL": "#ef4444"}

        for col, (vital, res) in zip(stat_cols, results.items()):
            short_name = vital.split("(")[0].strip()
            s = res["status"]
            with col:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="label">{short_name}</div>'
                    f'<div class="value" style="color:{STATUS_COLOR[s]};font-size:22px">{(STATUS_ICON[s] + " ") if STATUS_ICON[s] else ""}{s}</div>'
                    f'<div class="sub">Anomalies: {res["stats"]["anomaly_pct"]}%</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("")

        # Time-series charts
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=list(results.keys()),
            shared_xaxes=False,
            vertical_spacing=0.15,
            horizontal_spacing=0.08,
        )
        VITAL_COLORS = {
            "Heart Rate (bpm)":   "#f87171",
            "Systolic BP (mmHg)": "#60a5fa",
            "SpO2 (%)":           "#86efac",
            "Temperature (°C)":   "#fbbf24",
        }
        positions = [(1,1),(1,2),(2,1),(2,2)]

        for (row, col), (vital, res) in zip(positions, results.items()):
            color   = VITAL_COLORS.get(vital, "#94a3b8")
            times   = df_vitals["Time"]
            values  = df_vitals[vital]
            anoms   = res["anomaly_idx"]
            ranges  = VITAL_RANGES[vital]

            # Normal line
            fig.add_trace(go.Scatter(
                x=times, y=values,
                mode="lines", name=vital,
                line=dict(color=color, width=2),
                showlegend=False,
            ), row=row, col=col)

            # Anomaly markers
            if anoms:
                fig.add_trace(go.Scatter(
                    x=times.iloc[anoms], y=values.iloc[anoms],
                    mode="markers", name="Anomaly",
                    marker=dict(color="#ef4444", size=9, symbol="x"),
                    showlegend=False,
                ), row=row, col=col)

            # Reference lines
            for level, dash, fc in [
                ("low_warning", "dot", "#f59e0b33"),
                ("high_warning", "dot", "#f59e0b33"),
                ("low_critical", "dash", "#ef444433"),
                ("high_critical", "dash", "#ef444433"),
            ]:
                val = ranges[level]
                if 70 <= val <= 200 or vital == "Temperature (°C)":
                    fig.add_hline(
                        y=val, line_dash=dash,
                        line_color="#f59e0b" if "warning" in level else "#ef4444",
                        line_width=1, opacity=0.6, row=row, col=col
                    )

        fig.update_layout(
            height=550, paper_bgcolor="#0b0f1a", plot_bgcolor="#111827",
            font_color="#94a3b8",
            margin=dict(t=50, b=20, l=20, r=20),
        )
        fig.update_xaxes(gridcolor="#1e2a42", showgrid=True)
        fig.update_yaxes(gridcolor="#1e2a42", showgrid=True)
        for ann in fig.layout.annotations:
            ann.font.color = "#94a3b8"

        st.plotly_chart(fig, use_container_width=True)

        # Per-vital stats table
        st.markdown("### Vital Statistics Summary")
        stats_rows = []
        for vital, res in results.items():
            s = res["stats"]
            stats_rows.append({
                "Vital": vital,
                "Mean": s["mean"],
                "Std": s["std"],
                "Min": s["min"],
                "Max": s["max"],
                "Anomaly %": f"{s['anomaly_pct']}%",
                "Status": res["status"],
            })

        stats_df = pd.DataFrame(stats_rows)
        st.dataframe(
            stats_df.set_index("Vital"),
            use_container_width=True,
        )

    elif analyze_btn and df_vitals is None:
        st.warning("Please select or upload vitals data first.")