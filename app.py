import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="DeathByAI — Healthcare Bias Auditor",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

    :root {
        --bg-dark:     #0b0d13;
        --bg-card:     #12151f;
        --bg-card-alt: #181c2a;
        --accent:      #00d4aa;
        --accent2:     #38bdf8;
        --danger:      #ff4b6e;
        --warn:        #f59e0b;
        --text:        #e8eaed;
        --text-muted:  #7f8694;
        --border:      rgba(255,255,255,.06);
        --glow-green:  rgba(0,212,170,.12);
        --glow-red:    rgba(255,75,110,.12);
    }

    html, body, [data-testid="stAppViewContainer"],
    [data-testid="stHeader"], [data-testid="stSidebar"] {
        font-family: 'Inter', sans-serif !important;
    }

    /* ── Hero ───────────────────────────────────────────── */
    .hero-banner {
        position: relative;
        background: linear-gradient(135deg, #090b14 0%, #111627 40%, #0e1a2b 100%);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 2.8rem 2rem 2.2rem;
        margin-bottom: 1.8rem;
        text-align: center;
        overflow: hidden;
    }
    .hero-banner::before {
        content:'';
        position:absolute; inset:0;
        background:
            radial-gradient(circle at 25% 55%, var(--glow-green) 0%, transparent 55%),
            radial-gradient(circle at 75% 35%, var(--glow-red)   0%, transparent 55%);
        pointer-events:none;
    }
    .hero-banner h1 {
        font-size: 2.6rem;
        font-weight: 900;
        letter-spacing: -.02em;
        background: linear-gradient(90deg, var(--accent), var(--accent2), var(--danger));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .hero-banner .subtitle {
        color: var(--text-muted);
        font-size: 1.05rem;
        margin-top: .5rem;
    }

    /* ── Metric card ────────────────────────────────────── */
    .kpi-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1.4rem 1.2rem;
        text-align: center;
        transition: transform .2s ease, box-shadow .2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 36px rgba(0,0,0,.4);
    }
    .kpi-card .kpi-label {
        font-size: .78rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: .07em;
        margin-bottom: .25rem;
    }
    .kpi-card .kpi-value {
        font-size: 1.9rem;
        font-weight: 700;
        color: var(--text);
    }

    /* ── Section header ─────────────────────────────────── */
    .section-hdr {
        font-size: 1.2rem;
        font-weight: 700;
        color: var(--text);
        display: flex;
        align-items: center;
        gap: .45rem;
        margin-bottom: .7rem;
    }

    /* ── Glow divider ───────────────────────────────────── */
    .glow-line {
        height: 2px;
        border: none;
        background: linear-gradient(90deg, transparent, var(--accent), transparent);
        margin: 2.2rem 0;
        opacity: .4;
    }

    /* ── Badge chips ────────────────────────────────────── */
    .chip-ok {
        display: inline-block;
        background: linear-gradient(135deg, #053d2f, #065f46);
        color: #6ee7b7;
        padding: .55rem 1.5rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 1.1rem;
        border: 1px solid rgba(110,231,183,.2);
    }
    .chip-warn {
        display: inline-block;
        background: linear-gradient(135deg, #4a1525, #6b1e30);
        color: #fca5a5;
        padding: .55rem 1.5rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 1.1rem;
        border: 1px solid rgba(252,165,165,.2);
    }

    /* ── Score ring ──────────────────────────────────────── */
    .score-ring {
        width: 150px; height: 150px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto;
        position: relative;
    }
    .score-ring .inner {
        width: 120px; height: 120px;
        border-radius: 50%;
        background: var(--bg-dark);
        display: flex;
        align-items: center;
        justify-content: center;
        flex-direction: column;
    }
    .score-ring .inner .num {
        font-size: 2.2rem;
        font-weight: 800;
    }
    .score-ring .inner .of {
        font-size: .75rem;
        color: var(--text-muted);
    }

    /* hide default Streamlit footer */
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════
# HERO BANNER
# ═══════════════════════════════════════════════════════════════════════════
st.markdown(
    """
    <div class="hero-banner">
        <h1> DeathByAI — Healthcare Bias Auditor</h1>
        <p class="subtitle">
            Upload a healthcare prediction dataset &middot; Detect demographic bias &middot; Ensure model fairness
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 📂 Upload Dataset")
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="Upload a CSV with at least one prediction column and one demographic column.",
    )
    st.markdown("---")
    st.markdown(
        "**Example CSV format:**\n"
        "```\n"
        "age,gender,income_group,prediction\n"
        "65,M,Low,1\n"
        "42,F,Medium,0\n"
        "71,M,High,1\n"
        "56,F,Low,0\n"
        "60,M,Medium,1\n"
        "48,F,Low,0\n"
        "```"
    )
    st.markdown("---")
    


# ═══════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════════════
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # ── 1. Dataset Overview Panel ──────────────────────────────────────────
    st.markdown('<div class="section-hdr">📋 Dataset Overview</div>', unsafe_allow_html=True)
    st.dataframe(df.head(20), use_container_width=True, height=260)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">Total Rows</div>'
            f'<div class="kpi-value">{len(df):,}</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">Columns</div>'
            f'<div class="kpi-value">{df.shape[1]}</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">Missing Values</div>'
            f'<div class="kpi-value">{int(df.isnull().sum().sum()):,}</div></div>',
            unsafe_allow_html=True,
        )
    with c4:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">Numeric Cols</div>'
            f'<div class="kpi-value">{len(numeric_cols)}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="glow-line">', unsafe_allow_html=True)

    # ── 2. Column Selection ────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">⚙️ Configure Audit</div>', unsafe_allow_html=True)

    # Detect columns that contain only binary values (0 and 1)
    binary_cols = [
        c for c in df.columns
        if df[c].dropna().isin([0, 1]).all() and df[c].nunique() <= 2
    ]

    # Fallback: if no binary columns found, show all columns but warn later
    pred_options = binary_cols if binary_cols else df.columns.tolist()

    sel1, sel2 = st.columns(2)
    with sel1:
        default_pred_idx = (
            pred_options.index("prediction")
            if "prediction" in pred_options
            else 0
        )
        prediction_col = st.selectbox(
            "Prediction Column (model output 0 / 1)",
            options=pred_options,
            index=default_pred_idx,
        )
    with sel2:
        other_cols = [c for c in df.columns if c != prediction_col]
        sensitive_col = st.selectbox(
            "Sensitive Demographic Column",
            options=other_cols,
            index=other_cols.index("gender") if "gender" in other_cols else 0,
        )

    # ── Validate prediction column is binary ───────────────────────────────
    pred_values = df[prediction_col].dropna().unique()
    is_binary = set(pred_values).issubset({0, 1})

    if not is_binary:
        st.error(
            "⚠ **Prediction column must contain binary values (0 or 1).**  \n"
            f"The selected column `{prediction_col}` contains values: "
            f"{sorted(pred_values)[:10]}{'…' if len(pred_values) > 10 else ''}.  \n"
            "Please select a column that represents binary model predictions."
        )
        st.stop()

    # ── 3. Bias Detection Logic ────────────────────────────────────────────
    group_rates = (
        df.groupby(sensitive_col)[prediction_col]
        .mean()
        .reset_index()
        .rename(columns={prediction_col: "prediction_rate"})
    )
    group_rates["prediction_rate"] = group_rates["prediction_rate"].round(4)

    max_rate = group_rates["prediction_rate"].max()
    min_rate = group_rates["prediction_rate"].min()
    disparity = max_rate - min_rate
    fairness_score = round(max(0, 100 - disparity * 100), 1)
    bias_detected = disparity > 0.25

    # Percentage versions for display
    max_rate_pct = round(max_rate * 100, 2)
    min_rate_pct = round(min_rate * 100, 2)
    disparity_pct = round(disparity * 100, 2)

    st.markdown('<hr class="glow-line">', unsafe_allow_html=True)

    # ── 4. Fairness Metrics Table ──────────────────────────────────────────
    st.markdown('<div class="section-hdr">📊 Fairness Metrics</div>', unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">Highest Group Rate (%)</div>'
            f'<div class="kpi-value">{max_rate_pct}%</div></div>',
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">Lowest Group Rate (%)</div>'
            f'<div class="kpi-value">{min_rate_pct}%</div></div>',
            unsafe_allow_html=True,
        )
    with m3:
        disp_color = "#ff4b6e" if bias_detected else "#00d4aa"
        st.markdown(
            f'<div class="kpi-card" style="border-color:{disp_color}40;">'
            f'<div class="kpi-label">Disparity (%)</div>'
            f'<div class="kpi-value" style="color:{disp_color};">{disparity_pct}%</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")
    st.markdown("**Group Prediction Rates**")
    st.dataframe(
        group_rates.style.format({"prediction_rate": "{:.4f}"}).background_gradient(
            subset=["prediction_rate"], cmap="RdYlGn_r", vmin=0, vmax=1
        ),
        use_container_width=True,
    )

    st.markdown('<hr class="glow-line">', unsafe_allow_html=True)

    # ── 5. Visualizations ──────────────────────────────────────────────────
    st.markdown(
        '<div class="section-hdr">📈 Bias Visualization</div>',
        unsafe_allow_html=True,
    )

    viz_left, viz_right = st.columns(2)

    # ---- 5a. Plotly Bar Chart ---------------------------------------------
    with viz_left:
        st.markdown("**Prediction Rate by Group**")

        bar_colors = [
            "#ff4b6e" if r == max_rate and bias_detected else "#00d4aa"
            for r in group_rates["prediction_rate"]
        ]

        bar_fig = go.Figure(
            go.Bar(
                x=group_rates[sensitive_col],
                y=group_rates["prediction_rate"],
                marker_color=bar_colors,
                marker_line_width=0,
                text=group_rates["prediction_rate"].apply(lambda v: f"{v:.2%}"),
                textposition="outside",
                textfont=dict(color="#e8eaed", size=13, family="Inter"),
            )
        )
        bar_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#12151f",
            plot_bgcolor="#12151f",
            margin=dict(l=40, r=20, t=30, b=50),
            height=370,
            xaxis=dict(
                title=sensitive_col.replace("_", " ").title(),
                title_font=dict(color="#7f8694"),
                tickfont=dict(color="#9aa0a6"),
                gridcolor="rgba(255,255,255,.04)",
            ),
            yaxis=dict(
                title="Prediction Rate",
                title_font=dict(color="#7f8694"),
                tickfont=dict(color="#9aa0a6"),
                range=[0, min(1.15, max_rate * 1.35)],
                gridcolor="rgba(255,255,255,.06)",
            ),
        )
        # Threshold line at 0.5
        bar_fig.add_hline(
            y=0.5,
            line_dash="dot",
            line_color="#f59e0b",
            opacity=0.5,
            annotation_text="50 %",
            annotation_font_color="#f59e0b",
        )
        st.plotly_chart(bar_fig, use_container_width=True)

    # ---- 5b. Seaborn Heatmap ----------------------------------------------
    with viz_right:
        st.markdown("**Bias Heatmap**")

        heatmap_data = group_rates.set_index(sensitive_col).T

        fig_hm, ax_hm = plt.subplots(
            figsize=(max(5, len(group_rates) * 1.4), 3.2)
        )
        fig_hm.patch.set_facecolor("#12151f")
        ax_hm.set_facecolor("#12151f")

        cmap = sns.diverging_palette(350, 150, s=80, l=55, as_cmap=True)
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt=".3f",
            cmap=cmap,
            linewidths=2.5,
            linecolor="#1a1d29",
            cbar_kws={"shrink": 0.82, "label": "Rate"},
            ax=ax_hm,
            vmin=0,
            vmax=1,
            annot_kws={"fontsize": 14, "fontweight": "bold", "color": "white"},
        )
        ax_hm.set_ylabel("")
        ax_hm.set_xlabel("")
        ax_hm.tick_params(colors="#9aa0a6", labelsize=11)
        cbar = ax_hm.collections[0].colorbar
        cbar.ax.yaxis.label.set_color("#9aa0a6")
        cbar.ax.tick_params(colors="#9aa0a6")
        plt.tight_layout()

        st.pyplot(fig_hm)

    st.markdown('<hr class="glow-line">', unsafe_allow_html=True)

    # ── 6. Fairness Score Panel ────────────────────────────────────────────
    st.markdown(
        '<div class="section-hdr">🎯 AI Fairness Score</div>',
        unsafe_allow_html=True,
    )

    score_col, verdict_col = st.columns([1, 2])

    with score_col:
        # Circular score ring
        ring_color = "#00d4aa" if not bias_detected else "#ff4b6e"
        st.markdown(
            f"""
            <div class="score-ring"
                 style="background: conic-gradient(
                    {ring_color} 0deg,
                    {ring_color} {fairness_score * 3.6}deg,
                    #1e2230 {fairness_score * 3.6}deg,
                    #1e2230 360deg
                 );">
                <div class="inner">
                    <span class="num" style="color:{ring_color};">{fairness_score}</span>
                    <span class="of">/ 100</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("")

        # Streamlit progress bar (clamped 0‑1)
        st.progress(max(0.0, min(1.0, fairness_score / 100)))

    with verdict_col:
        if bias_detected:
            st.markdown(
                '<br><span class="chip-warn">⚠ Bias Detected</span>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <p style="color:#9aa0a6; margin-top:.9rem; line-height:1.7;">
                The disparity between demographic groups is <b style="color:#ff4b6e;">{disparity:.2%}</b>,
                which exceeds the <b>25 %</b> threshold.<br>
                Consider retraining the model with balanced data, applying post‑processing calibration,
                or auditing feature importance to reduce discriminatory patterns.
                </p>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<br><span class="chip-ok">✅ Model Appears Fair</span>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <p style="color:#9aa0a6; margin-top:.9rem; line-height:1.7;">
                The disparity between demographic groups is <b style="color:#00d4aa;">{disparity:.2%}</b>,
                which is within the <b>25 %</b> acceptable range.<br>
                Continue monitoring prediction rates as the model evolves and new data is introduced.
                </p>
                """,
                unsafe_allow_html=True,
            )

    # ── 7. Detailed Breakdown (expandable) ─────────────────────────────────
    st.markdown('<hr class="glow-line">', unsafe_allow_html=True)

    with st.expander("🔍 Detailed Group Breakdown", expanded=False):
        for _, row in group_rates.iterrows():
            grp = row[sensitive_col]
            rate = row["prediction_rate"]
            pct = rate * 100
            bar_html = (
                f'<div style="display:flex;align-items:center;gap:.8rem;margin:.45rem 0;">'
                f'<span style="min-width:100px;color:#e8eaed;font-weight:600;">{grp}</span>'
                f'<div style="flex:1;background:#1e2230;border-radius:8px;height:22px;overflow:hidden;">'
                f'<div style="width:{pct}%;height:100%;background:linear-gradient(90deg,#00d4aa,#38bdf8);'
                f'border-radius:8px;transition:width .4s ease;"></div></div>'
                f'<span style="min-width:60px;text-align:right;color:#9aa0a6;">{rate:.2%}</span>'
                f'</div>'
            )
            st.markdown(bar_html, unsafe_allow_html=True)

else:
    # ── Empty state ────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="text-align:center; padding:5rem 1rem 4rem;">
            <p style="font-size:4.5rem; margin:0; line-height:1;">📤</p>
            <h3 style="color:#e8eaed; font-weight:700; margin-top:1rem;">
                Upload a CSV to begin the audit
            </h3>
            <p style="color:#7f8694; max-width:520px; margin:.8rem auto 0; line-height:1.65;">
                Use the sidebar to upload a healthcare prediction dataset.<br>
                The auditor will analyse prediction rates across demographic groups,
                surface potential bias, and compute a fairness score.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
