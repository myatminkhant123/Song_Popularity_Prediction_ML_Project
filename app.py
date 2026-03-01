"""
🎵 Song Popularity Predictor — Streamlit App
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
import shap
from streamlit_shap import st_shap

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🎵 Song Popularity Predictor",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
        min-height: 100vh;
    }

    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    }

    .hero-title {
        font-size: 3.2rem;
        font-weight: 900;
        background: linear-gradient(120deg, #a78bfa, #818cf8, #38bdf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        letter-spacing: -1px;
        margin-bottom: 0.2rem;
    }

    .hero-subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1.1rem;
        font-weight: 300;
        margin-bottom: 2rem;
    }

    .metric-card {
        background: linear-gradient(135deg, rgba(167,139,250,0.15), rgba(56,189,248,0.1));
        border: 1px solid rgba(167,139,250,0.3);
        border-radius: 16px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        backdrop-filter: blur(10px);
    }

    .metric-card h2 {
        color: #a78bfa;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
    }

    .metric-card p {
        color: #94a3b8;
        font-size: 0.85rem;
        margin: 0;
        margin-top: 0.2rem;
    }

    .prediction-popular {
        background: linear-gradient(135deg, rgba(52,211,153,0.2), rgba(16,185,129,0.1));
        border: 2px solid rgba(52,211,153,0.5);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
    }

    .prediction-not-popular {
        background: linear-gradient(135deg, rgba(251,113,133,0.2), rgba(239,68,68,0.1));
        border: 2px solid rgba(251,113,133,0.5);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
    }

    .prediction-title {
        font-size: 2rem;
        font-weight: 800;
        color: white;
        margin-bottom: 0.5rem;
    }

    .prediction-confidence {
        font-size: 1rem;
        color: #94a3b8;
    }

    .section-header {
        font-size: 1.6rem;
        font-weight: 700;
        color: white;
        margin-bottom: 0.3rem;
        border-left: 4px solid #a78bfa;
        padding-left: 0.8rem;
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.05);
        border-radius: 8px 8px 0 0;
        color: #94a3b8;
        padding: 0.5rem 1.5rem;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(167,139,250,0.3), rgba(56,189,248,0.2));
        color: white !important;
        border-bottom: 2px solid #a78bfa;
    }

    .stSlider > div { color: #94a3b8; }

    .info-box {
        background: rgba(56,189,248,0.1);
        border: 1px solid rgba(56,189,248,0.3);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        color: #bae6fd;
        font-size: 0.9rem;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1e3f 0%, #12122a 100%);
        border-right: 1px solid rgba(167,139,250,0.2);
    }

    div[data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
</style>
""", unsafe_allow_html=True)


# ─── LOAD MODELS ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    models_path = "models"
    if not os.path.exists(models_path):
        return None
    with open(f"{models_path}/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(f"{models_path}/lr_model.pkl", "rb") as f:
        lr = pickle.load(f)
    with open(f"{models_path}/dt_model.pkl", "rb") as f:
        dt = pickle.load(f)
    with open(f"{models_path}/xgb_model.pkl", "rb") as f:
        xgb = pickle.load(f)
    with open(f"{models_path}/svm_model.pkl", "rb") as f:
        svm = pickle.load(f)
    with open(f"{models_path}/kmeans.pkl", "rb") as f:
        kmeans = pickle.load(f)
    with open(f"{models_path}/shap_explainer.pkl", "rb") as f:
        shap_data = pickle.load(f)
    with open(f"{models_path}/metadata.pkl", "rb") as f:
        meta = pickle.load(f)
    return scaler, lr, dt, xgb, svm, kmeans, shap_data, meta


# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🎵 Song Popularity Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Machine Learning • XGBoost • SHAP Explainability</div>', unsafe_allow_html=True)
st.markdown("---")

# ─── LOAD CHECK ───────────────────────────────────────────────────────────────
data = load_models()

if data is None:
    st.error("⚠️ Models not found! Please run the training script first:")
    st.code("python train_models.py", language="bash")
    st.info("This will train all models on 232,000+ Spotify tracks and save them.")
    st.stop()

scaler, lr, dt, xgb, svm, kmeans, shap_data, meta = data
FEATURES = meta['features']
threshold = meta['popularity_threshold']
results = meta['results']
cluster_names = meta['cluster_names']
cluster_summary = pd.DataFrame(meta['cluster_summary'])
feature_importances = meta['feature_importances']
pca_df = pd.DataFrame(meta['pca_df'])
shap_explainer = shap_data['explainer']


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎛️ Song Features")
    st.markdown("*Adjust the sliders to describe a song*")
    st.markdown("---")

    danceability = st.slider("💃 Danceability", 0.0, 1.0, 0.65)
    energy = st.slider("⚡ Energy", 0.0, 1.0, 0.70)
    valence = st.slider("😊 Valence (Happiness)", 0.0, 1.0, 0.55)
    acousticness = st.slider("🎸 Acousticness", 0.0, 1.0, 0.15)
    loudness = st.slider("🔊 Loudness (dB)", -60.0, 0.0, -6.0)
    tempo = st.slider("🥁 Tempo (BPM)", 50.0, 250.0, 120.0)
    speechiness = st.slider("🎤 Speechiness", 0.0, 1.0, 0.08)
    instrumentalness = st.slider("🎹 Instrumentalness", 0.0, 1.0, 0.0)
    liveness = st.slider("🎪 Liveness", 0.0, 1.0, 0.12)

    st.markdown("---")
    model_choice = st.selectbox(
        "🤖 Choose Prediction Model",
        ["XGBoost", "Logistic Regression", "Decision Tree", "SVM"],
        index=0
    )


# ─── PREDICTION ENGINE ────────────────────────────────────────────────────────
def predict(features_dict, model_name):
    X = np.array([[
        features_dict['acousticness'],
        features_dict['danceability'],
        features_dict['energy'],
        features_dict['instrumentalness'],
        features_dict['liveness'],
        features_dict['loudness'],
        features_dict['speechiness'],
        features_dict['tempo'],
        features_dict['valence'],
    ]])
    X_scaled = scaler.transform(X)

    model_map = {
        "Logistic Regression": lr,
        "Decision Tree": dt,
        "XGBoost": xgb,
        "SVM": svm,
    }
    model = model_map[model_name]
    pred = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0][1]
    cluster = kmeans.predict(X_scaled)[0]
    return X_scaled, pred, proba, cluster


features_input = {
    'danceability': danceability, 'energy': energy, 'valence': valence,
    'acousticness': acousticness, 'loudness': loudness, 'tempo': tempo,
    'speechiness': speechiness, 'instrumentalness': instrumentalness,
    'liveness': liveness
}

X_scaled, prediction, confidence, cluster_id = predict(features_input, model_choice)
cluster_label = cluster_names.get(cluster_id, f"Cluster {cluster_id}")


# ─── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Predict", "🧠 SHAP Explainability", "📊 Model Performance", "🔵 Clustering", "📈 Feature Importance"
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<div class="section-header">🎯 Prediction Result</div>', unsafe_allow_html=True)
        st.markdown("")

        if prediction == 1:
            st.markdown(f"""
            <div class="prediction-popular">
                <div class="prediction-title">✅ POPULAR</div>
                <div class="prediction-confidence">This song is likely to be a hit!<br>
                <strong style="color:#34d399; font-size:1.4rem;">{confidence:.1%}</strong> confidence by {model_choice}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="prediction-not-popular">
                <div class="prediction-title">❌ NOT POPULAR</div>
                <div class="prediction-confidence">This song may not chart highly.<br>
                <strong style="color:#fb7185; font-size:1.4rem;">{1 - confidence:.1%}</strong> confidence by {model_choice}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        cluster_colors = {0: "#f59e0b", 1: "#818cf8", 2: "#34d399"}
        color = cluster_colors.get(cluster_id, "#94a3b8")
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.05); border-radius:12px; padding:1rem 1.2rem;
                    border-left: 4px solid {color}; margin-top:1rem;">
            <strong style="color:{color}; font-size:1.1rem;">Song Cluster: {cluster_label}</strong><br>
            <span style="color:#94a3b8; font-size:0.9rem;">
            This song shares audio characteristics with the <em>{cluster_label}</em> group.
            </span>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="section-header">📡 Your Song Profile</div>', unsafe_allow_html=True)
        st.markdown("")

        radar_features = ['danceability', 'energy', 'valence', 'acousticness',
                          'speechiness', 'liveness', 'instrumentalness']
        radar_values = [features_input[f] for f in radar_features]
        radar_labels = ['Dance', 'Energy', 'Valence', 'Acoustic',
                        'Speech', 'Live', 'Instrumental']

        fig, ax = plt.subplots(figsize=(6, 3.5), facecolor='#1a1a2e')
        colors_bar = ['#a78bfa' if v >= 0.5 else '#64748b' for v in radar_values]
        bars = ax.barh(radar_labels, radar_values, color=colors_bar, height=0.6)
        ax.set_xlim(0, 1)
        ax.axvline(x=0.5, color=(1,1,1,0.25), linestyle='--', linewidth=1.5,
                   alpha=0.8, label='Median')
        ax.set_facecolor('#1a1a2e')
        fig.patch.set_alpha(0)
        ax.tick_params(colors='white', labelsize=10)
        ax.xaxis.label.set_color('white')
        for spine in ax.spines.values():
            spine.set_edgecolor((1, 1, 1, 0.15))
        ax.set_xlabel("Feature Value", color='#94a3b8', fontsize=9)
        for bar, val in zip(bars, radar_values):
            ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{val:.2f}', va='center', color='white', fontsize=8)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-header">📊 Dataset Overview</div>', unsafe_allow_html=True)
    st.markdown("")
    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.markdown(f"""<div class="metric-card"><h2>232K+</h2><p>Spotify Tracks Trained On</p></div>""",
                    unsafe_allow_html=True)
    with mc2:
        st.markdown(f"""<div class="metric-card"><h2>{threshold:.0f}</h2><p>Popular Score Threshold</p></div>""",
                    unsafe_allow_html=True)
    with mc3:
        best_model = max(results, key=lambda m: results[m]['f1'])
        best_f1 = results[best_model]['f1']
        st.markdown(f"""<div class="metric-card"><h2>{best_f1:.2%}</h2><p>Top F1 Score ({best_model})</p></div>""",
                    unsafe_allow_html=True)
    with mc4:
        st.markdown(f"""<div class="metric-card"><h2>5</h2><p>ML Models Compared</p></div>""",
                    unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SHAP EXPLAINABILITY (NEW)
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">🧠 SHAP Explainability Dashboard</div>', unsafe_allow_html=True)
    st.markdown("*Using state-of-the-art Game Theory mathematics to explain EXACTLY why the XGBoost model made its decision.*")
    st.markdown("")
    
    st.info("💡 **How to interpret this:** Red features push the probability of the song being **popular** HIGHER. Blue features pull the probability LOWER.")
    
    # Generate SHAP value for the single input row
    shap_values = shap_explainer.shap_values(X_scaled)
    expected_value = shap_explainer.expected_value
    
    # We must pass real feature names and values back matched properly
    input_df = pd.DataFrame(X_scaled, columns=FEATURES)

    # Plot SHAP Force Plot
    st.markdown("### The Decision Path for Your Song")
    st_shap(shap.force_plot(expected_value, shap_values[0,:], input_df.iloc[0,:]), height=150)
    
    st.markdown("### Decision Waterfall")
    try:
        # Shap waterfall works best on Explaination objects
        shap_explanation = shap.Explanation(
            values=shap_values[0], 
            base_values=expected_value, 
            data=input_df.iloc[0].values, 
            feature_names=FEATURES
        )
        # Using matplotlib to render the SHAP waterfall internally
        fig_shap, ax_shap = plt.subplots(figsize=(8, 5))
        shap.plots.waterfall(shap_explanation, show=False)
        fig_shap.set_facecolor('#1a1a2e')
        fig_shap.patch.set_alpha(0)
        ax_shap.tick_params(colors='white')
        st.pyplot(fig_shap, use_container_width=True)
        plt.clf()
    except Exception as e:
        st.error("Waterfall plotting requires different SHAP formatting.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">📊 Model Performance Comparison</div>', unsafe_allow_html=True)
    
    results_df = pd.DataFrame(results).T.round(4)
    if 'Recall' in results_df.columns:
        results_df.columns = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    elif 'f1' in results_df.columns:
        results_df.rename(columns={'accuracy': 'Accuracy', 'f1': 'F1 Score', 'roc_auc': 'ROC-AUC'}, inplace=True)

    def highlight_best(s):
        is_max = s == s.max()
        return ['background-color: rgba(167,139,250,0.3); color: white' if v else '' for v in is_max]

    styled = results_df.style.apply(highlight_best, axis=0).format("{:.4f}")
    st.dataframe(styled, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**F1 Score Comparison**")
        fig, ax = plt.subplots(figsize=(6, 3.5), facecolor='#1a1a2e')
        models_list = list(results.keys())
        f1_scores = [results[m].get('F1 Score') or results[m].get('f1') for m in models_list]
        bar_colors = ['#a78bfa' if s == max(f1_scores) else '#4f46e5' for s in f1_scores]
        bars = ax.bar(models_list, f1_scores, color=bar_colors, width=0.5, edgecolor='none')
        ax.set_ylim(0, 1)
        ax.set_facecolor('#1a1a2e')
        fig.patch.set_alpha(0)
        ax.tick_params(colors='white', labelsize=9)
        ax.set_xticklabels(models_list, rotation=10, ha='right')
        for spine in ax.spines.values():
            spine.set_edgecolor((1, 1, 1, 0.15))
        for bar, val in zip(bars, f1_scores):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', color='white', fontsize=9)
        ax.set_ylabel("F1 Score", color='#94a3b8', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with col_b:
        st.markdown("**ROC-AUC Comparison**")
        fig, ax = plt.subplots(figsize=(6, 3.5), facecolor='#1a1a2e')
        roc_scores = [results[m].get('ROC-AUC') or results[m].get('roc_auc') for m in models_list]
        bar_colors2 = ['#38bdf8' if s == max(roc_scores) else '#0ea5e9' for s in roc_scores]
        bars2 = ax.bar(models_list, roc_scores, color=bar_colors2, width=0.5, edgecolor='none')
        ax.set_ylim(0, 1)
        ax.set_facecolor('#1a1a2e')
        fig.patch.set_alpha(0)
        ax.tick_params(colors='white', labelsize=9)
        ax.set_xticklabels(models_list, rotation=10, ha='right')
        for spine in ax.spines.values():
            spine.set_edgecolor((1, 1, 1, 0.15))
        for bar, val in zip(bars2, roc_scores):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', color='white', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — CLUSTERING
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">🔵 KMeans Clustering (k=3)</div>', unsafe_allow_html=True)
    st.markdown("*Unsupervised learning — Songs grouped by audio profile similarity*")
    st.markdown("")

    col_c, col_d = st.columns([1.5, 1])
    with col_c:
        fig, ax = plt.subplots(figsize=(8, 5), facecolor='#1a1a2e')
        cluster_colors_map = {0: '#f59e0b', 1: '#818cf8', 2: '#34d399'}
        for c in [0, 1, 2]:
            mask = pca_df['Cluster'] == c
            sample = pca_df[mask].sample(min(500, mask.sum()), random_state=42)
            ax.scatter(sample['PC1'], sample['PC2'],
                       c=cluster_colors_map[c], label=cluster_names[c],
                       alpha=0.5, s=10, edgecolors='none')
        ax.set_facecolor('#1a1a2e')
        fig.patch.set_alpha(0)
        ax.tick_params(colors='white', labelsize=9)
        ax.set_xlabel("Principal Component 1", color='#94a3b8', fontsize=9)
        ax.set_ylabel("Principal Component 2", color='#94a3b8', fontsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor((1, 1, 1, 0.15))
        legend = ax.legend(fontsize=9, labelcolor='white', framealpha=0.15,
                           facecolor='#000000')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with col_d:
        for cid, cname in cluster_names.items():
            color = cluster_colors_map[cid]
            top_features_vals = cluster_summary.loc[cid].sort_values(ascending=False)
            high_feats = top_features_vals.head(3)
            feat_str = " · ".join([f"{k} ({v:.2f})" for k, v in high_feats.items()])

            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.04); border-left: 3px solid {color};
                        border-radius: 8px; padding: 0.8rem 1rem; margin-bottom: 0.8rem;">
                <strong style="color:{color}; font-size:1rem;">{cname}</strong><br>
                <span style="color:#94a3b8; font-size:0.82rem;">Top traits: {feat_str}</span>
            </div>
            """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">📈 Feature Rank (XGBoost)</div>', unsafe_allow_html=True)
    st.markdown("*Which audio features best predict song popularity in XGBoost?*")

    importance_df = pd.Series(feature_importances).sort_values(ascending=True)

    col_e, col_f = st.columns([1.5, 1])
    with col_e:
        fig, ax = plt.subplots(figsize=(7, 4.5), facecolor='#1a1a2e')
        gradient_colors = plt.cm.plasma(np.linspace(0.3, 0.9, len(importance_df)))
        bars = ax.barh(importance_df.index, importance_df.values,
                       color=gradient_colors, height=0.6)
        ax.set_facecolor('#1a1a2e')
        fig.patch.set_alpha(0)
        ax.tick_params(colors='white', labelsize=10)
        ax.set_xlabel("Importance Score", color='#94a3b8', fontsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor((1, 1, 1, 0.15))
        for bar, val in zip(bars, importance_df.values):
            ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                    f'{val:.3f}', va='center', color='white', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with col_f:
        sorted_importance = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
        medals = ['🥇', '🥈', '🥉'] + ['🔹'] * 10
        for i, (feat, score) in enumerate(sorted_importance):
            bar_width = int(score * 200)
            st.markdown(f"""
            <div style="margin-bottom:0.6rem;">
                <div style="display:flex; justify-content:space-between; margin-bottom:3px;">
                    <span style="color:white; font-size:0.9rem;">{medals[i]} {feat.capitalize()}</span>
                    <span style="color:#a78bfa; font-size:0.85rem; font-weight:600;">{score:.3f}</span>
                </div>
                <div style="background:rgba(255,255,255,0.1); border-radius:4px; height:6px;">
                    <div style="background:linear-gradient(90deg,#a78bfa,#38bdf8);
                                width:{bar_width}px; max-width:100%; height:6px; border-radius:4px;">
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#475569; font-size:0.8rem; padding:1rem 0;">
    🎵 Song Popularity Predictor · Built with XGBoost, SHAP, and Streamlit
</div>
""", unsafe_allow_html=True)
