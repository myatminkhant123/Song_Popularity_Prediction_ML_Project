"""
Train and save ML models using SpotifyFeatures.csv (232K rows)
Added XGBoost and SHAP capabilities.
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
import xgboost as xgb
import shap

print("Loading SpotifyFeatures.csv...")
df = pd.read_csv("SpotifyFeatures.csv")
df = df.dropna(subset=['track_name'])

# ─── FEATURE COLUMNS ──────────────────────────────────────────────────────────
FEATURES = [
    'acousticness', 'danceability', 'energy', 'instrumentalness',
    'liveness', 'loudness', 'speechiness', 'tempo', 'valence'
]

# ─── SUPERVISED LEARNING SETUP ────────────────────────────────────────────────
print("\nPreparing supervised learning data...")
X = df[FEATURES].copy()
popularity_median = df['popularity'].median()
y = (df['popularity'] >= popularity_median).astype(int)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ─── TRAIN ALL MODELS ─────────────────────────────────────────────────────────
results = {}

print("Training Logistic Regression...")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
y_proba_lr = lr.predict_proba(X_test)[:, 1]
results['Logistic Regression'] = {
    'accuracy': accuracy_score(y_test, lr.predict(X_test)),
    'f1': f1_score(y_test, lr.predict(X_test)),
    'roc_auc': roc_auc_score(y_test, y_proba_lr),
}

print("Training Decision Tree...")
dt = DecisionTreeClassifier(max_depth=10, random_state=42)
dt.fit(X_train, y_train)
y_proba_dt = dt.predict_proba(X_test)[:, 1]
results['Decision Tree'] = {
    'accuracy': accuracy_score(y_test, dt.predict(X_test)),
    'f1': f1_score(y_test, dt.predict(X_test)),
    'roc_auc': roc_auc_score(y_test, y_proba_dt),
}

print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_proba_rf = rf.predict_proba(X_test)[:, 1]
results['Random Forest'] = {
    'accuracy': accuracy_score(y_test, rf.predict(X_test)),
    'f1': f1_score(y_test, rf.predict(X_test)),
    'roc_auc': roc_auc_score(y_test, y_proba_rf),
}

print("Training XGBoost (New & Best!)...")
xgb_model = xgb.XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    use_label_encoder=False, eval_metric='logloss', random_state=42
)
xgb_model.fit(X_train, y_train)
y_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
results['XGBoost'] = {
    'accuracy': accuracy_score(y_test, xgb_model.predict(X_test)),
    'f1': f1_score(y_test, xgb_model.predict(X_test)),
    'roc_auc': roc_auc_score(y_test, y_proba_xgb),
}

print("Training SVM (using subset)...")
svm = SVC(probability=True, kernel='rbf', random_state=42)
svm.fit(X_train[:20000], y_train[:20000])
y_proba_svm = svm.predict_proba(X_test)[:, 1]
results['SVM'] = {
    'accuracy': accuracy_score(y_test, svm.predict(X_test)),
    'f1': f1_score(y_test, svm.predict(X_test)),
    'roc_auc': roc_auc_score(y_test, y_proba_svm),
}

# ─── UNSUPERVISED: KMEANS CLUSTERING ──────────────────────────────────────────
print("\nTraining KMeans (k=3)...")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)

pca = PCA(n_components=2)
pca_data = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])
pca_df['Cluster'] = cluster_labels

df['Cluster'] = cluster_labels
cluster_summary = df.groupby('Cluster')[FEATURES].mean().round(4)
cluster_names = {0: '🎸 High Energy', 1: '🎵 Balanced', 2: '🎹 Acoustic/Calm'}

importances = pd.Series(xgb_model.feature_importances_, index=FEATURES).sort_values(ascending=False)

# ─── SHAP EXPLAINER BACKGROUND DATA ───────────────────────────────────────────
print("\nComputing SHAP explainer (using XGBoost)...")
explainer = shap.TreeExplainer(xgb_model)
# Sample 500 rows for expected value/background so Streamlit doesn't choke
shap_background = X_train[:500] 

# ─── SAVE ALL MODELS & DATA ───────────────────────────────────────────────────
print("\nSaving models to /models/...")
os.makedirs("models", exist_ok=True)

with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("models/lr_model.pkl", "wb") as f:
    pickle.dump(lr, f)
with open("models/dt_model.pkl", "wb") as f:
    pickle.dump(dt, f)
with open("models/rf_model.pkl", "wb") as f:
    pickle.dump(rf, f)
with open("models/xgb_model.pkl", "wb") as f:
    pickle.dump(xgb_model, f)
with open("models/svm_model.pkl", "wb") as f:
    pickle.dump(svm, f)
with open("models/kmeans.pkl", "wb") as f:
    pickle.dump(kmeans, f)
with open("models/pca.pkl", "wb") as f:
    pickle.dump(pca, f)
with open("models/shap_explainer.pkl", "wb") as f:
    pickle.dump({'explainer': explainer, 'background': shap_background, 'expected_value': explainer.expected_value}, f)

meta = {
    'features': FEATURES,
    'popularity_threshold': float(popularity_median),
    'results': results,
    'cluster_names': cluster_names,
    'cluster_summary': cluster_summary.to_dict(),
    'feature_importances': importances.to_dict(),
    'pca_df': pca_df.sample(2000, random_state=42).to_dict(orient='records'), # Sample PCA so pickle isn't huge
    'test_size': len(y_test),
    'train_size': len(y_train),
}
with open("models/metadata.pkl", "wb") as f:
    pickle.dump(meta, f)

print("✅ All models updated & saved!")
