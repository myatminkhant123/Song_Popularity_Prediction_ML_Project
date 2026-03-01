# 🎵 Song Popularity Predictor: Enterprise ML Application

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-%23150458.svg?style=for-the-badge)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)

An end-to-end Machine Learning pipeline and fully interactive web application that predicts whether a song has the mathematical characteristics of a Billboard Hot 100 hit. 

Trained on an exhaustive dataset of **232,000+ Spotify tracks**, this application utilizes advanced tree-based ensemble methods (**XGBoost**, **Random Forest**) alongside unsupervised **KMeans Clustering** to profile audio feature performance. It natively integrates **SHAP (SHapley Additive exPlanations)** to provide mathematical "glass-box" game-theory explainability for every single model prediction.

---

## 🎯 Key Features & Technical Architecture

### 1. Robust Predictive Modeling (Supervised Learning)
Classifiers trained on `SpotifyFeatures.csv` (232K rows) to predict target labels thresholded by the median popularity score.
* **Algorithm Stack:** XGBoost Classifier, Random Forest, Decision Trees, Support Vector Machines (SVM), Logistic Regression.
* **Evaluation Metrics:** ROC-AUC, F1 Score, Precision, Recall, Accuracy.
* **Outcome:** The XGBoost algorithm achieved the highest robust F1 score by effectively capturing non-linear feature interactions (e.g., highly energetic tracks needing correspondingly high danceability thresholds).

### 2. Audio Profiling (Unsupervised Learning)
* **Algorithm:** KMeans (k=3) optimized via Silhouette Scoring and the Elbow Method.
* **Dimensionality Reduction:** Principal Component Analysis (PCA) projecting 9 real-valued audio features into a 2D Euclidean space for geometric boundary visualization.
* **Outcome:** Songs segment mathematically into *High Energy*, *Balanced*, and *Acoustic/Calm* genres based purely on raw audio frequencies and metrics.

### 3. "Glass-Box" Explainability (SHAP Game Theory)
* Integrates `TreeExplainer` and calculates marginal Shapley values for the precise features influencing the XGBoost prediction.
* Renders real-time interactive **Waterfall** and **Force plots** illustrating precisely which subset of variables drove the model probability either up or down for an inputted song.

### 4. Live Streamlit Web App
* Glassmorphism dark-mode UI/UX.
* Interactive sidebar for users to input custom song metrics.
* Dynamic dataset charting using Matplotlib and Seaborn.

---

## � Quick Start (Run Locally)

This app is production-ready. To launch it locally on your machine:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/myatminkhant123/Song_Popularity_Prediction_ML_Project.git
   cd Song_Popularity_Prediction_ML_Project
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train Models (Generates `.pkl` binaries in `/models/`):**
   *(Note: Models are generated dynamically to bypass GitHub LFS limits)*
   ```bash
   python train_models.py
   ```

4. **Launch the Dashboard:**
   ```bash
   streamlit run app.py
   ```

---

## � Dataset Metadata
* **Spotify Ultimate Tracks DB**: ~232,000+ tracks outlining fundamental audio acoustic features (Energy, Tempo, Valence, Acousticness, Liveness, Speechiness, Loudness).
* **Target Value:** `Popularity` feature extracted, encoded, and thresholded into binary classification targets.

---

## � Contact & Context
This architecture was originally structured for the **CCS2213 Machine Learning** course at **Albukhary International University (AIU)** and heavily expanded into a production web deployment demonstrating enterprise ML explainability principles.
