# 🎵 Song Popularity Prediction Using Machine Learning

This project applies **machine learning** to predict and analyze the popularity of songs using **Spotify** and **Billboard Hot 100** datasets.  
It combines **unsupervised learning (KMeans clustering)** and **supervised learning (Random Forest, Logistic Regression, SVM)** to uncover patterns in music features such as tempo, danceability, and energy.

---

## 📊 Project Overview

The goal is to identify the factors that influence song popularity and build a predictive model to classify whether a song will be *popular* or *not popular*.

### 🎯 Objectives
- Analyze key audio features (energy, tempo, valence, danceability, etc.)
- Cluster songs into distinct groups using **KMeans (k=3)**
- Predict song popularity using **Random Forest**, **Decision Tree**, **SVM**, and **Logistic Regression**
- Derive insights for artists, record labels, and streaming platforms

---

## 🧠 Machine Learning Techniques

### 🔹 Unsupervised Learning
- **Algorithm:** KMeans Clustering (k=3)
- **Evaluation:** Elbow Method, Silhouette Score
- **Visualization:** PCA 2D plots
- **Result:** 3 distinct clusters representing song types (high-energy, balanced, acoustic)

### 🔹 Supervised Learning
- **Models Tested:** Logistic Regression, Decision Tree, Random Forest, SVM
- **Metrics:** Accuracy, Precision, Recall, F1 Score, ROC-AUC
- **Best Model:** Random Forest Classifier (Highest F1 and ROC-AUC)
- **Feature Importance:** Energy, Danceability, Loudness

---

## 📈 Key Findings
- KMeans clustering revealed 3 major song categories.
- Random Forest outperformed other models in classification accuracy.
- Energy, Danceability, and Acousticness were the strongest predictors of popularity.
- SHAP analysis (conceptual) used to demonstrate model explainability.

---

## 💡 Business Insights
- Streaming platforms can tailor playlists using cluster-based segmentation.
- Record labels can focus on audio profiles that align with successful features.
- Marketing teams can prioritize promotion for songs in high-energy clusters.

---

## 🧰 Tools & Libraries
Python · Scikit-Learn · Pandas · NumPy · Matplotlib · Seaborn · PCA · KMeans · Random Forest

---

## 📂 Datasets
- [Ultimate Spotify Tracks DB](https://www.kaggle.com/datasets/zaheenhamidani/ultimate-spotify-tracks-db)
- [Billboard Hot 100 Dataset](https://www.kaggle.com/code/devraai/analyzing-billboard-hot-100-with-spotify-data/notebook)

---

## 📄 Project Report
Full report included in the repository: `ML_Project_finalized.docx`

---

## 👥 Team Members
- Naing Naing  
- Khalid Waleed Ahmed Abdullah Ghaleb  
- Myat Min Khant  
- Mohamed Nawran Nasar Mohamed

---

## 🧾 License
This project was developed as part of the **CCS2213 Machine Learning** course at **Albukhary International University (AIU)**.
