# 🧠 Hybrid Category Recommender with Anomaly Detection  

> **Live Demo:** [Streamlit App](https://recommendation-system-project-c8cgulvvsynvm77sfiyfwb.streamlit.app/)  

---

## 📌 Project Overview  

This project implements a **hybrid recommendation system** enhanced with **anomaly detection**.  
It combines:  
- **Task 1 — Recommendation System**: Suggests the most relevant product categories to users using both baseline heuristics and machine learning (SVD embeddings + logistic regression).  
- **Task 2 — Anomaly Detection**: Identifies abnormal or fraudulent users with an **Isolation Forest model** trained on engineered user behaviour features.  

The system is interactive, deployed as a **Streamlit application**.  

---

## 📅 Daily Progress Logs
To follow the development journey of this project, check out the daily updates:

- [Day 1 - Data Exploration & Setup](daily_logs/day1.md)
- [Day 2 - Data Cleaning & Preprocessing](daily_logs/Day2.md)
- [Day 3 – Ranking Model, Anomaly Filtering, and Streamlit Deployment](daily_logs/Day3.md)

---

## 📊 Data & Preprocessing  

### **Data Sources**
- **Events data**:  
  Columns include `timestamp`, `visitorid`, `event`, `itemid`, `categoryid_final`.  
- **Derived features**:  
  - Session-level: recency, frequency, last-gap times.  
  - User-level: aggregate counts, entropy, ratios for anomaly detection.  

### **Preprocessing Steps**
1. Normalized timestamps into integer milliseconds.  
2. Standardized column types (`visitorid`, `itemid`, `categoryid_final`).  
3. Dropped rows with missing category IDs.  
4. Built user features table (`task2_user_features.parquet`).  
5. Sharded large files (like user embeddings) into compressed `.parquet` parts to fit GitHub storage limits.  

---

## ⚙️ System Architecture  

### **Task 1 — Hybrid Recommender**
- **Baseline**: Rule-based scoring with frequency and recency.  
- **Hybrid ML model**:  
  - Extracted SVD embeddings for users & items.  
  - Engineered behavioural features.  
  - Logistic regression ranking model trained to predict “add” events.  

**Flow**:  
Events → Candidate generation → Feature engineering → Model scoring → Top-K categories  

### **Task 2 — Anomaly Detection**
- **Algorithm**: Isolation Forest  
- **Input**: 16-dimensional per-user feature vector.  
- **Output**:  
  - `anom_score`: anomaly likelihood.  
  - Flagged users list (top quantile by score).  

**Flow**:  
User features → Isolation Forest → Flagged user IDs  

---

## 🖥️ Streamlit Application  

The app has **two main tabs**:  

### 🔹 Recommendations  
- Upload an events dataset or use the provided sample (`sample_events.parquet`).  
- Pick a user from top active users or enter a user ID manually.  
- Configure parameters:  
  - **Top-K recommendations**  
  - **Lookback hours**  
  - **Max recent views**  
- Choose whether to **exclude anomalous users**.  
- Get Top-K recommended categories.  

### 🔹 Anomaly Audit  
- Loads pre-engineered user features (`task2_user_features.parquet`).  
- Runs the trained **Isolation Forest** model.  
- Displays:  
  - Number of flagged users.  
  - Preview of flagged users.  
- Supports **per-user anomaly scoring**.  
- Allows exporting flagged users as CSV.  

---

## ✅ Evaluation  

### **Recommendation System**
- Metrics: **Coverage**, **Top-1 Accuracy**, **Top-3 Accuracy**.  
- After cleaning (removing anomalies), ranking performance improved across all metrics.  

### **Anomaly Detection**
- Proxy labels used for evaluation.  
- Precision/Recall tradeoff analysed.  
- Optimal operating point: flagging ~5–10% of users balances recall and false positives.  

---

## 📂 Repository Structure  

```
recsys_items/
│── streamlit_app.py               # Main Streamlit app
│── requirements.txt               # Dependencies
│── sample_events.parquet          # Small demo dataset
│── best_candidate_ranker.pkl      # Logistic regression model (Task 1)
│── inference_metadata.json        # Feature metadata (Task 1)
│── svd_item_factors.parquet       # Item embeddings
│── svd_user_factors_part*.parquet # Sharded user embeddings
│
recsys_artifacts/ (Task 2)
│── task2_isolation_forest.joblib  # Trained anomaly model
│── task2_user_features.parquet    # User features
│── task2_flagged_users.csv        # Pre-flagged anomalies
│── task2_operating_point.json     # Threshold config
```

## 🚀 How to Run  

1. Clone the repository  
   ```bash
   git clone https://github.com/julietkukuia/Recommendation-System-Project.git
   cd Recommendation-System-Project/recsys_items
   ```

2. Install dependencies  
   ```bash
   pip install -r requirements.txt
   ```

3. Launch Streamlit app  
   ```bash
   streamlit run streamlit_app.py
   ```

---

## ⚙️ Tech Stack
- **Python** (Pandas, NumPy, Scikit-learn)  
- **Jupyter Notebook**  
- **Matplotlib / Seaborn** for visualisation  
- **Machine Learning** for recommendations
- **GitHub** for version control & documentation
- **Streamlit** for deployment

---

📈 **Methodology – CRISP-DM Framework**
Business Understanding
- Define the business problem: improve user engagement and conversion rates through personalised recommendations.
- Formulate analytical questions, e.g.:
  - Which items should be recommended to each user given their history?
  - How do user preferences change over time?
  - Which categories drive the most transactions?

---

📌 Deliverables

1. Data preparation scripts
2. Modelling notebooks
3. Evaluation report with visualisations
4. README documentation (this file)
5. Final presentation slides

---
## 🚀 Live Demo

Try the deployed app on Streamlit Cloud:
The project is deployed here:
**👉 Live Streamlit App:**  
🔗 [![Open in Streamlit](https://recommendation-system-project-c8cgulvvsynvm77sfiyfwb.streamlit.app/)

**What you can do in the app**
- **Task 1 – Ranking:** Enter recent category views to get top-k category recommendations.
- **Task 2 – Anomaly Detection:** Upload the user features table (or use the default) to flag anomalous users at a chosen share or using the frozen operating point.

---

🧠 Key Learnings
- Time-aware property alignment prevents future data leakage.
- Collaborative filtering works well but benefits from hybridising with content-based signals.
- Efficient chunk loading & preprocessing are essential for large datasets.
- Removing anomalies improves recommendation fairness and quality.
- Hybrid models (embeddings + features) outperform pure rule-based baselines.
- The system is modular: can extend to item-level recommendations, fraud detection, or real-time scoring.

---

👤 Author
Developed as part of a recommender system + anomaly detection project.
Juliet Fafali Kukuia – Data Analyst @ getINNOtized
