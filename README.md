# Personalised Recommendation System 🔍

## 📌 Project Overview
This project develops a personalised recommendation system using real-world e-commerce behaviour data.  
The system leverages historical user interactions, item properties, and category hierarchy to provide relevant, timely suggestions — addressing scenarios like:


---

## 🎯 Objective
- Improve user engagement through tailored recommendations  
- Increase click-through and conversion rates  
- Demonstrate the use of machine learning in real-world datasets  

---
📂 Dataset Description

The dataset (from a real-world e-commerce site) contains:
- events.csv – user behaviour logs over 4.5 months
     - Columns: timestamp, visitorid, event (view, addtocart, transaction), itemid, transactionid
- item_properties_part1.csv & item_properties_part2.csv – historical changes to item attributes
     - Columns: timestamp, itemid, property, value
     - Includes properties like categoryid, available, numeric and hashed features.
- category_tree.csv – category hierarchy
     - Columns: categoryid, parentid
     - Used to group categories into higher-level clusters and compute category depth.

---
## 📂 Project Structure

| Folder/File          | Description |
|----------------------|-------------|
| `data/`              | Raw and processed datasets |
| `notebooks/`         | Jupyter notebooks for analysis and model building |
| `scripts/`           | Python scripts for preprocessing and model training |
| `daily_logs/`        | Daily work updates and progress reports |
| `README.md`          | Main project documentation |
| `requirements.txt`   | List of project dependencies |


---

## 📅 Daily Progress Logs
To follow the development journey of this project, check out the daily updates:

- [Day 1 - Data Exploration & Setup](daily_logs/day1.md)
- [Day 2 - Data Cleaning & Preprocessing](daily_logs/Day2.md)  
*(More will be added as the project progresses)*

---

## ⚙️ Tech Stack
- **Python** (Pandas, NumPy, Scikit-learn)  
- **Jupyter Notebook**  
- **Matplotlib / Seaborn** for visualisation  
- **Machine Learning** for recommendations
- **GitHub** for version control & documentation

---

📈 Methodology – CRISP-DM Framework
1. Business Understanding
- Define the business problem: improve user engagement and conversion rates through personalised recommendations.
- Formulate analytical questions, e.g.:
  Which items should be recommended to each user given their history?
  How do user preferences change over time?
  Which categories drive the most transactions?

2. Data Understanding
- Explore each dataset: volume, structure, missing values, anomalies.
- Profile events by type, time, user, and item.
- Understand category hierarchy depth and distribution.

3. Data Preparation
- Chunk loading for large files to prevent memory errors.
- Filter and merge relevant properties (e.g., categoryid) into events.
- Implement time-aware joins so that events get the correct category at the event time.
- Encode features for collaborative and content-based models.

4. Modelling
- Baselines: Popularity-based recommendation.
- Collaborative filtering: ALS (Alternating Least Squares), BPR (Bayesian Personalised Ranking).
- Content-based: Cosine similarity on item features.
- Hybrid: Weighted blend of collaborative & content models.

5. Evaluation
- Ranking metrics: Precision@K, Recall@K, MAP, NDCG.
- Business metrics: Item coverage, diversity, novelty.
- Segment analysis: New vs returning users, top vs tail items.

6. Deployment & Visualisation
- Interactive dashboard to explore trends & model results.
- Scalable batch inference pipeline for generating recommendation

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

🔗 **App URL:**  
https://recommendation-system-project-nqxms4rszkjgmwmk8eu6bv.streamlit.app/

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://recommendation-system-project-nqxms4rszkjgmwmk8eu6bv.streamlit.app/)

**What you can do in the app**
- **Task 1 – Ranking:** Enter recent category views to get top-k category recommendations.
- **Task 2 – Anomaly Detection:** Upload the user features table (or use the default) to flag anomalous users at a chosen share or using the frozen operating point.

---

🧠 Key Learnings
- Time-aware property alignment prevents future data leakage.
- Collaborative filtering works well but benefits from hybridising with content-based signals.
- Efficient chunk loading & preprocessing are essential for large datasets.

---

👤 Author

Juliet Fafali Kukuia – Data Analyst @ getINNOtized
