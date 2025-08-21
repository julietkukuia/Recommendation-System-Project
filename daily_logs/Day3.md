# 📅 Day 3 – Ranking Model, Anomaly Filtering, and Streamlit Deployment

**Date:** 2025-08-21
**Author:** Juliet Fafali Kukuia
**Project:** Personalised Recommendation System using E-commerce Behaviour Data

---

## ✅ Tasks Completed

### 1) Task 1 – Candidate Ranking model

* **Built a compact candidate set** from each visitor’s recent view sequence (24h window; lookback=100).
* **Engineered 7 light features** per candidate category:
  `cnt, share, is_last, last_gap_min, pos_from_end, tail_streak_if_last, n_views`.
* **Group-aware evaluation**: `GroupShuffleSplit` by `visitorid` + a **class-overlap check** to avoid label leakage.
* **Trained 3 models** and picked the best:

  * Logistic Regression (multinomial) ✅ **selected**
  * HistGradientBoosting
  * Random Forest
* **Before cleaning (original events)**

  * Rule baseline (count + recency): **coverage 91.51% • top1 82.04% • top3 98.83%**
  * Logistic regression: **coverage 91.64% • top1 \~82.9% • top3 \~99.5%**
* **After cleaning (see Task 2)**

  * Rule baseline: **coverage 97.62% • top1 99.46% • top3 100.00%**
  * **Logistic regression (final): coverage 97.74% • top1 99.57% • top3 100.00%**
* **Saved artifacts** for reproducible inference:

  * `recsys_artifacts/task1_ranking_best_logistic_regression.joblib`
  * `recsys_artifacts/task1_ranking_dataset.parquet`
  * `recsys_artifacts/task1_before_after_cleaning.csv` & `.json` (comparison table)

---

### 2) Task 2 – User Anomaly Detection & Cleaning

* **Built 16 user-level features** (events mix, activity cadence, sessionization, entropy, rates):
  `events, view, addtocart, transaction, uniq_items, uniq_cats, cat_entropy, events_per_active_day, add_to_view_rate, txn_to_cart_rate, night_share, p95_gap_s, short_gap_share, n_sessions, sessions_per_day, uniq_item_ratio`.
* **Trained IsolationForest**, tuned via proxy labels; audited **precision/recall vs share** (1–20%).
* **Chose operating point:** **4% flagged share**, froze the **score threshold**.
* **Produced cleaned snapshot** by removing flagged users:

  * Example earlier run: **events before 4,427,915 → after 2,831,382 (−36%)**
  * Wrote scored users and operating point to disk.
* **Saved artifacts**

  * Model: `recsys_artifacts/task2_isolation_forest.joblib`
  * Features: `recsys_artifacts/task2_user_features.parquet`
  * Operating point: `recsys_artifacts/task2_operating_point.json`
  * (Optional snapshot) `recsys_artifacts/events_clean_q04.parquet`

---

### 3) Inference Services & Deployment

* **FastAPI microservice** (`app.py`)

  * `POST /rank` – ranks candidate categories from recent views (falls back to rule if model missing).
  * `POST /anomaly` – returns anomaly score, binary flag, and threshold used.
* **Streamlit app** (`streamlit_app.py`)

  * **Task 1 tab:** interactive ranking from a small views table + quick demo from cleaned events (if present).
  * **Task 2 tab:** upload user-features parquet/CSV, choose flagged share (or use frozen), score & download.
  * Caching for models/data; graceful handling when artifacts are missing.
* **Deployment (Streamlit Cloud)**

  * Set **Python 3.13** in Advanced settings.
  * Pinned deps to avoid build hangs:

    * `streamlit==1.36.0`, `pandas==2.2.2`, `numpy==1.26.4`, `scikit-learn==1.4.2`, `pyarrow==15.0.2`, `joblib==1.3.2`
  * Pushed only **small, necessary artifacts**:
    `task1_ranking_best_logistic_regression.joblib`, `task2_isolation_forest.joblib`, `task2_operating_point.json`.

---

## 📊 Key Findings / Observations

* **Cleaning anomalies transforms performance**: coverage ↑ and top-1 jumps from \~82% to **\~99.6%**.
* **Logistic Regression** matched large models on top-3 while being **fast & memory-light**.
* **Group-aware splits** and explicit **class-overlap checks** prevent optimistic leakage.
* **Freezing the anomaly operating point** ensures **reproducible** behaviour across runs and environments.

---

## ⚠️ Challenges & Fixes

| Challenge                                                       | How It Was Resolved                                                                                |
| --------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `train_test_split` with `stratify` failed due to tail classes   | Dropped ultra-rare classes (`min_count=5`) and switched to **`GroupShuffleSplit`** by `visitorid`. |
| Memory errors with large models                                 | Downcasted to **float32** features, used compact features, and picked **Logistic Regression**.     |
| Timestamp arithmetic errors (`TypeError` on pandas `Timestamp`) | Centralized conversion with `ensure_ts` and used `Timedelta`/`.total_seconds()`.                   |
| Streamlit Cloud build hang / version conflicts                  | **Pinned versions** + **Python 3.13** in Advanced settings; shipped only small artifacts.          |
| Large files in GitHub web UI                                    | Avoided heavy datasets; when needed, pushed via **git** (not web) or kept snapshots local.         |

---

## 🔜 Next Steps

* Prepare powerpoint slides for presentation on this project
* Provide a **demo visitor sampler** and **example payloads** for the FastAPI endpoints.

---

## 📂 Files Updated / Created

| File                                                             | Description                                                           |
| ---------------------------------------------------------------- | --------------------------------------------------------------------- |
| `streamlit_app.py`                                               | Two-tab UI for ranking and anomaly scoring (upload, score, download). |
| `app.py`                                                         | FastAPI endpoints `/rank` and `/anomaly`.                             |
| `recsys_artifacts/task1_ranking_best_logistic_regression.joblib` | Final Task 1 ranking model.                                           |
| `recsys_artifacts/task1_ranking_dataset.parquet`                 | Candidate dataset used for training/eval.                             |
| `recsys_artifacts/task1_before_after_cleaning.csv` / `.json`     | KPI comparison before vs after cleaning.                              |
| `recsys_artifacts/task2_isolation_forest.joblib`                 | Trained IsolationForest for anomalies.                                |
| `recsys_artifacts/task2_user_features.parquet`                   | 16-feature user table for Task 2.                                     |
| `recsys_artifacts/task2_operating_point.json`                    | Frozen threshold & share for reproducible scoring.                    |
| *(optional)* `recsys_artifacts/events_clean_q04.parquet`         | Cleaned events snapshot used for the demo panel.                      |
| `requirements.txt`                                               | Pinned versions for reliable local & cloud builds.                    |

---

## 🧠 Key Learnings

* **Data quality first**: removing abnormal users massively improves downstream recommenders.
* **Leakage control** (by group) is as important as model choice for trustworthy metrics.
* **Lightweight models + compact features** win on speed, stability, and deployability.
* **Pin your environment**: explicit versions and a minimal artifact set make Cloud deploys painless.

---

*End of Day 3 log.*
