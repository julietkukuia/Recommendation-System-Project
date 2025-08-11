📅 Day 2 – Outlier Removal, Category Enrichment & Sparse Matrix Creation

**Date:** 2025-08-10  
**Author:** Juliet Fafali Kukuia  
**Project:** Personalised Recommendation System using E-commerce Behaviour Data  

---

### ✅ Tasks Completed

* **Outlier Detection & Removal**

  * Calculated user and item activity quantiles.
  * Set thresholds: **users >163 interactions** and **items >2302 interactions** flagged as unrealistic.
  * Dropped **1,472,348 rows** (\~25% of dataset) to reduce noise from bots/bulk buyers.

* **Category Missing Value Handling**

  * Found **5.57% NaNs** in `categoryid`.
  * Forward-filled category IDs within each `itemid` timeline to preserve chronological consistency.
  * Marked unknown categories for tracking.

* **Merging with Category Tree**

  * Joined `events_filtered` with `category_tree.csv`.
  * Extracted `parentid` and `category_depth` for each category.
  * Resolved merge column conflicts (`parentid_x`, `parentid_y`) by renaming to a unified `parentid`.

* **Encoding & Sparse Matrix Preparation**

  * Encoded `visitorid` → `user_id` and `itemid` → `item_id` using `LabelEncoder`.
  * Assigned event weights: `view=1`, `addtocart=3`, `transaction=5`.
  * Created a **sparse CSR interaction matrix**:

    * Shape: **(1,376,968 users × 230,615 items)**
    * Non-zero entries: **4,427,915 interactions**
    * Saved as `data/interaction_sparse.pkl` for modelling.

---

### 📊 Key Findings / Observations

* Extreme outliers can heavily skew recommendations and memory usage.
* Forward-fill method proved reliable for filling missing `categoryid` values in event timelines.
* Sparse matrix storage reduced potential memory usage from **\~2.5 TB** (dense) to a manageable size.
* A random preview of the interaction matrix confirms meaningful user–item connections.

---

### ⚠️ Challenges & Fixes

| Challenge                                                                              | How It Was Resolved                                                   |
| -------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| Duplicate merge columns (`parentid_x` / `parentid_y`) after joining with category tree | Explicitly selected and renamed columns before merge.                 |
| Memory errors when pivoting to dense format                                            | Switched to CSR sparse matrix format using `scipy.sparse.csr_matrix`. |
| Empty preview when sampling interactions                                               | Adjusted sampling to ensure overlap between selected users and items. |

---

### 🔜 Next Steps

* Conduct **bonus EDA** to strengthen final report:

  1. **Time-based trends** – interaction patterns over time.
  2. **Category-level popularity** – most purchased/engaged categories.
  3. **Repeat user behaviour** – retention and purchase patterns.
  4. **Conversion funnel** – view → add-to-cart → purchase rates.
* Implement **ALS recommender model** using the sparse matrix.
* Evaluate recommendation quality and diversity.

---

### 📂 Files Updated / Created

| File                                       | Description                                                     |
| ------------------------------------------ | --------------------------------------------------------------- |
| `notebooks/Recommendation System Project.ipynb` | Outlier filtering, missing value handling, category enrichment. |
| `data/category_tree.csv`                   | Category hierarchy data for enrichment.                         |
| `data/interaction_sparse.pkl`              | Sparse user–item interaction matrix for model training.         |

---

### 🧠 Key Learnings

* Outlier removal is a critical preprocessing step for scalable recommender systems.
* Sparse matrices are essential for handling large-scale interaction data efficiently.
* Careful handling of merge keys avoids data duplication or column conflicts.

---

