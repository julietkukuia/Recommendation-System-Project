# 🗓 Day 1 – Data Integration & Time-Aware Category Assignment

**Date:** 2025-08-9  
**Author:** Juliet Fafali Kukuia  
**Project:** Personalised Recommendation System using E-commerce Behaviour Data  

---

## ✅ Tasks Completed
- [x] Loaded and combined **item_properties_part1.csv** and **item_properties_part2.csv** in chunks to avoid memory errors.
- [x] Extracted category mappings where `property == 'categoryid'`, renamed `value` → `categoryid`, and converted to integers.
- [x] Merged **events.csv** with static category mapping for initial enrichment.
- [x] Attempted time-aware category assignment using `pandas.merge_asof` with `by='itemid'` and `on='timestamp'`.
- [x] Explored forward-fill alternative for assigning categories chronologically.

---

## 📊 Key Findings / Observations
- The dataset is large and requires **chunk loading** for processing.
- `merge_asof` failed due to **non-monotonic timestamps** within `itemid` groups.
- Potential data quality issues include:
  - Mixed timestamp formats (string, int).
  - Duplicate timestamps for the same `itemid`.
  - Missing `categoryid` values.
- Forward-fill method could be more robust for messy data, but still requires cleanup.

---

## ⚠️ Challenges & Fixes
| Challenge | How It Was Resolved |
|-----------|---------------------|
| `merge_asof` error: "left keys must be sorted" | Applied multiple sorting attempts on `itemid` + `timestamp`; error persists. |
| Large file size slowing debugging | Used **chunksize** in `read_csv` and concatenation to process in memory safely. |
| Mixed timestamp types | Identified need to normalize timestamps before joins. |

---

## 🔜 Next Steps
- [ ] Run diagnostics to find `itemid` groups with out-of-order timestamps.
- [ ] Decide between:
  - Fixing dataset for `merge_asof`
  - Switching fully to forward-fill approach for robustness.
- [ ] Once category assignment is stable:
  - Integrate **category_tree.csv** for root categories & depth.
  - Build user and item feature matrices for model training.

---

## 📂 Files Updated / Created
| File | Description |
|------|-------------|
| `notebooks/data_integration.ipynb` | Loaded and merged datasets, attempted time-aware join. |
| `data/item_properties_part1.csv` | Source dataset (chunk processed). |
| `data/item_properties_part2.csv` | Source dataset (chunk processed). |
| `data/events.csv` | Source dataset for user interactions. |

---

## 🖼 Screenshots (if applicable)
*(No screenshots for today — focus was on backend integration.)*

---

## 🔗 Related Links
- [Main Project README](../README.md)

---

## 🧠 Key Learnings
- Time-aware joins are sensitive to timestamp order; even slight disorder in grouped data can break merges.
- Forward-fill after chronological sort is a viable backup for time alignment in recommender systems.
- Always validate data per group before attempting joins to avoid silent errors.

---
