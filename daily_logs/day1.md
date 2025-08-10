📌 **Project**

**Personalised Recommendation System using E-commerce Behaviour Data
Work Summary for Today**

---

Today’s focus was on integrating the three key datasets:
1. events.csv – user interactions (views, add-to-carts, transactions) with timestamps.
2. item_properties_part1.csv and item_properties_part2.csv – historical item attributes (e.g., category, availability) with change timestamps.
3. category_tree.csv – hierarchical category structure.

---

**Goals for the Day**
- Load and combine large datasets without memory errors.
- Extract and clean category mappings from item properties.
- Merge event data with category information.
- Implement time-aware category assignment using Pandas merge_asof or an equivalent method to prevent data leakage.

---

**Steps Completed**

1. Loaded item_properties in chunks to handle large file sizes efficiently:

    chunks1 = pd.read_csv("item_properties_part1.csv", chunksize=1_000_000)
    chunks2 = pd.read_csv("item_properties_part2.csv", chunksize=1_000_000)
    item_props = pd.concat([chunk for chunk in chunks1] + [chunk for chunk in chunks2], ignore_index=True)

2. Extracted category mapping:
- Filtered rows where property == 'categoryid'
- Renamed value → categoryid
- Converted to integer type for consistency.

3. Merged events with static category mapping for initial enrichment.
   
4. Attempted time-aware category assignment:
- Goal: For each event, assign the most recent category at or before the event timestamp.
- Used Pandas merge_asof with by='itemid' and on='timestamp'.
- Encountered persistent ValueError: left keys must be sorted despite multiple sorting and cleaning attempts.

5. Explored alternative forward-fill method:
- Stacked events and category logs per item.
- Sorted chronologically.
- Forward-filled category values down the timeline.
- Still in progress due to data irregularities.

---

**Challenges Faced**

- merge_asof strict sorting requirement: Data contains itemid groups with non-monotonic timestamps.
- Hidden data issues such as:
  - Mixed timestamp types (string, int).
  - Duplicate timestamps for the same itemid.
  - Potential missing category values.
  - Large dataset size makes iterative debugging slower.
 
---

**Next Steps**

1. Run a “bad itemid” diagnostic to identify and fix groups with out-of-order timestamps.
2. Decide whether to:
- Continue fixing data for merge_asof
- Fully switch to the forward-fill timeline approach for robustness.
3. Once category enrichment is stable:
- Integrate category_tree.csv to map root categories and depth.
- Build user and item feature matrices for model training.

---

**Tools Used**

Python: pandas for data loading, cleaning, and merging.
Jupyter Notebook for iterative development and debugging.

---

**Key Learnings**
1. When working with historical property logs, time alignment is critical to avoid future-data leakage in recommender systems.
2. For large, messy datasets, forward-fill after chronological sort can be a safer alternative to merge_asof.
3. Always verify per-group monotonic ordering before attempting time-aware joins.

Author: [Your Name]
Date: 10 Aug 2025
