# --- Robust normalizer for uploaded / snapshot events ---
def normalize_events(df: pd.DataFrame) -> pd.DataFrame:
    """Return a frame with columns: visitorid(int64), itemid(int64), ts(datetime)"""
    # Accept common alternative names
    rename_map = {}
    cols_lower = {c.lower(): c for c in df.columns}
    if "visitorid" not in df.columns and "visitor_id" in cols_lower:
        rename_map[cols_lower["visitor_id"]] = "visitorid"
    if "itemid" not in df.columns and "item_id" in cols_lower:
        rename_map[cols_lower["item_id"]] = "itemid"
    if "timestamp" not in df.columns and "time" in cols_lower:
        rename_map[cols_lower["time"]] = "timestamp"
    if rename_map:
        df = df.rename(columns=rename_map)

    need = {"visitorid","itemid","timestamp"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Events file must contain columns {need}. Missing: {missing}")

    out = df.copy()

    # Parse timestamp (ms epoch or ISO)
    ts = out["timestamp"]
    if np.issubdtype(ts.dtype, np.number):
        out["ts"] = pd.to_datetime(ts, unit="ms", errors="coerce")
    else:
        out["ts"] = pd.to_datetime(ts, errors="coerce")

    # Coerce ids to numeric, drop bad rows
    out["visitorid"] = pd.to_numeric(out["visitorid"], errors="coerce")
    out["itemid"]    = pd.to_numeric(out["itemid"],    errors="coerce")
    out = out.dropna(subset=["visitorid","itemid","ts"]).copy()

    # Unify dtypes to match co-vis table
    out["visitorid"] = out["visitorid"].astype("int64")
    out["itemid"]    = out["itemid"].astype("int64")

    # Sort
    out = out.sort_values(["visitorid","ts"]).reset_index(drop=True)
    return out

def suggest_visitors_with_neighbors(ev_norm: pd.DataFrame, covis: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Return visitor IDs whose last item exists in the co-vis table (so recs will work)."""
    # visitors and their last item
    last_item = (
        ev_norm.groupby("visitorid", as_index=False)
               .tail(1)[["visitorid","itemid"]]
               .rename(columns={"itemid":"last_item"})
    )
    # which last items have neighbors
    items_with_neighbors = set(covis["itemid"].unique())
    good = last_item[last_item["last_item"].isin(items_with_neighbors)].copy()
    return good.head(top_n)

# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import json, joblib
from pathlib import Path
from collections import Counter, defaultdict
from typing import Optional, List

# ---------- paths that work both locally and on streamlit cloud ----------
HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
ART = HERE / "recsys_artifacts"
assert ART.exists(), f"Artifacts folder not found at {ART}"

# ---------- artifacts expected ----------
# task 1a: category-from-views ranker (logistic regression)
TASK1_MODEL_FILE = ART / "task1_ranking_best_logistic_regression.joblib"
# task 1b: next-item recommender (covisitation)
COVIS_FILE = ART / "task1_next_item_covis.parquet"
SEEN_FILE  = ART / "task1_seen_items.parquet"
# optional events snapshots (cleaned)
SNAPSHOT_FILES = [
    ART / "events_clean_frozen.parquet",
    ART / "events_clean_q04.parquet",
    ART / "events_clean.parquet",
]
# task 2: anomaly detection
TASK2_MODEL_FILE = ART / "task2_isolation_forest.joblib"
TASK2_OP_FILE    = ART / "task2_operating_point.json"

# ---------- shared utils ----------
def ensure_ts_series(ts_like):
    """parse ms epoch or ISO string to pandas.Timestamp"""
    try:
        return pd.to_datetime(int(ts_like), unit="ms")
    except Exception:
        return pd.to_datetime(ts_like)

def ensure_ts_df(df: pd.DataFrame, col="timestamp") -> pd.DataFrame:
    out = df.copy()
    if np.issubdtype(out[col].dtype, np.number):
        out["ts"] = pd.to_datetime(out[col], unit="ms", errors="coerce")
    else:
        out["ts"] = pd.to_datetime(out[col], errors="coerce")
    return out

@st.cache_data
def load_snapshot_default() -> Optional[pd.DataFrame]:
    for p in SNAPSHOT_FILES:
        if p.exists():
            return pd.read_parquet(p)
    return None

# =========================================================
# =============  PANEL A: Next-Item Recommender ===========
# =========================================================

@st.cache_data
def load_covis_and_seen():
    covis, seen = None, None
    if COVIS_FILE.exists():
        covis = pd.read_parquet(COVIS_FILE)
        covis = covis.astype({"itemid":"int64","neighbor":"int64","score":"float32"})
    if SEEN_FILE.exists():
        seen = pd.read_parquet(SEEN_FILE)
        # seen schema: visitorid | seen_items(list[int])
    return covis, seen

def last_k_items_for_user(ev: pd.DataFrame, user_id: int, k: int = 3) -> List[int]:
    if ev is None or ev.empty:
        return []
    sub = ev.loc[ev["visitorid"] == user_id, ["timestamp","itemid"]]
    if sub.empty:
        return []
    sub = ensure_ts_df(sub).sort_values("ts")
    return list(pd.unique(sub["itemid"].astype("int64").tail(k)))[::-1]  # most-recent-first

def recommend_from_covis(covis: pd.DataFrame,
                         last_items: List[int],
                         k: int = 10,
                         exclude: Optional[set] = None) -> pd.DataFrame:
    """
    Sum neighbor scores from the co-vis table over the last items (with mild recency weighting).
    Falls back to global popularity from COVIS if last_items is empty.
    """
    if covis is None or covis.empty:
        return pd.DataFrame(columns=["rank","itemid","score"])

    if exclude is None:
        exclude = set()

    # popularity fallback
    if not last_items:
        pop = (covis.groupby("neighbor", as_index=False)["score"]
                     .sum()
                     .sort_values("score", ascending=False))
        pop = pop.loc[~pop["neighbor"].isin(exclude)].head(k).reset_index(drop=True)
        pop.insert(0, "rank", np.arange(1, len(pop)+1))
        pop = pop.rename(columns={"neighbor":"itemid"})
        return pop[["rank","itemid","score"]]

    # recency weights (most recent gets weight 1.0)
    recency = np.linspace(1.0, 0.6, num=len(last_items))
    scores = defaultdict(float)

    # accumulate neighbor scores
    for r_w, it in zip(recency, last_items):
        neigh = covis.loc[covis["itemid"] == int(it), ["neighbor","score"]]
        for n, s in neigh.itertuples(index=False):
            if n in exclude or n in last_items:
                continue
            scores[int(n)] += float(r_w) * float(s)

    if not scores:
        return recommend_from_covis(covis, [], k=k, exclude=exclude)

    recs = (pd.DataFrame([{"itemid": i, "score": sc} for i, sc in scores.items()])
              .sort_values("score", ascending=False)
              .head(k)
              .reset_index(drop=True))
    recs.insert(0, "rank", np.arange(1, len(recs)+1))
    return recs[["rank","itemid","score"]]

# =========================================================
# ==========  PANEL B: Property-from-Views (Task 1A) ======
# =========================================================

TASK1_FEATURES = ["cnt","share","is_last","last_gap_min","pos_from_end","tail_streak_if_last","n_views"]

@st.cache_resource
def load_task1_model():
    if TASK1_MODEL_FILE.exists():
        return joblib.load(TASK1_MODEL_FILE)
    return None

@st.cache_data
def load_category_tree():
    candidates = [
        ART / "category_tree.csv",
        HERE / "data/category_tree.csv",
        HERE / "category_tree.csv",
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            cols = {c.lower(): c for c in df.columns}
            cid = cols.get("categoryid","categoryid")
            pid = cols.get("parentid","parentid")
            if cid in df.columns and pid in df.columns:
                return (df.rename(columns={cid:"categoryid", pid:"parentid"})
                          [["categoryid","parentid"]]
                          .drop_duplicates())
    return None

def candidates_from_views(view_df: pd.DataFrame) -> pd.DataFrame:
    if view_df.empty:
        return pd.DataFrame(columns=["candidate_cat"] + TASK1_FEATURES)
    v = view_df.sort_values("ts").copy()
    v["categoryid_final"] = v["categoryid_final"].astype(int)

    cats = v["categoryid_final"].tolist()
    counts = Counter(cats)
    n = len(cats)
    last_cat = cats[-1]
    last_pos = {c: i for i, c in enumerate(cats)}

    # tail streak length for the last category
    tail_streak = 1
    for i in range(n - 2, -1, -1):
        if cats[i] == last_cat:
            tail_streak += 1
        else:
            break

    now_ts = v["ts"].max()
    rows = []
    for c, cnt in counts.items():
        li = last_pos[c]
        last_gap_min = float((now_ts - v.iloc[li]["ts"]).total_seconds() / 60.0)
        rows.append({
            "candidate_cat": int(c),
            "cnt": int(cnt),
            "share": cnt / n,
            "is_last": int(c == last_cat),
            "last_gap_min": last_gap_min,
            "pos_from_end": int(n - li),
            "tail_streak_if_last": int(tail_streak if c == last_cat else 0),
            "n_views": int(n)
        })
    return pd.DataFrame(rows)

def rank_with_model(cand: pd.DataFrame, k: int):
    if cand.empty:
        return cand
    mdl = load_task1_model()
    if mdl is None:
        # fallback to simple rule if model not shipped
        cand["score"] = cand["cnt"] + 0.5 * (1.0 / (1.0 + cand["last_gap_min"]))
        cand["source"] = "rule"
    else:
        try:
            X = cand[TASK1_FEATURES].astype("float32").to_numpy()
            proba = mdl.predict_proba(X)[:, 1]
            cand["score"] = proba
            cand["source"] = "model"
        except Exception:
            cand["score"] = cand["cnt"] + 0.5 * (1.0 / (1.0 + cand["last_gap_min"]))
            cand["source"] = "rule"
    cand = cand.sort_values("score", ascending=False).head(k).reset_index(drop=True)
    cand.insert(0, "rank", np.arange(1, len(cand) + 1))
    return cand[["rank","candidate_cat","score","source"]]

def enrich_with_parent(out: pd.DataFrame) -> pd.DataFrame:
    tree = load_category_tree()
    if tree is None or out.empty:
        return out
    merged = out.merge(tree, left_on="candidate_cat", right_on="categoryid", how="left").drop(columns=["categoryid"])
    cols = ["rank","candidate_cat","parentid","score","source"]
    return merged[[c for c in cols if c in merged.columns]]

# =========================================================
# ==============  PANEL C: Anomaly Detection  =============
# =========================================================

@st.cache_resource
def load_task2_model_and_op():
    if TASK2_MODEL_FILE.exists() and TASK2_OP_FILE.exists():
        iso = joblib.load(TASK2_MODEL_FILE)
        op  = json.load(open(TASK2_OP_FILE))
        return iso, op
    return None, None

def score_and_flag_users(user_feats: pd.DataFrame, share_override: Optional[float] = None) -> pd.DataFrame:
    iso, op = load_task2_model_and_op()
    if iso is None or op is None:
        raise RuntimeError("Isolation Forest model or operating point not found.")
    feat_cols = op["feature_cols"]
    X = user_feats[feat_cols].to_numpy("float32")
    scores = -iso.decision_function(X)
    thr = float(op["threshold"]) if share_override is None else float(np.quantile(scores, 1 - float(share_override)))
    out = user_feats[["visitorid"]].copy()
    out["anom_score"] = scores
    out["is_anom"] = (scores >= thr).astype(int)
    out["threshold_used"] = thr
    return out

# =========================================================
# ========================   UI   =========================
# =========================================================

st.set_page_config(page_title="Recsys Inference", layout="wide")
st.title("Recommendation System Inference")

with st.sidebar:
    st.markdown("**Artifacts present**")
    st.write("COVIS table:", COVIS_FILE.exists())
    st.write("Seen table:", SEEN_FILE.exists())
    st.write("Task 1 (category) model:", TASK1_MODEL_FILE.exists())
    st.write("Task 2 model:", TASK2_MODEL_FILE.exists())
    st.write("Task 2 operating point:", TASK2_OP_FILE.exists())

tabA, tabB, tabC = st.tabs(["Next-Item Recommender", "Predict Category from Views", "Anomaly Detection"])

# ---------- Tab A: Next-Item ----------
with tabA:
    st.subheader("Recommend next items from recent activity (co-visitation)")
    covis, seen_tbl = load_covis_and_seen()
    ev = load_snapshot_default()

    # input options
    col1, col2 = st.columns([1,1])
    with col1:
        uid = st.number_input("Visitor ID", min_value=0, value=0, step=1, help="Enter a known visitorid from your dataset")
        k = st.number_input("Top-K items", min_value=1, max_value=50, value=8, step=1)
        use_events = st.checkbox("Use cleaned events snapshot to derive last items", value=True)
        exclude_seen = st.checkbox("Exclude already seen items", value=True)
    with col2:
        st.caption("Optional: Upload an events file (csv or parquet) if you want to test with fresh data")
        uploaded = st.file_uploader("Upload events", type=["csv","parquet"])
        if uploaded is not None:
            ev = pd.read_parquet(uploaded) if uploaded.name.endswith(".parquet") else pd.read_csv(uploaded)

    # determine last items
    last_items = []
    if use_events and ev is not None:
        required_cols = {"visitorid","itemid","timestamp"}
        if required_cols.issubset(set(ev.columns)):
            last_items = last_k_items_for_user(ev, int(uid), k=3)
    # otherwise leave last_items empty and fall back to popularity in covis

    # exclusion set
    exclude_set = set()
    if exclude_seen and seen_tbl is not None and "seen_items" in seen_tbl.columns:
        row = seen_tbl.loc[seen_tbl["visitorid"] == int(uid)]
        if not row.empty:
            try:
                # seen_items stored as list; if serialized as string, try to eval safely
                vals = row.iloc[0]["seen_items"]
                if isinstance(vals, str):
                    import ast
                    vals = ast.literal_eval(vals)
                exclude_set = set(int(x) for x in vals)
            except Exception:
                exclude_set = set()

    if st.button("Recommend"):
        try:
            recs = recommend_from_covis(covis, last_items=last_items, k=int(k), exclude=exclude_set)
            if last_items:
                st.write(f"Last items used (most-recent-first): {last_items}")
            else:
                st.write("No last items found for this user — falling back to global popularity from co-visitation.")

            if recs.empty:
                st.warning("No recommendations could be produced. Make sure your co-visitation table has neighbors.")
            else:
                st.dataframe(recs, use_container_width=True)
                st.download_button(
                    "Download recommendations CSV",
                    recs.to_csv(index=False).encode("utf-8"),
                    file_name=f"next_item_recs_user_{uid}.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"Recommendation failed: {e}")

# ---------- Tab B: Predict Category / Property from Views ----------
with tabB:
    st.subheader("Predict next add-to-cart category from recent views")
    st.caption("Paste a small table of this user’s recent views (timestamp + categoryid_final).")

    left, right = st.columns([2, 1])
    with left:
        sample = pd.DataFrame({
            "timestamp": ["1690000000000","1690000300000","1690000600000","1690000900000"],
            "categoryid_final": [1613,1509,1613,1613],
        })
        df_in = st.data_editor(sample, num_rows="dynamic", use_container_width=True)
        kk = st.number_input("Top-K categories", min_value=1, max_value=10, value=3, step=1)
        if st.button("Compute predicted categories"):
            try:
                views = pd.DataFrame({
                    "ts": df_in["timestamp"].apply(ensure_ts_series),
                    "categoryid_final": df_in["categoryid_final"].astype(int)
                }).dropna()
                cand = candidates_from_views(views)
                out = rank_with_model(cand, k=int(kk))
                out = enrich_with_parent(out)
                st.dataframe(out, use_container_width=True)
                st.download_button(
                    "Download CSV",
                    out.to_csv(index=False).encode("utf-8"),
                    file_name="predicted_categories.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(f"Could not compute categories: {e}")

    with right:
        st.caption("Or pick a visitor from your cleaned snapshot (if present).")
        ev_snap = load_snapshot_default()
        if ev_snap is None or ev_snap.empty or "categoryid_final" not in ev_snap.columns:
            st.info("No events snapshot found (or missing categoryid_final).")
        else:
            uids = ev_snap["visitorid"].dropna().astype(int).unique()
            try:
                pick_uid = st.selectbox("Visitor", options=sorted(uids)[:5000])
                dfu = ev_snap.loc[(ev_snap["visitorid"] == pick_uid) & (ev_snap["event"] == "view"), ["timestamp","categoryid_final"]]
                if not dfu.empty:
                    dfu = ensure_ts_df(dfu)
                    cand = candidates_from_views(dfu[["ts","categoryid_final"]])
                    out = rank_with_model(cand, k=3)
                    out = enrich_with_parent(out)
                    st.dataframe(out, use_container_width=True)
            except Exception:
                pass

# ---------- Tab C: Anomaly Detection ----------
with tabC:
    st.subheader("Flag anomalous users (Isolation Forest)")
    uploaded_feats = st.file_uploader("Upload user features (parquet or csv)", type=["parquet","csv"])
    if uploaded_feats is not None:
        user_feats = pd.read_parquet(uploaded_feats) if uploaded_feats.name.endswith(".parquet") else pd.read_csv(uploaded_feats)
    else:
        # try default features (if you shipped them)
        default_feats = ART / "task2_user_features.parquet"
        user_feats = pd.read_parquet(default_feats) if default_feats.exists() else None

    if user_feats is None:
        st.info("No user features provided. Upload a file or place task2_user_features.parquet in recsys_artifacts.")
    else:
        st.write("Rows:", len(user_feats))
        with st.expander("Preview"):
            st.dataframe(user_feats.head(10), use_container_width=True)

        use_frozen = st.checkbox("Use frozen operating point", value=True)
        share_override = None if use_frozen else st.slider("Flagged share", 0.01, 0.20, 0.04, 0.01)

        if st.button("Score users"):
            try:
                scored = score_and_flag_users(user_feats, share_override=share_override)
                st.success(f"Flagged {int(scored['is_anom'].sum())} users")
                st.dataframe(scored.sort_values("anom_score", ascending=False).head(50), use_container_width=True)
                st.download_button("Download scored users CSV",
                                   scored.to_csv(index=False).encode("utf-8"),
                                   file_name="task2_scored_users.csv",
                                   mime="text/csv")
            except Exception as e:
                st.error(f"Scoring failed: {e}")

