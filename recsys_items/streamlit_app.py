# streamlit_app.py
import os, json, glob
from typing import Tuple

import numpy as np
import pandas as pd
import joblib
import streamlit as st
from collections import Counter

APP_DIR = os.path.dirname(__file__)  # current folder (e.g., recsys_items)

# ============================== small helpers ==============================
def file_here(name: str) -> str | None:
    p = os.path.join(APP_DIR, name)
    return p if os.path.exists(p) else None

# =================== Task 1: Recommender loaders + helpers ===================
@st.cache_resource
def load_model_and_meta() -> Tuple[object | None, dict | None, str]:
    """Load model + metadata if present; else return (None, None, 'baseline')."""
    model_path = file_here("best_candidate_ranker.pkl")
    meta_path  = file_here("inference_metadata.json")
    if model_path and meta_path:
        try:
            model = joblib.load(model_path)
            with open(meta_path) as f:
                meta = json.load(f)
            mode = "hybrid" if meta.get("feature_set") == "hybrid_svd" else "baseline_model"
            return model, meta, mode
        except Exception as e:
            st.warning(f"Could not load model/metadata; using Baseline. Details: {e}")
    return None, None, "baseline"

@st.cache_data
def load_embeddings():
    """Load SVD embeddings (or shards) if present; else return empty DFs."""
    u_path = file_here("svd_user_factors.parquet")
    v_path = file_here("svd_item_factors.parquet")

    if u_path and v_path:
        U = pd.read_parquet(u_path)
        V = pd.read_parquet(v_path)
    else:
        # fallback to user shards if present
        u_shards = sorted(glob.glob(os.path.join(APP_DIR, "svd_user_factors_part*.parquet")))
        if u_shards and v_path:
            U = pd.concat([pd.read_parquet(p) for p in u_shards], ignore_index=True)
            V = pd.read_parquet(v_path)
        else:
            return pd.DataFrame(), pd.DataFrame()

    if "visitorid" in U: U["visitorid"] = U["visitorid"].astype("int64")
    if "candidate_cat" in V: V["candidate_cat"] = V["candidate_cat"].astype("int64")
    return U, V

@st.cache_data
def try_load_events(uploaded, default_path: str) -> pd.DataFrame:
    # uploaded CSV/Parquet takes priority
    if uploaded is not None:
        try:
            if uploaded.name.lower().endswith(".parquet"):
                return pd.read_parquet(uploaded)
            return pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read uploaded file: {e}")

    # repo file (CSV or Parquet)
    try:
        path = os.path.join(APP_DIR, default_path)
        if default_path.lower().endswith(".parquet"):
            return pd.read_parquet(path)
        return pd.read_csv(path)
    except Exception:
        pass

    # optional fallback shards (big local datasets)
    shards = sorted(glob.glob(os.path.join(APP_DIR, "events_enriched_part*.parquet")))
    if shards:
        return pd.concat([pd.read_parquet(p) for p in shards], ignore_index=True)

    return pd.DataFrame()

@st.cache_data
def get_active_users(events: pd.DataFrame, limit: int = 500) -> pd.DataFrame:
    need = {"event","visitorid"}
    if events.empty or need - set(events.columns):
        return pd.DataFrame(columns=["visitorid","n_views"])
    v = events[events["event"] == "view"]
    counts = (v.groupby("visitorid").size()
                .sort_values(ascending=False)
                .head(limit)
                .rename("n_views")
                .reset_index())
    counts["visitorid"] = counts["visitorid"].astype("int64")
    return counts

def user_debug(events: pd.DataFrame, uid: int, time_window_sec: int, views_lookback: int) -> dict:
    out = {"has_rows": False, "n_views_total": 0, "n_views_window": 0}
    cols = ["timestamp","visitorid","event","categoryid_final"]
    if not set(cols).issubset(events.columns):
        return out
    df0 = events.loc[events["visitorid"] == uid, cols].dropna().copy()
    if df0.empty: return out
    out["has_rows"] = True
    if np.issubdtype(df0["timestamp"].dtype, np.datetime64):
        df0["ts"] = pd.to_datetime(df0["timestamp"])
    else:
        df0["ts"] = pd.to_datetime(pd.to_numeric(df0["timestamp"], errors="coerce"), unit="ms")
    v = df0[df0["event"] == "view"][["ts","categoryid_final"]].copy()
    out["n_views_total"] = int(len(v))
    if v.empty: return out
    t_end = v["ts"].max()
    v2 = v[(v["ts"] >= t_end - pd.Timedelta(seconds=time_window_sec)) & (v["ts"] <= t_end)].tail(views_lookback)
    out["n_views_window"] = int(len(v2))
    out["t_last"] = str(t_end)
    return out

def build_recent_candidates(events_enriched: pd.DataFrame, user_id: int,
                            time_window_sec: int = 24*60*60, views_lookback: int = 100) -> pd.DataFrame:
    cols = ["timestamp","visitorid","event","categoryid_final"]
    if not set(cols).issubset(events_enriched.columns):
        return pd.DataFrame(columns=["visitorid","candidate_cat","cnt","share","is_last",
                                     "last_gap_min","pos_from_end","tail_streak_if_last","n_views"])
    df0 = events_enriched.loc[events_enriched["visitorid"] == user_id, cols].dropna().copy()
    if df0.empty:
        return pd.DataFrame(columns=["visitorid","candidate_cat","cnt","share","is_last",
                                     "last_gap_min","pos_from_end","tail_streak_if_last","n_views"])
    df0 = df0.rename(columns={"categoryid_final":"categoryid"}).astype({"categoryid":"int64"})
    if np.issubdtype(df0["timestamp"].dtype, np.datetime64):
        df0["ts"] = pd.to_datetime(df0["timestamp"])
    else:
        df0["ts"] = pd.to_datetime(pd.to_numeric(df0["timestamp"], errors="coerce"), unit="ms")

    views = df0[df0["event"] == "view"][["ts","categoryid"]].copy()
    if views.empty:
        return pd.DataFrame(columns=["visitorid","candidate_cat","cnt","share","is_last",
                                     "last_gap_min","pos_from_end","tail_streak_if_last","n_views"])

    t_end = views["ts"].max()
    v = views[(views["ts"] >= t_end - pd.Timedelta(seconds=time_window_sec)) & (views["ts"] <= t_end)].tail(views_lookback)

    cats = v["categoryid"].astype(int).tolist()
    if not cats:
        return pd.DataFrame(columns=["visitorid","candidate_cat","cnt","share","is_last",
                                     "last_gap_min","pos_from_end","tail_streak_if_last","n_views"])

    n = len(cats)
    counts = Counter(cats)
    last_pos = {c:i for i,c in enumerate(cats)}
    last_cat = cats[-1]

    tail_streak = 1
    for i in range(n - 2, -1, -1):
        if cats[i] == last_cat:
            tail_streak += 1
        else:
            break

    rows = []
    for c, cnt in counts.items():
        li = last_pos[c]; last_ts_c = v.iloc[li]["ts"]
        rows.append({
            "visitorid": int(user_id),
            "candidate_cat": int(c),
            "cnt": int(cnt),
            "share": cnt / n,
            "is_last": int(c == last_cat),
            "last_gap_min": float((t_end - last_ts_c).total_seconds() / 60.0),
            "pos_from_end": int(n - li),
            "tail_streak_if_last": int(tail_streak if c == last_cat else 0),
            "n_views": int(n)
        })
    return pd.DataFrame(rows)

def score_candidates_baseline(candidates: pd.DataFrame, topk: int = 5, alpha: float = 0.5) -> pd.DataFrame:
    X = candidates.copy()
    X["score"] = X["cnt"] + alpha * (1.0 / (1.0 + X["last_gap_min"]))
    out = X[["visitorid","candidate_cat","score"]].sort_values("score", ascending=False).head(topk).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out

def score_candidates_hybrid(candidates: pd.DataFrame, model, meta,
                            U: pd.DataFrame, V: pd.DataFrame, topk: int = 5) -> pd.DataFrame:
    svd_u_cols = meta["svd_user_cols"]; svd_i_cols = meta["svd_item_cols"]; hyb_features = meta["hybrid_features"]
    if U.empty or V.empty:
        return score_candidates_baseline(candidates, topk=topk)
    X = (candidates.assign(visitorid=lambda d: d["visitorid"].astype("int64"),
                           candidate_cat=lambda d: d["candidate_cat"].astype("int64"))
                    .merge(U, on="visitorid", how="left")
                    .merge(V, on="candidate_cat", how="left"))
    X[svd_u_cols + svd_i_cols] = X[svd_u_cols + svd_i_cols].fillna(0.0)
    X_mat = X[hyb_features].astype(np.float32).to_numpy()
    proba = getattr(model, "predict_proba", None)
    scores = proba(X_mat)[:, 1] if proba else model.decision_function(X_mat)
    out = X[["visitorid","candidate_cat"]].copy()
    out["score"] = scores
    out = out.sort_values("score", ascending=False).head(topk).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out

# =================== Task 2: Anomaly (IsolationForest) ===================
@st.cache_resource
def load_task2_artifacts():
    """Load IsolationForest model, user features, flagged users, and feature list."""
    def _maybe(name): 
        p = os.path.join(APP_DIR, name); 
        return p if os.path.exists(p) else None

    model_path   = _maybe("task2_isolation_forest.joblib")
    feats_path   = _maybe("task2_user_features.parquet")
    flagged_path = _maybe("task2_flagged_users.csv")
    cfg_path     = _maybe("task2_config.json")

    model   = joblib.load(model_path) if model_path else None
    feats   = pd.read_parquet(feats_path) if feats_path else pd.DataFrame()
    flagged = pd.read_csv(flagged_path) if flagged_path else pd.DataFrame()
    cfg     = json.load(open(cfg_path)) if cfg_path else {}

    # Decide feature list
    feature_cols = cfg.get("feature_cols")
    if not feature_cols and not feats.empty:
        drop_cols = {"visitorid", "anom_score", "proxy_anom"}
        feature_cols = [c for c in feats.columns
                        if c not in drop_cols and pd.api.types.is_numeric_dtype(feats[c])]
        feature_cols = sorted(feature_cols)

    return model, feats, flagged, cfg, feature_cols

def iforest_score_user(uid: int):
    """Return (result_dict, warning_str). Never raises on shape mismatch."""
    model, feats, _, cfg, feature_cols = load_task2_artifacts()
    if model is None or feats.empty or not feature_cols:
        return None, "Artifacts missing (model/features)."

    row = feats.loc[feats["visitorid"] == uid]
    if row.empty:
        return None, "No precomputed features for this user."

    X = row[feature_cols].astype(np.float32).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    want = getattr(model, "n_features_in_", X.shape[1])

    if X.shape[1] != want:
        if len(feature_cols) >= want:
            use_cols = feature_cols[:want]  # deterministic subset
            X = row[use_cols].astype(np.float32).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            warn = f"Feature mismatch fixed by selecting first {want} of {len(feature_cols)} features."
        else:
            return None, f"Per-user scoring unavailable: only {X.shape[1]} features present, model expects {want}."
    else:
        warn = None

    pred = int(model.predict(X)[0] == -1)          # 1 if anomalous
    ascore = float(-model.score_samples(X)[0])     # larger => more anomalous
    out = {"visitorid": int(uid), "flagged": pred, "anomaly_score": ascore, "n_features_used": int(X.shape[1])}
    return out, warn

# ================================ UI =================================
st.set_page_config(page_title="Hybrid Recommender", page_icon="🧠", layout="wide")

tab_rec, tab_anom = st.tabs(["Recommendations", "Anomaly audit"])

# --------------------------- Recommendations ---------------------------
with tab_rec:
    st.title("🧠 Hybrid Category Recommender")

    model, META, mode = load_model_and_meta()
    U, V = load_embeddings()

    st.sidebar.header("Data source")
    uploaded = st.sidebar.file_uploader("Upload CSV/Parquet", type=["csv","parquet"])
    default_path = st.sidebar.text_input("or repo file path", value="sample_events.parquet")

    events_enriched = try_load_events(uploaded, default_path)

    if model is None or META is None:
        st.info("Running in **Baseline mode** (no model / embeddings found). "
                "Push `best_candidate_ranker.pkl` and `inference_metadata.json` to enable Hybrid mode.")
    else:
        st.success(f"Running in **Hybrid mode** ({META.get('champion_tag','model')}).")

    if events_enriched.empty:
        st.warning("Provide data with columns: `timestamp, visitorid, event, categoryid_final`.")
    else:
        st.caption(f"Loaded {len(events_enriched):,} rows, {events_enriched['visitorid'].nunique():,} users.")

        active_df = get_active_users(events_enriched, limit=500)
        pick_from_list = st.checkbox("Pick from top active users", value=True)

        if pick_from_list and not active_df.empty:
            user_id = st.selectbox(
                "User",
                options=active_df["visitorid"].tolist(),
                format_func=lambda u: f"{u}  views={int(active_df.loc[active_df['visitorid']==u,'n_views'].iloc[0])}"
            )
        else:
            default_uid = int(active_df["visitorid"].iloc[0]) if not active_df.empty else int(events_enriched["visitorid"].iloc[0])
            user_id = st.number_input("User id", min_value=0, value=default_uid, step=1)

        user_id = int(user_id)

        # Optional: block recommendations for anomalous users (if artifacts exist)
        exclude_anom = st.checkbox("Exclude flagged/anomalous users", value=True)
        result_t2, warn_t2 = iforest_score_user(user_id)
        if exclude_anom and result_t2 and result_t2["flagged"] == 1:
            st.warning("This user is **flagged** by the anomaly detector.")
            st.error("Recommendations blocked by anomaly policy. Uncheck **Exclude flagged/anomalous users** to override.")
        else:
            colA, colB, colC = st.columns(3)
            with colA:
                topk = st.slider("Top K", 1, 20, 5)
            with colB:
                time_window_hours = st.slider("Lookback hours", 1, 72, 24)
            with colC:
                views_lookback = st.slider("Max recent views", 10, 300, 100, step=10)

            dbg = user_debug(events_enriched, user_id, time_window_hours*3600, views_lookback)
            with st.expander("Debug current user"):
                st.write(dbg)
                if result_t2:
                    st.write({"anomaly_flag": result_t2["flagged"], "anomaly_score": result_t2["anomaly_score"]})
                elif warn_t2:
                    st.info(warn_t2)

            if st.button("Recommend"):
                cand = build_recent_candidates(
                    events_enriched, user_id,
                    time_window_sec=time_window_hours*3600,
                    views_lookback=views_lookback
                )
                if cand.empty:
                    st.warning("No candidates for this user in the selected window.")
                else:
                    if model and META and mode.startswith("hybrid"):
                        recs = score_candidates_hybrid(cand, model, META, U, V, topk=topk)
                    else:
                        recs = score_candidates_baseline(cand, topk=topk, alpha=0.5)
                    st.subheader("Top K Categories")
                    st.dataframe(recs, use_container_width=True)

# ------------------------------ Anomaly audit ------------------------------
with tab_anom:
    st.title("Task 2 — Anomaly audit")

    model_t2, feats_t2, flagged_t2, cfg_t2, feat_cols_t2 = load_task2_artifacts()

    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Flagged users", f"{(len(flagged_t2) if not flagged_t2.empty else 0):,}")
    with c2: st.metric("User features rows", f"{(len(feats_t2) if not feats_t2.empty else 0):,}")
    with c3: st.metric("Model loaded", "Yes" if model_t2 is not None else "No")

    if not flagged_t2.empty:
        st.subheader("Flagged users (preview)")
        st.dataframe(flagged_t2.head(10), use_container_width=True)
        st.download_button(
            "Download flagged users CSV",
            data=flagged_t2.to_csv(index=False).encode("utf-8"),
            file_name="task2_flagged_users.csv",
            mime="text/csv",
        )

    st.subheader("Score a specific user")
    uid_default = int(feats_t2["visitorid"].iloc[0]) if not feats_t2.empty else 0
    uid_input = st.number_input("User id", min_value=0, step=1, value=uid_default)
    result, warn = iforest_score_user(int(uid_input))
    if result is None:
        st.warning(warn)
    else:
        if warn: st.info(warn)
        st.json(result)
