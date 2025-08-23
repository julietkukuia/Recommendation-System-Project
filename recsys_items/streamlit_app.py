# streamlit_app.py
import json, glob, os
import numpy as np
import pandas as pd
import joblib
import streamlit as st
from collections import Counter
from typing import Tuple, Optional

APP_DIR = os.path.dirname(__file__)          # .../recsys_items
REPO_ROOT = os.path.dirname(APP_DIR)         # repo root (parent of recsys_items)

# ------------------ helpers ------------------
def file_here(name: str) -> Optional[str]:
    p = os.path.join(APP_DIR, name)
    return p if os.path.exists(p) else None

def resolve_repo_path(pth: str) -> str:
    return pth if os.path.isabs(pth) else os.path.join(APP_DIR, pth)

def find_first(candidates: list[str]) -> Optional[str]:
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None

# ------------------ cached artefact loaders (Task 1) ------------------
@st.cache_resource
def load_model_and_meta() -> Tuple[object | None, dict | None, str]:
    model_path = find_first([
        os.path.join(APP_DIR, "best_candidate_ranker.pkl")
    ])
    meta_path = find_first([
        os.path.join(APP_DIR, "inference_metadata.json")
    ])
    if model_path and meta_path:
        try:
            model = joblib.load(model_path)
            with open(meta_path) as f:
                meta = json.load(f)
            mode = "hybrid" if meta.get("feature_set") == "hybrid_svd" else "baseline_model"
            return model, meta, mode
        except Exception as e:
            st.warning(f"Could not load model/metadata, falling back to baseline. Details: {e}")
    return None, None, "baseline"

@st.cache_data
def load_embeddings():
    u_path = file_here("svd_user_factors.parquet")
    v_path = file_here("svd_item_factors.parquet")

    if not (u_path and v_path):
        # try user shards + single item factors
        u_shards = sorted(glob.glob(os.path.join(APP_DIR, "svd_user_factors_part*.parquet")))
        if u_shards and file_here("svd_item_factors.parquet"):
            U = pd.concat([pd.read_parquet(p) for p in u_shards], ignore_index=True)
            V = pd.read_parquet(file_here("svd_item_factors.parquet"))
        else:
            return pd.DataFrame(), pd.DataFrame()
    else:
        U = pd.read_parquet(u_path)
        V = pd.read_parquet(v_path)

    if "visitorid" in U: U["visitorid"] = U["visitorid"].astype("int64")
    if "candidate_cat" in V: V["candidate_cat"] = V["candidate_cat"].astype("int64")
    return U, V

@st.cache_data
def try_load_events(uploaded, default_path: str) -> pd.DataFrame:
    # 1) Uploaded
    if uploaded is not None:
        try:
            uploaded.seek(0)
            if uploaded.name.lower().endswith(".parquet"):
                return pd.read_parquet(uploaded)
            return pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read uploaded file: {e}")

    # 2) Repo file (relative to app dir)
    try:
        abs_path = resolve_repo_path(default_path)
        if abs_path.lower().endswith(".parquet"):
            return pd.read_parquet(abs_path)
        return pd.read_csv(abs_path)
    except Exception:
        pass

    # 3) Fallback shards
    shards = sorted(glob.glob(os.path.join(APP_DIR, "events_enriched_part*.parquet")))
    if shards:
        return pd.concat([pd.read_parquet(p) for p in shards], ignore_index=True)

    return pd.DataFrame()

# ------------------ Task 2: Anomaly artifacts (optional) ------------------
@st.cache_resource
def load_anomaly_assets():
    """
    Returns dict with any of:
      flagged_df: DataFrame of flagged users (must include a user id column)
      flagged_set: set of user ids
      user_feats: DataFrame of task2 user features (optional)
      iso_model: isolation forest model (optional)
      op: dict with operating point (e.g., threshold or top-q) (optional)
    If nothing is found, returns minimal dict.
    """
    # Candidate locations: in app dir, in repo root, or in 'recsys_artifacts/'
    candidates = lambda name: [
        os.path.join(APP_DIR, name),
        os.path.join(REPO_ROOT, name),
        os.path.join(APP_DIR, "recsys_artifacts", name),
        os.path.join(REPO_ROOT, "recsys_artifacts", name),
    ]

    flagged_path = find_first(candidates("task2_flagged_users.csv"))
    feats_path   = find_first(candidates("task2_user_features.parquet"))
    model_path   = find_first(candidates("task2_isolation_forest.joblib"))
    op_path      = find_first(candidates("task2_operating_point.json"))

    out = {
        "flagged_df": pd.DataFrame(),
        "flagged_set": set(),
        "user_feats": pd.DataFrame(),
        "iso_model": None,
        "op": {},
    }

    try:
        if flagged_path:
            fdf = pd.read_csv(flagged_path)
            # detect id col (visitorid, user_id, etc.)
            id_col = None
            for c in fdf.columns:
                if "visitor" in c.lower() or c.lower() in {"user","userid","user_id"}:
                    id_col = c; break
            if id_col is not None:
                fdf[id_col] = fdf[id_col].astype("int64")
                out["flagged_df"] = fdf
                out["flagged_set"] = set(fdf[id_col].tolist())
    except Exception as e:
        st.warning(f"Could not read task2_flagged_users.csv: {e}")

    try:
        if feats_path:
            out["user_feats"] = pd.read_parquet(feats_path)
    except Exception as e:
        st.warning(f"Could not read task2_user_features.parquet: {e}")

    try:
        if model_path:
            out["iso_model"] = joblib.load(model_path)
    except Exception as e:
        st.warning(f"Could not load task2_isolation_forest.joblib: {e}")

    try:
        if op_path:
            with open(op_path) as f:
                out["op"] = json.load(f)
    except Exception as e:
        st.warning(f"Could not read task2_operating_point.json: {e}")

    return out

# ------------------ utilities: active user + debug ------------------
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

# ------------------ feature engineering (Task 1) ------------------
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
        if cats[i] == last_cat: tail_streak += 1
        else: break

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

# ------------------ scoring (Task 1) ------------------
def score_candidates_baseline(candidates: pd.DataFrame, topk: int = 5, alpha: float = 0.5) -> pd.DataFrame:
    X = candidates.copy()
    X["score"] = X["cnt"] + alpha * (1.0 / (1.0 + X["last_gap_min"]))
    out = X[["visitorid","candidate_cat","score"]].sort_values("score", ascending=False).head(topk).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out

def score_candidates_hybrid(candidates: pd.DataFrame, model, meta, U: pd.DataFrame, V: pd.DataFrame, topk: int = 5) -> pd.DataFrame:
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

# ============================ UI ============================
st.set_page_config(page_title="Hybrid Recommender", page_icon="🧠", layout="centered")
st.title("🧠 Hybrid Category Recommender")

model, META, mode = load_model_and_meta()
U, V = load_embeddings()
ANOM = load_anomaly_assets()

tabs = st.tabs(["Recommendations", "Anomaly audit"])

with tabs[0]:
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

        # Anomaly gate (optional, only if flagged list available)
        honor_anomaly_filter = st.checkbox("Exclude flagged/anomalous users", value=True,
                                           help="If checked and the selected user is flagged by Task 2, "
                                                "recommendations will be blocked.")
        is_flagged = user_id in ANOM.get("flagged_set", set())
        if is_flagged:
            st.warning("This user is **flagged** by the anomaly detector.")
        else:
            st.info("This user is not flagged by the anomaly detector.") if ANOM.get("flagged_set") else st.caption("Anomaly list not loaded.")

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

        btn_disabled = honor_anomaly_filter and is_flagged
        if btn_disabled:
            st.error("Recommendations blocked by anomaly policy. Uncheck **Exclude flagged/anomalous users** to override.")

        if st.button("Recommend", disabled=btn_disabled):
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

with tabs[1]:
    st.subheader("Task 2 — Anomaly audit")
    have_any = any([not ANOM["flagged_df"].empty, not ANOM["user_feats"].empty, ANOM["iso_model"] is not None])
    if not have_any:
        st.info(
            "No anomaly artifacts found. To enable this tab, add any of these files to the repo "
            "(either in **recsys_items/**, repo root, or **recsys_artifacts/**):\n"
            "- task2_flagged_users.csv\n- task2_user_features.parquet\n- task2_isolation_forest.joblib\n- task2_operating_point.json"
        )
    else:
        # Summary cards
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Flagged users",
                      value=f"{len(ANOM['flagged_set']):,}" if ANOM["flagged_set"] else "—")
        with c2:
            st.metric("User features rows",
                      value=f"{len(ANOM['user_feats']):,}" if not ANOM["user_feats"].empty else "—")
        with c3:
            st.metric("Model loaded",
                      value="Yes" if ANOM["iso_model"] is not None else "No")

        # Show flagged table (capped for UI)
        if not ANOM["flagged_df"].empty:
            st.markdown("**Flagged users (preview)**")
            st.dataframe(ANOM["flagged_df"].head(500), use_container_width=True)
            csv = ANOM["flagged_df"].to_csv(index=False).encode("utf-8")
            st.download_button("Download flagged users CSV", data=csv, file_name="task2_flagged_users.csv", mime="text/csv")

        # Per-user score if we have features + model
        if (ANOM["iso_model"] is not None) and (not ANOM["user_feats"].empty):
            st.markdown("---")
            st.markdown("**Score a specific user**")
            try:
                # detect id column in features
                uf = ANOM["user_feats"]
                id_col = None
                for c in uf.columns:
                    if "visitor" in c.lower() or c.lower() in {"user","userid","user_id"}:
                        id_col = c; break
                if id_col is None:
                    st.warning("Could not detect user id column in task2_user_features.parquet.")
                else:
                    uid_val = st.number_input("User id", min_value=0, value=int(uf[id_col].iloc[0]), step=1)
                    row = uf[uf[id_col] == int(uid_val)]
                    if row.empty:
                        st.info("User not found in features.")
                    else:
                        X = row.drop(columns=[id_col])
                        # IsolationForest uses higher score -> more normal; we want anomaly score
                        # Use decision_function (higher -> less anomalous) or score_samples (more negative -> more anomalous)
                        model = ANOM["iso_model"]
                        if hasattr(model, "score_samples"):
                            s = float(model.score_samples(X)[0])
                            st.write(f"**score_samples** (more negative = more anomalous): `{s:.4f}`")
                        if hasattr(model, "decision_function"):
                            d = float(model.decision_function(X)[0])
                            st.write(f"**decision_function** (lower = more anomalous): `{d:.4f}`")
                        if ANOM["flagged_set"]:
                            st.write(f"Flagged? **{'Yes' if int(uid_val) in ANOM['flagged_set'] else 'No'}**")
            except Exception as e:
                st.warning(f"Per-user scoring unavailable: {e}")
