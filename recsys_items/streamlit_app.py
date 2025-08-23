# streamlit_app.py
import json
import numpy as np
import pandas as pd
import joblib
import streamlit as st
from collections import Counter
import glob

# ------------------ cached artefact loaders ------------------
@st.cache_data
def load_embeddings():
    import glob, os
    import pandas as pd
    # item factors (as before)
    V_path = "svd_item_factors.parquet"
    if not os.path.exists(V_path):
        V_path = "recsys_items/svd_item_factors.parquet"
    V = pd.read_parquet(V_path)
    V["candidate_cat"] = V["candidate_cat"].astype("int64")

    # user factors from shards
    shard_paths = sorted(glob.glob("recsys_items/svd_user_factors_part*.parquet"))
    U = pd.concat([pd.read_parquet(p) for p in shard_paths], ignore_index=True)
    U["visitorid"] = U["visitorid"].astype("int64")
    svd_u_cols = [c for c in U.columns if c.startswith("svd_u_")]
    U[svd_u_cols] = U[svd_u_cols].astype("float32")   # <<< upcast for model

    return U, V


@st.cache_data
def load_events_from_csv(file) -> pd.DataFrame:
    if hasattr(file, "read"):
        return pd.read_csv(file)
    return pd.read_csv(file)

@st.cache_data
def try_load_events(uploaded, default_path: str) -> pd.DataFrame:
    # priority: uploaded CSV
    if uploaded is not None:
        return load_events_from_csv(uploaded)

    # explicit path csv or parquet
    try:
        if default_path.lower().endswith(".parquet"):
            return pd.read_parquet(default_path)
        return load_events_from_csv(default_path)
    except Exception:
        pass

    # fallback shards
    shards = sorted(glob.glob("events_enriched_part*.parquet"))
    if shards:
        frames = [pd.read_parquet(p) for p in shards]
        return pd.concat(frames, ignore_index=True)

    return pd.DataFrame()

# ------------------ utilities: active user + debug ------------------
def suggest_active_user(events: pd.DataFrame, min_views: int = 10):
    if events.empty or {"event","visitorid"}.difference(events.columns):
        return None
    v = events[events["event"] == "view"]
    if v.empty:
        return None
    counts = v.groupby("visitorid").size().sort_values(ascending=False)
    if counts.empty:
        return None
    eligible = counts[counts >= min_views]
    return int((eligible.index[0] if not eligible.empty else counts.index[0]))

@st.cache_data
def get_active_users(events: pd.DataFrame, limit: int = 500) -> pd.DataFrame:
    if events.empty or {"event","visitorid"}.difference(events.columns):
        return pd.DataFrame(columns=["visitorid","n_views"])
    v = events[events["event"] == "view"]
    counts = (v.groupby("visitorid").size()
                .sort_values(ascending=False)
                .head(limit)
                .rename("n_views")
                .reset_index())
    counts["visitorid"] = counts["visitorid"].astype("int64")
    return counts

# ✅ Fixed here
def user_debug(events: pd.DataFrame, uid: int, time_window_sec: int, views_lookback: int) -> dict:
    out = {"has_rows": False, "n_views_total": 0, "n_views_window": 0}
    cols = ["timestamp","visitorid","event","categoryid_final"]
    if not set(cols).issubset(events.columns):
        return out
    df0 = events.loc[events["visitorid"] == uid, cols].dropna().copy()
    if df0.empty:
        return out
    out["has_rows"] = True
    if np.issubdtype(df0["timestamp"].dtype, np.datetime64):
        df0["ts"] = pd.to_datetime(df0["timestamp"])
    else:
        df0["ts"] = pd.to_datetime(pd.to_numeric(df0["timestamp"], errors="coerce"), unit="ms")

    v = df0[df0["event"] == "view"][["ts","categoryid_final"]].copy()
    out["n_views_total"] = int(len(v))
    if v.empty:
        return out
    t_end = v["ts"].max()
    v2 = v[(v["ts"] >= t_end - pd.Timedelta(seconds=time_window_sec)) & (v["ts"] <= t_end)].tail(views_lookback)
    out["n_views_window"] = int(len(v2))
    out["t_last"] = str(t_end)
    return out

# ------------------ feature engineering ------------------
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

# ------------------ scoring ------------------
def score_candidates(candidates: pd.DataFrame, model, meta, U: pd.DataFrame, V: pd.DataFrame, topk: int = 5) -> pd.DataFrame:
    svd_u_cols    = meta["svd_user_cols"]
    svd_i_cols    = meta["svd_item_cols"]
    hyb_features  = meta["hybrid_features"]

    candidates["visitorid"] = candidates["visitorid"].astype("int64")
    candidates["candidate_cat"] = candidates["candidate_cat"].astype("int64")

    X = (candidates.merge(U, on="visitorid", how="left")
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

model, META = load_model_and_meta()
U, V = load_embeddings()

st.sidebar.header("Data source")
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
default_path = st.sidebar.text_input("or repo file path", value="sample_events.parquet")

events_enriched = try_load_events(uploaded, default_path)

if events_enriched.empty:
    st.info("Provide data with columns: timestamp, visitorid, event, categoryid_final. "
            "You can upload a CSV or keep sample_events.parquet or include shards named events_enriched_part*.parquet.")
else:
    st.caption(f"Loaded {len(events_enriched):,} rows and {events_enriched['visitorid'].nunique():,} users")

    active_df = get_active_users(events_enriched, limit=500)
    pick_from_list = st.checkbox("Pick from top active users", value=True)

    if pick_from_list and not active_df.empty:
        user_id = st.selectbox(
            "User",
            options=active_df["visitorid"].tolist(),
            format_func=lambda u: f"{u}  views={int(active_df.loc[active_df['visitorid']==u,'n_views'].iloc[0])}"
        )
    else:
        try:
            default_uid = int(events_enriched["visitorid"].iloc[0])
        except Exception:
            default_uid = 0
        user_id = st.number_input("User id", min_value=0, value=default_uid, step=1)

    user_id = int(user_id)
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

    if st.button("Recommend"):
        cand = build_recent_candidates(
            events_enriched, user_id,
            time_window_sec=time_window_hours*3600,
            views_lookback=views_lookback
        )
        if cand.empty:
            st.warning("No candidates for this user in the selected window. Try a listed active user or increase Lookback.")
        else:
            recs = score_candidates(cand, model, META, U, V, topk=topk)
            st.subheader("Top K Categories")
            st.dataframe(recs, use_container_width=True)

