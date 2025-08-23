# streamlit_app.py
import json, glob, pathlib
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import streamlit as st
from collections import Counter

# ====== Paths (robust to where the app is launched from) ======
BASE_DIR = Path(__file__).parent                 # recsys_items/
REPO_ROOT = Path.cwd()                           # repo root when Streamlit Cloud runs
ART_DIR = (REPO_ROOT / "recsys_artifacts") if (REPO_ROOT / "recsys_artifacts").exists() else REPO_ROOT

# Task 1 artefacts (live in recsys_items/)
MODEL_PATH = BASE_DIR / "best_candidate_ranker.pkl"
META_PATH  = BASE_DIR / "inference_metadata.json"
V_PATH     = BASE_DIR / "svd_item_factors.parquet"
U_WHOLE    = BASE_DIR / "svd_user_factors.parquet"   # optional single file
U_SHARDS   = sorted(BASE_DIR.glob("svd_user_factors_part*.parquet"))

# Sample events (small)
SAMPLE_EVENTS = BASE_DIR / "sample_events.parquet"

# Task 2 artefacts (you pushed these)
FLAG_FILE      = ART_DIR / "task2_flagged_users.csv"          # columns: visitorid, is_flagged
SCORED_FILE    = ART_DIR / "task2_user_features.parquet"      # has anom_score (optional; for debug)
OPER_POINT     = ART_DIR / "task2_operating_point.json"       # optional config
EVENTS_CLEAN   = ART_DIR / "events_clean.parquet"             # optional large file

# ====== Cache: load Task 1 artefacts ======
@st.cache_resource
def load_model_and_meta():
    model = joblib.load(MODEL_PATH)
    with open(META_PATH, "r") as f:
        meta = json.load(f)
    return model, meta

@st.cache_data
def load_embeddings():
    # item factors
    V = pd.read_parquet(V_PATH)
    V["candidate_cat"] = V["candidate_cat"].astype("int64")

    # user factors from shards (preferred), else single file
    if U_SHARDS:
        parts = [pd.read_parquet(p) for p in U_SHARDS]
        U = pd.concat(parts, ignore_index=True)
    else:
        U = pd.read_parquet(U_WHOLE)

    U["visitorid"] = U["visitorid"].astype("int64")
    return U, V

# ====== Cache: data loaders ======
@st.cache_data
def load_events_from_csv(file) -> pd.DataFrame:
    if hasattr(file, "read"):
        return pd.read_csv(file)
    return pd.read_csv(file)

@st.cache_data
def try_load_events(uploaded, default_path_str: str) -> pd.DataFrame:
    # 1) uploaded CSV takes precedence
    if uploaded is not None:
        return load_events_from_csv(uploaded)

    # 2) explicit path (relative to repo root)
    try:
        p = Path(default_path_str)
        p = p if p.is_absolute() else (REPO_ROOT / p)
        if p.suffix.lower() == ".parquet":
            return pd.read_parquet(p)
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()

@st.cache_data
def load_clean_events() -> pd.DataFrame:
    if EVENTS_CLEAN.exists():
        return pd.read_parquet(EVENTS_CLEAN)
    return pd.DataFrame()

# ====== Cache: Task 2 tables ======
@st.cache_data
def load_anomaly_tables():
    flagged = pd.DataFrame(columns=["visitorid", "is_flagged"])
    scored  = pd.DataFrame()
    cfg     = {"block_flagged": False}

    if FLAG_FILE.exists():
        flagged = pd.read_csv(FLAG_FILE)
        # make robust to column names
        if "visitorid" in flagged.columns:
            flagged["visitorid"] = flagged["visitorid"].astype("int64")
        flag_col = None
        for c in flagged.columns:
            if c.lower() in {"flag", "is_flagged", "proxy_anom", "anomalous"}:
                flag_col = c; break
        if flag_col and flag_col != "is_flagged":
            flagged = flagged.rename(columns={flag_col: "is_flagged"})
        if "is_flagged" not in flagged.columns:
            flagged["is_flagged"] = 1  # assume all listed are flagged

    if SCORED_FILE.exists():
        scored = pd.read_parquet(SCORED_FILE)
        if "visitorid" in scored.columns:
            scored["visitorid"] = scored["visitorid"].astype("int64")

    if OPER_POINT.exists():
        try:
            with open(OPER_POINT, "r") as f:
                cfg.update(json.load(f))
        except Exception:
            pass

    return flagged, scored, cfg, EVENTS_CLEAN.exists()

# ====== Utilities: users ======
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

# ====== Feature engineering (same logic as training) ======
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

    # parse timestamps robustly
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

# ====== Scoring ======
def score_candidates(candidates: pd.DataFrame, model, meta, U: pd.DataFrame, V: pd.DataFrame, topk: int = 5) -> pd.DataFrame:
    svd_u_cols = meta["svd_user_cols"]
    svd_i_cols = meta["svd_item_cols"]
    hyb_features = meta["hybrid_features"]

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
flagged_df, scored_df, op_cfg, has_clean_events = load_anomaly_tables()

st.sidebar.header("Data source")
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
default_path = st.sidebar.text_input(
    "or repo file path",
    value=str(SAMPLE_EVENTS.relative_to(REPO_ROOT)) if SAMPLE_EVENTS.exists() else "recsys_items/sample_events.parquet"
)

use_clean = False
if has_clean_events and uploaded is None:
    use_clean = st.sidebar.toggle(
        "Use cleaned events (Task 2)",
        value=True,
        help="Reads recsys_artifacts/events_clean.parquet if present."
    )

# choose source
events_enriched = load_clean_events() if use_clean else try_load_events(uploaded, default_path)

if events_enriched.empty:
    st.info("Provide data with columns: timestamp, visitorid, event, categoryid_final. "
            "Use the uploader or keep a repo file like recsys_items/sample_events.parquet.")
    st.stop()

# Active users list
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

# anomaly status + controls
is_flagged = False
if not flagged_df.empty and "visitorid" in flagged_df and "is_flagged" in flagged_df:
    match = flagged_df.loc[flagged_df["visitorid"] == user_id, "is_flagged"]
    is_flagged = bool(int(match.iloc[0])) if not match.empty else False

cols = st.columns(3)
with cols[0]:
    topk = st.slider("Top K", 1, 20, 5)
with cols[1]:
    time_window_hours = st.slider("Lookback hours", 1, 72, 24)
with cols[2]:
    views_lookback = st.slider("Max recent views", 10, 300, 100, step=10)

# show debug + anomaly badge
dbg = user_debug(events_enriched, user_id, time_window_hours*3600, views_lookback)
with st.expander("Debug current user"):
    st.write(dbg)

if is_flagged:
    st.warning("⚠️ This user is flagged as anomalous by Task 2.", icon="⚠️")

block_if_flagged = st.checkbox("Block recommendations for flagged users", value=bool(op_cfg.get("block_flagged", False)))

if st.button("Recommend"):
    if is_flagged and block_if_flagged:
        st.error("Recommendations are blocked for this user (anomalous).")
        st.stop()

    cand = build_recent_candidates(
        events_enriched, user_id,
        time_window_sec=time_window_hours * 3600,
        views_lookback=views_lookback
    )
    if cand.empty:
        st.warning("No candidates for this user in the selected window. Try a listed active user or increase Lookback.")
    else:
        recs = score_candidates(cand, model, META, U, V, topk=topk)
        st.subheader("Top K Categories")
        st.dataframe(recs, use_container_width=True)
