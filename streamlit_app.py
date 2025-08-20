
import streamlit as st
import pandas as pd
import numpy as np
import json, joblib
from pathlib import Path
from collections import Counter
from typing import Optional

HERE = Path(__file__).resolve().parent
ART = HERE / "recsys_artifacts"
assert ART.exists(), f"Artifacts folder not found at {ART}"

TASK1_MODEL_FILE = ART / "task1_ranking_best_logistic_regression.joblib"
TASK2_MODEL_FILE = ART / "task2_isolation_forest.joblib"
TASK2_OP_FILE    = ART / "task2_operating_point.json"

TASK1_FEATURES = ["cnt","share","is_last","last_gap_min","pos_from_end","tail_streak_if_last","n_views"]

@st.cache_resource
def load_task1_model():
    return joblib.load(TASK1_MODEL_FILE)

@st.cache_resource
def load_task2_model_and_op():
    iso = joblib.load(TASK2_MODEL_FILE)
    op  = json.load(open(TASK2_OP_FILE))
    return iso, op

@st.cache_data
def load_user_feats_default():
    p = ART / "task2_user_features.parquet"
    return pd.read_parquet(p) if p.exists() else None

@st.cache_data
def load_events_snapshot_default():
    for name in ["events_clean_frozen.parquet","events_clean_q04.parquet","events_clean.parquet"]:
        p = ART / name
        if p.exists():
            return pd.read_parquet(p)
    return None

def ensure_ts(ts):
    try:
        return pd.to_datetime(int(ts), unit="ms")
    except Exception:
        return pd.to_datetime(ts)

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

def rank_with_model(cand: pd.DataFrame, k: int) -> pd.DataFrame:
    if cand.empty:
        return cand
    mdl = load_task1_model()
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

def score_and_flag_users(user_feats: pd.DataFrame, share_override: Optional[float] = None) -> pd.DataFrame:
    iso, op = load_task2_model_and_op()
    feat_cols = op["feature_cols"]
    X = user_feats[feat_cols].to_numpy("float32")
    scores = -iso.decision_function(X)
    thr = float(op["threshold"]) if share_override is None else float(np.quantile(scores, 1 - float(share_override)))
    out = user_feats[["visitorid"]].copy()
    out["anom_score"] = scores
    out["is_anom"] = (scores >= thr).astype(int)
    out["threshold_used"] = thr
    return out

st.set_page_config(page_title="Recsys Inference", layout="wide")
st.title("Recommendation System Inference")

with st.sidebar:
    st.markdown("**Artifacts present**")
    st.write("Task 1 model:", TASK1_MODEL_FILE.exists())
    st.write("Task 2 model:", TASK2_MODEL_FILE.exists())
    st.write("Task 2 operating point:", TASK2_OP_FILE.exists())

tab1, tab2 = st.tabs(["Task 1 Ranking", "Task 2 Anomaly Detection"])

with tab1:
    st.subheader("Rank categories from recent views")

    left, right = st.columns([2, 1])
    with left:
        st.caption("Edit the table. Timestamp can be epoch ms or ISO string.")
        sample = pd.DataFrame({
            "timestamp": ["1690000000000","1690000300000","1690000600000","1690000900000"],
            "categoryid_final": [1613,1509,1613,1613],
        })
        df_in = st.data_editor(sample, num_rows="dynamic", use_container_width=True)
        k = st.number_input("Top k", min_value=1, max_value=10, value=3, step=1)

        if st.button("Compute ranking"):
            try:
                views = pd.DataFrame({
                    "ts": df_in["timestamp"].apply(ensure_ts),
                    "categoryid_final": df_in["categoryid_final"].astype(int)
                }).dropna()
                cand = candidates_from_views(views)
                out = rank_with_model(cand, k)
                st.success("Ranking computed")
                st.dataframe(out, use_container_width=True)
                st.download_button("Download CSV", out.to_csv(index=False).encode("utf-8"),
                                   file_name="task1_ranking.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Could not compute ranking. {e}")

    with right:
        st.caption("Quick demo using a cleaned events snapshot if present")
        ev = load_events_snapshot_default()
        if ev is None:
            st.info("No events snapshot found in recsys_artifacts")
        else:
            uids = ev["visitorid"].dropna().astype(int).unique()
            uid = st.selectbox("Pick a visitor", options=sorted(uids)[:5000])
            dfu = ev.loc[(ev["visitorid"] == uid) & (ev["event"] == "view"), ["timestamp","categoryid_final"]]
            if not dfu.empty:
                dfu = dfu.assign(
                    ts=lambda d: pd.to_datetime(d["timestamp"], unit="ms")
                    if np.issubdtype(d["timestamp"].dtype, np.number)
                    else pd.to_datetime(d["timestamp"])
                )
                out = rank_with_model(candidates_from_views(dfu[["ts","categoryid_final"]]), k=3)
                st.dataframe(out, use_container_width=True)

with tab2:
    st.subheader("Flag anomalous users")
    uploaded = st.file_uploader("Upload user features parquet or csv", type=["parquet","csv"])
    if uploaded is not None:
        user_feats = pd.read_parquet(uploaded) if uploaded.name.endswith(".parquet") else pd.read_csv(uploaded)
    else:
        user_feats = load_user_feats_default()

    if user_feats is None:
        st.info("Place task2_user_features.parquet in recsys_artifacts or upload a file")
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
                st.download_button("Download scored users CSV", scored.to_csv(index=False).encode("utf-8"),
                                   file_name="task2_scored_users.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Scoring failed. {e}")
