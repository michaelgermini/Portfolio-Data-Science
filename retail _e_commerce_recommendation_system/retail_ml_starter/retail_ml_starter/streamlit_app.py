import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from reco.recommender import MFRecommender
from reco.metrics import evaluate_topk
from reco.data_utils import generate_implicit_data, train_test_split_leave_one_out


# -------------- Utilities (Reco) --------------
def build_interaction_matrix(
    df: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "item_id",
    value_col: str | None = None,
):
    """
    Build dense implicit interaction matrix from a dataframe.
    - If value_col is provided, any positive value is treated as 1, else binary by presence.
    Returns: R (ndarray), user_to_idx, item_to_idx, idx_to_user, idx_to_item
    """
    users = df[user_col].astype(str).unique().tolist()
    items = df[item_col].astype(str).unique().tolist()

    user_to_idx = {u: i for i, u in enumerate(users)}
    item_to_idx = {i: j for j, i in enumerate(items)}
    idx_to_user = {i: u for u, i in user_to_idx.items()}
    idx_to_item = {j: i for i, j in item_to_idx.items()}

    R = np.zeros((len(users), len(items)), dtype=np.float32)
    if value_col is None:
        for _, row in df.iterrows():
            R[user_to_idx[str(row[user_col])], item_to_idx[str(row[item_col])]] = 1.0
    else:
        for _, row in df.iterrows():
            val = float(row[value_col])
            if val > 0:
                R[user_to_idx[str(row[user_col])], item_to_idx[str(row[item_col])]] = 1.0
    return R, user_to_idx, item_to_idx, idx_to_user, idx_to_item


def compute_recommendations_for_all_users(model: MFRecommender, R_train: np.ndarray, k: int = 10):
    scores = model.predict_scores(np.arange(R_train.shape[0]))
    # mask seen
    scores = np.where(R_train > 0, -np.inf, scores)
    topk_idx = np.argpartition(-scores, kth=min(k, scores.shape[1]-1), axis=1)[:, :k]
    # sort the topk per row by score desc
    sorted_topk = np.take_along_axis(topk_idx, np.argsort(-np.take_along_axis(scores, topk_idx, axis=1), axis=1), axis=1)
    sorted_scores = np.take_along_axis(scores, sorted_topk, axis=1)
    return sorted_topk, sorted_scores


# -------------- Utilities (Forecast) --------------
def compute_sarima_forecast(series: pd.Series, test_size: int, order=(2, 1, 2), seasonal_order=(1, 1, 1, 7)):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    train = series.iloc[:-test_size]
    test = series.iloc[-test_size:]
    model = SARIMAX(
        train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False)
    fc = res.forecast(steps=test_size)
    return train, test, fc


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))) * 100.0)


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# -------------- Streamlit UI --------------
st.set_page_config(page_title="Retail: Recommendation + Forecast", layout="wide")
st.title("Retail / E‑commerce — Recommendation & Forecasting")

page = st.sidebar.radio("Menu", ["Recommendation", "Forecasting"], key="menu")


# ========== Recommandation ==========
if page == "Recommendation":
    st.subheader("Product recommendation — MF (collaborative filtering)")
    st.write("Expected CSV: columns `user_id`, `item_id` and optional `value` (positive => interaction).")

    up = st.file_uploader("Uploader interactions (CSV)", type=["csv"], key="reco_csv")
    col_l, col_r = st.columns(2)
    with col_l:
        n_factors = st.slider("n_factors", min_value=8, max_value=128, value=48, step=8, key="n_factors")
        n_iters = st.slider("n_iters", min_value=1, max_value=30, value=8, step=1, key="n_iters")
        neg_ratio = st.slider("neg_ratio (negatives/positive)", min_value=1, max_value=10, value=4, step=1, key="neg_ratio")
    with col_r:
        lr = st.number_input("learning rate", min_value=0.0005, max_value=0.5, value=0.05, step=0.005, format="%.4f", key="lr")
        reg = st.number_input("regularization", min_value=0.0, max_value=0.5, value=0.01, step=0.005, format="%.4f", key="reg")
        k_top = st.slider("Top-K to recommend", min_value=5, max_value=50, value=10, step=5, key="k_top")

    if up is not None:
        df = pd.read_csv(up)
        expected_cols = set(["user_id", "item_id"]) - set(df.columns)
        if expected_cols:
            st.error("Missing required columns: " + ", ".join(sorted(expected_cols)))
            st.stop()
        value_col = "value" if "value" in df.columns else None
        R, user_to_idx, item_to_idx, idx_to_user, idx_to_item = build_interaction_matrix(df, "user_id", "item_id", value_col)
        st.caption(f"Matrix size: {R.shape[0]} users × {R.shape[1]} items")
    else:
        st.info("No CSV uploaded. Using synthetic demo data (400×600, density 1.5%).")
        R = generate_implicit_data(n_users=400, n_items=600, density=0.015, seed=7)
        user_to_idx = {str(i): i for i in range(R.shape[0])}
        item_to_idx = {str(j): j for j in range(R.shape[1])}
        idx_to_user = {i: str(i) for i in range(R.shape[0])}
        idx_to_item = {j: str(j) for j in range(R.shape[1])}

    # Safety for extremely large dense matrices
    if R.size > 5_000_000:
        st.warning("Matrix is very large for a dense NumPy model. Downsample your data or use a sparse/ALS implementation.")

    if st.button("Train & evaluate", type="primary", key="train_eval_reco"):
        with st.spinner("Training in progress..."):
            R_train, test_items = train_test_split_leave_one_out(R, seed=7)
            model = MFRecommender(
                n_factors=n_factors, lr=lr, reg=reg, n_iters=n_iters, neg_ratio=neg_ratio, seed=7
            ).fit(R_train)
            prec, ndcg = evaluate_topk(model, R_train, test_items, k=k_top)

        met1, met2 = st.columns(2)
        with met1:
            st.metric("Precision@K", f"{prec:.4f}")
        with met2:
            st.metric("NDCG@K", f"{ndcg:.4f}")

        st.divider()
        st.subheader("Recommendations by user")
        user_display = st.selectbox(
            "User",
            options=[idx_to_user[i] for i in range(R.shape[0])],
            index=0,
            key="user_select",
        )
        uid = user_to_idx[str(user_display)]
        rec_items = model.recommend(R_train, user_id=uid, k=k_top)
        rec_labels = [idx_to_item[int(j)] for j in rec_items]
        st.write(pd.DataFrame({"rank": np.arange(1, len(rec_labels) + 1), "item_id": rec_labels}))

        # Download all-user recommendations
        all_topk_idx, all_topk_scores = compute_recommendations_for_all_users(model, R_train, k=k_top)
        rows = []
        for u_idx in range(all_topk_idx.shape[0]):
            u_label = idx_to_user[u_idx]
            for r_idx in range(all_topk_idx.shape[1]):
                item_idx = int(all_topk_idx[u_idx, r_idx])
                rows.append(
                    {
                        "user_id": u_label,
                        "item_id": idx_to_item[item_idx],
                        "rank": r_idx + 1,
                        "score": float(all_topk_scores[u_idx, r_idx]),
                    }
                )
        reco_df = pd.DataFrame(rows)
        csv_buf = io.StringIO()
        reco_df.to_csv(csv_buf, index=False)
        st.download_button(
            "Download recommendations (CSV)",
            data=csv_buf.getvalue(),
            file_name="recommendations_topk.csv",
            mime="text/csv",
            key="download_reco_csv",
        )


# ========== Prévision ==========
if page == "Forecasting":
    st.subheader("Sales forecasting — SARIMA")
    st.write("Expected CSV: a date column and a value column (e.g. `date`, `sales`).")

    upf = st.file_uploader("Uploader série temporelle (CSV)", type=["csv"], key="forecast_csv")
    default_series = None
    df_ts = None

    if upf is not None:
        df_ts = pd.read_csv(upf)
        # guess columns
        date_cols_guess = [c for c in df_ts.columns if c.lower() in ("date", "ds", "time", "timestamp")] or list(df_ts.columns[:1])
        value_cols_guess = [c for c in df_ts.columns if c.lower() in ("y", "sales", "value")] or list(df_ts.columns[1:2])
        date_col = st.selectbox("Date column", options=df_ts.columns.tolist(), index=df_ts.columns.get_loc(date_cols_guess[0]), key="date_col")
        value_col = st.selectbox("Value column", options=df_ts.columns.tolist(), index=df_ts.columns.get_loc(value_cols_guess[0]), key="value_col")
        df_ts[date_col] = pd.to_datetime(df_ts[date_col])
        df_ts = df_ts[[date_col, value_col]].dropna().sort_values(date_col)
        df_ts = df_ts.rename(columns={date_col: "ds", value_col: "y"})
        series = pd.Series(df_ts["y"].values, index=pd.DatetimeIndex(df_ts["ds"]))
    else:
        st.info("No CSV uploaded. Using a synthetic series (n=400, season=7).")
        n = 400
        t = np.arange(n)
        rng = np.random.default_rng(11)
        y = 50 + 0.05 * t + 5 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 2, size=n)
        series = pd.Series(y, index=pd.date_range("2022-01-01", periods=n, freq="D"), name="sales")

    # Controls
    c1, c2, c3 = st.columns(3)
    with c1:
        test_pct = st.slider("Test size (%)", min_value=5, max_value=40, value=14, step=1, key="test_pct")
        test_size = max(1, int(len(series) * test_pct / 100))
    with c2:
        p = st.number_input("p", min_value=0, max_value=5, value=2, step=1, key="sarima_p")
        d = st.number_input("d", min_value=0, max_value=2, value=1, step=1, key="sarima_d")
        q = st.number_input("q", min_value=0, max_value=5, value=2, step=1, key="sarima_q")
    with c3:
        P = st.number_input("P", min_value=0, max_value=3, value=1, step=1, key="sarima_P")
        D = st.number_input("D", min_value=0, max_value=2, value=1, step=1, key="sarima_D")
        Q = st.number_input("Q", min_value=0, max_value=3, value=1, step=1, key="sarima_Q")
        s = st.number_input("seasonal period s", min_value=1, max_value=365, value=7, step=1, key="sarima_s")

    if st.button("Train & forecast", type="primary", key="train_forecast"):
        with st.spinner("Fitting SARIMA model..."):
            train, test, fc = compute_sarima_forecast(series, test_size, order=(p, d, q), seasonal_order=(P, D, Q, s))
        # Metrics
        met1, met2 = st.columns(2)
        with met1:
            st.metric("RMSE", f"{rmse(test.values, fc.values):.3f}")
        with met2:
            st.metric("MAPE", f"{mape(test.values, fc.values):.2f}%")

        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.plot(train.index, train.values, label="train")
        ax.plot(test.index, test.values, label="test")
        ax.plot(test.index, fc.values, label="forecast")
        ax.set_title("SARIMA forecast vs actuals")
        ax.legend()
        st.pyplot(fig, clear_figure=True)

        # Download forecast
        out_df = pd.DataFrame({"ds": test.index, "y_true": test.values, "y_hat": fc.values})
        csv_buf = io.StringIO()
        out_df.to_csv(csv_buf, index=False)
        st.download_button(
            "Download forecast (CSV)",
            data=csv_buf.getvalue(),
            file_name="forecast_sarima.csv",
            mime="text/csv",
            key="download_forecast_csv",
        )


