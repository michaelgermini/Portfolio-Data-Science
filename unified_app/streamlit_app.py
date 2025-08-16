import pandas as pd
import streamlit as st
import plotly.express as px
import requests
from pathlib import Path
 
# ML/Preprocessing imports for House Prices page
import numpy as np
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, precision_recall_fscore_support, accuracy_score, confusion_matrix, mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer


st.set_page_config(page_title="Unified dashboards – Inequalities & Climate", layout="wide")


def render_inegalites() -> None:
    st.header("Global inequalities – Country comparator")
    st.caption("Upload your CSV (World Bank/UN/IMF) or use a default indicator (GDP per capita). Compare two sets of countries side-by-side.")

    @st.cache_data(show_spinner=False)
    def fetch_worldbank_indicator(indicator: str) -> pd.DataFrame:
        try:
            if indicator == "NY.GDP.PCAP.CD":
                df = pd.read_csv(
                    "https://raw.githubusercontent.com/owid/owid-datasets/master/datasets/GDP%20per%20capita%20(Penn%20World%20Table)/GDP%20per%20capita%20(Penn%20World%20Table).csv"
                )
                if {"Year", "GDP per capita"}.issubset(df.columns):
                    df = df.rename(columns={"GDP per capita": "value", "Entity": "country", "Year": "year"})
                    df["indicator"] = "gdp_per_capita"
                    return df[["country", "year", "indicator", "value"]]
        except Exception:
            pass
        return pd.DataFrame()

    def pivot_long(df: pd.DataFrame) -> pd.DataFrame:
        if {"country", "year", "indicator", "value"}.issubset(df.columns):
            return df
        if "Country Name" in df.columns and any(c.isdigit() for c in df.columns):
            value_cols = [c for c in df.columns if c.isdigit()]
            tidy = df.melt(id_vars=["Country Name"], value_vars=value_cols, var_name="year", value_name="value")
            tidy = tidy.rename(columns={"Country Name": "country"})
            tidy["indicator"] = "value"
            tidy["year"] = pd.to_numeric(tidy["year"], errors="coerce").astype("Int64")
            return tidy
        return df

    def country_selector(label: str, options, key: str):
        default = ["France", "Germany"] if {"France", "Germany"}.issubset(set(options)) else options[:2]
        return st.multiselect(label, options=options, default=default, key=key)

    with st.sidebar:
        st.subheader("Data – Inequalities")
        uploaded = st.file_uploader("Upload a CSV (country, year, indicator, value)", type=["csv"], key="ineg_csv")
        with st.expander("Load from opendata.swiss (CSV URL)"):
            csv_url = st.text_input("CSV URL", key="ineg_csv_url")
            load_click = st.button("Load URL", key="ineg_load_url")
            if load_click and csv_url:
                try:
                    df_ext = pd.read_csv(csv_url)
                    st.session_state["ineg_ext_df"] = df_ext
                except Exception as e:
                    st.error(f"Failed to load CSV from URL: {e}")
            if "ineg_ext_df" in st.session_state:
                df_ext = st.session_state["ineg_ext_df"]
                st.caption("Map columns from the external CSV")
                cols = df_ext.columns.tolist()
                col_country = st.selectbox("Country column", cols, key="ineg_map_country")
                col_year = st.selectbox("Year column", cols, key="ineg_map_year")
                col_value = st.selectbox("Value column", cols, key="ineg_map_value")
                indicator_name = st.text_input("Indicator name", value="external_indicator", key="ineg_map_indicator")
                if st.button("Use this dataset", key="ineg_use_ext"):
                    try:
                        tmp = df_ext[[col_country, col_year, col_value]].copy()
                        tmp.columns = ["country", "year", "value"]
                        tmp["indicator"] = indicator_name
                        # Coerce year
                        tmp["year"] = pd.to_numeric(tmp["year"], errors="coerce").astype("Int64")
                        st.session_state["ineg_final_df"] = tmp.dropna(subset=["year"])  # store tidy
                        st.success("External dataset mapped. It will be used below.")
                    except Exception as e:
                        st.error(f"Mapping failed: {e}")
        indicator_choice = st.selectbox("Default indicator if no CSV is provided", ["GDP per capita (NY.GDP.PCAP.CD)", "None"], index=0, key="ineg_indic")

    @st.cache_data(show_spinner=False)
    def load_local_sample() -> pd.DataFrame:
        try:
            sample_path = Path(__file__).resolve().parents[1] / "exploration_inegalites" / "data" / "inegalites_sample.csv"
            # If the inequalities app has not created it yet, try to generate a simple fallback here
            if not sample_path.exists():
                sample_path.parent.mkdir(parents=True, exist_ok=True)
                df_tmp = pd.DataFrame({
                    "country": ["France","Germany","United States","India"],
                    "year": [2000,2000,2000,2000],
                    "indicator": ["gdp_per_capita_usd"]*4,
                    "value": [28000, 25000, 36000, 450]
                })
                df_tmp.to_csv(sample_path, index=False)
            df_local = pd.read_csv(sample_path)
            return pivot_long(df_local)
        except Exception:
            return pd.DataFrame()

    df = pd.DataFrame()
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            df = pivot_long(df)
        except Exception as e:
            st.error(f"Erreur lors de la lecture du CSV: {e}")
    else:
        # 0) External mapped dataset if present
        if "ineg_final_df" in st.session_state:
            df = st.session_state["ineg_final_df"].copy()
        # 1) Else try local sample
        if df.empty:
            df = load_local_sample()
        # 2) Sinon tenter la source distante par défaut
        if df.empty and indicator_choice.startswith("GDP"):
            df = fetch_worldbank_indicator("NY.GDP.PCAP.CD")

    if df.empty:
        st.info("No data available. Upload a CSV (country, year, indicator, value) or enable internet for the default source.")
        return

    available_countries = sorted(df["country"].dropna().unique().tolist())
    available_indicators = sorted(df["indicator"].dropna().unique().tolist())

    # Query params load (deep-link)
    qp = st.query_params
    if "ineg_c_left" in qp and "ineg_c_right" in qp:
        st.session_state.setdefault("ineg_countries_left", qp.get_all("ineg_c_left"))
        st.session_state.setdefault("ineg_countries_right", qp.get_all("ineg_c_right"))
    if "ineg_indicator" in qp:
        st.session_state.setdefault("ineg_sel_ind", qp.get("ineg_indicator"))
    if "ineg_year_min" in qp and "ineg_year_max" in qp:
        try:
            st.session_state.setdefault("ineg_year", (int(qp.get("ineg_year_min")), int(qp.get("ineg_year_max"))))
        except Exception:
            pass

    with st.sidebar:
        selected_indicator = st.selectbox("Indicator", available_indicators, index=0, key="ineg_sel_ind")
        countries_left = country_selector("Countries – left panel", available_countries, key="ineg_countries_left")
        countries_right = country_selector("Countries – right panel", available_countries, key="ineg_countries_right")
        df_years = pd.to_numeric(df["year"], errors="coerce")
        year_min, year_max = int(df_years.min()), int(df_years.max())
        year_range = st.slider("Period", min_value=year_min, max_value=year_max, value=st.session_state.get("ineg_year", (year_min, year_max)), key="ineg_year")

        with st.expander("Presets"):
            presets = {
                "G7": ["Canada", "France", "Germany", "Italy", "Japan", "United Kingdom", "United States"],
                "BRICS": ["Brazil", "Russia", "India", "China", "South Africa"],
            }
            preset_name = st.selectbox("Preset", list(presets.keys()), index=0, key="ineg_preset")
            colp1, colp2 = st.columns(2)
            with colp1:
                if st.button("Apply to left"):
                    st.session_state["ineg_countries_left"] = [c for c in presets[preset_name] if c in available_countries]
            with colp2:
                if st.button("Apply to right"):
                    st.session_state["ineg_countries_right"] = [c for c in presets[preset_name] if c in available_countries]

    def filter_df(base: pd.DataFrame, countries, indicator: str, year_min: int, year_max: int) -> pd.DataFrame:
        tmp = base.copy()
        tmp = tmp[(tmp["indicator"] == indicator)]
        if countries:
            tmp = tmp[tmp["country"].isin(countries)]
        tmp["year"] = pd.to_numeric(tmp["year"], errors="coerce")
        tmp = tmp[(tmp["year"] >= year_min) & (tmp["year"] <= year_max)]
        tmp = tmp.dropna(subset=["year"]).sort_values(["country", "year"]) 
        return tmp

    # Persist to query params
    try:
        st.query_params.update({
            "ineg_indicator": selected_indicator,
            "ineg_year_min": str(year_range[0]),
            "ineg_year_max": str(year_range[1]),
        })
        # repeatable keys for arrays
        st.query_params.setlist("ineg_c_left", countries_left)
        st.query_params.setlist("ineg_c_right", countries_right)
    except Exception:
        pass

    # Tabs while keeping sidebar
    tab_ts, tab_latest, tab_rank = st.tabs(["Time series", "Latest values", "Ranking"])

    with tab_ts:
        left, right = st.columns(2)
        with left:
            st.subheader("Left panel")
            dfl = filter_df(df, countries_left, selected_indicator, year_range[0], year_range[1])
            if dfl.empty:
                st.warning("No data for selection.")
            else:
                fig = px.line(dfl, x="year", y="value", color="country", markers=True, title=selected_indicator)
                st.plotly_chart(fig, use_container_width=True)
        with right:
            st.subheader("Right panel")
            dfr = filter_df(df, countries_right, selected_indicator, year_range[0], year_range[1])
            if dfr.empty:
                st.warning("No data for selection.")
            else:
                fig = px.line(dfr, x="year", y="value", color="country", markers=True, title=selected_indicator)
                st.plotly_chart(fig, use_container_width=True)

    with tab_latest:
        df_period = filter_df(df, available_countries, selected_indicator, year_range[0], year_range[1])
        if df_period.empty:
            st.warning("No data for selection.")
        else:
            latest = df_period.sort_values("year").groupby("country").tail(1)
            st.dataframe(latest[["country", "year", "value"]].rename(columns={"value": "last_year_value"}), use_container_width=True)

    with tab_rank:
        df_all = df[df["indicator"] == selected_indicator].copy()
        if df_all.empty:
            st.warning("No data for selection.")
        else:
            rank_year = st.select_slider("Ranking year", options=sorted(pd.to_numeric(df_all["year"], errors="coerce").dropna().unique().astype(int).tolist()), value=int(pd.to_numeric(df_all["year"], errors="coerce").max()))
            scope = st.radio("Scope", ["All countries", "Selected (left)", "Selected (right)"], horizontal=True)
            if scope == "Selected (left)":
                df_all = df_all[df_all["country"].isin(countries_left)]
            elif scope == "Selected (right)":
                df_all = df_all[df_all["country"].isin(countries_right)]
            df_rank = df_all[pd.to_numeric(df_all["year"], errors="coerce") == rank_year]
            df_rank = df_rank.dropna(subset=["value"]).sort_values("value", ascending=False).head(25)
            if df_rank.empty:
                st.warning("No data for selected year.")
            else:
                fig = px.bar(df_rank, x="value", y="country", orientation="h", title=f"Ranking – {selected_indicator} in {rank_year}")
                st.plotly_chart(fig, use_container_width=True)

    st.caption("Tip: Import multiple indicators by concatenating your sources (UN/World Bank/IMF) in tidy long format: country, year, indicator, value.")


def render_climat() -> None:
    st.header("CO₂ & Energy evolution – Interactive dashboard")
    st.caption("Explore the evolution of CO₂ emissions and energy consumption since 1960. 3D timeline animated by decade.")

    @st.cache_data(show_spinner=False)
    def load_owid_co2() -> pd.DataFrame:
        try:
            url = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"
            return pd.read_csv(url)
        except Exception:
            return pd.DataFrame()

    def decade(year: int) -> str:
        try:
            d = int(year) // 10 * 10
            return f"{d}s"
        except Exception:
            return "NA"

    with st.sidebar:
        st.subheader("Data – Climate & Energy")
        country = st.text_input("Country filter (code or partial name)", value="France", key="cl_country")
        z_axis = st.selectbox("Z axis (height)", ["co2", "energy_per_capita", "primary_energy_consumption"], index=0, key="cl_z")

    df = load_owid_co2()
    if df.empty:
        st.info("Impossible de charger les données OWID. Vérifiez votre connexion ou importez un CSV compatible (à implémenter).")
        return

    df = df[(df["year"] >= 1960)]
    if country:
        mask = df["country"].str.contains(country, case=False, na=False) | df["iso_code"].str.contains(country, case=False, na=False)
        df = df[mask]
    if df.empty:
        st.warning("Aucun enregistrement après filtrage.")
        return

    df["decade"] = df["year"].apply(decade)
    if "energy_per_capita" not in df.columns:
        df["energy_per_capita"] = None
    if "primary_energy_consumption" not in df.columns:
        df["primary_energy_consumption"] = None

    left, right = st.columns([2, 1])
    with left:
        st.subheader("3D timeline (decade scrub)")
        fig3d = px.scatter_3d(
            df,
            x="year",
            y="co2",
            z=z_axis,
            color="country",
            animation_frame="decade",
            hover_name="country",
            size_max=18,
            opacity=0.7,
            title=f"CO₂ vs {z_axis} – animé par décennie",
        )
        fig3d.update_layout(scene=dict(xaxis_title="Year", yaxis_title="CO₂ (Mt)", zaxis_title=z_axis))
        st.plotly_chart(fig3d, use_container_width=True)

    with right:
        st.subheader("Time series")
        metric = st.selectbox("Variable", ["co2", z_axis], index=0, key="cl_metric")
        fig2d = px.line(df, x="year", y=metric, color="country", markers=True)
        st.plotly_chart(fig2d, use_container_width=True)
    st.caption("Source: Our World in Data (OWID).")


st.sidebar.title("Data Science Portfolio")
page = st.sidebar.radio(
    "Section",
    [
        "Inequalities",
        "Climate & Energy",
        "ML: House Prices",
        "ML: Spam Classification",
        "DL: Image Classification",
        "DL: Chatbot NLP",
        "Data for Good: AQI Forecast",
        "Data for Good: Local Climate 2050",
        "Storytelling: Pandemics Map",
        "Storytelling notes",
        "Swiss Geo (STAC)",
    ],
    index=0,
)

if page == "Inequalities":
    render_inegalites()
elif page == "Climate & Energy":
    render_climat()
elif page == "ML: House Prices":
    def render_ml_house_prices() -> None:
        st.header("ML – House Prices (regression)")
        st.caption("Kaggle House Prices – ridge/lasso/GBR with preprocessing and cross‑validation.")

        with st.expander("About this page"):
            st.markdown("- Upload Kaggle `train.csv` to generate CV metrics and visuals (predicted vs actual, residuals, feature importance).\n- Uses a robust preprocessing pipeline (impute, scale, one‑hot).\n- Models: Ridge (CV predictions) + Gradient Boosting (feature importances).")

        uploaded_train = st.file_uploader("Upload Kaggle train.csv", type=["csv"], key="hp_train")
        use_log = st.checkbox("Use log target (log1p(SalePrice))", value=True)

        # Auto-use bundled sample if no upload
        sample_path = Path(__file__).resolve().parents[1] / "ml_immo" / "data" / "house_prices" / "train.csv"
        if uploaded_train is None:
            if sample_path.exists():
                st.caption(f"Using bundled sample: `{sample_path.as_posix()}`")
                st.download_button(
                    label="Download sample CSV",
                    data=sample_path.read_bytes(),
                    file_name="train_sample.csv",
                    mime="text/csv",
                )
                uploaded_train = sample_path.open("rb")
            else:
                st.info("Upload the Kaggle training file to see metrics and charts. The bundled sample is not available.")
                st.markdown("Notebook: `ml_immo/notebooks/house_prices_modeling.ipynb`")
                return

        # Load data
        try:
            df = pd.read_csv(uploaded_train)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            return

        if "SalePrice" not in df.columns:
            st.error("Column `SalePrice` not found. Please upload the Kaggle training set.")
            return

        # Basic preprocessing schema
        y = df["SalePrice"].astype(float)
        X = df.drop(columns=["SalePrice"]).copy()
        numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

        preprocess = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]),
                    numeric_cols,
                ),
                (
                    "cat",
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(handle_unknown="ignore")),
                    ]),
                    categorical_cols,
                ),
            ]
        )

        # Cross-validated predictions with Ridge
        ridge = Pipeline([
            ("prep", preprocess),
            ("model", Ridge(alpha=10.0)),
        ])

        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        y_target = np.log1p(y) if use_log else y.values
        try:
            y_pred_cv = cross_val_predict(ridge, X, y_target, cv=cv, n_jobs=-1)
        except Exception as e:
            st.error(f"Model failed during CV: {e}")
            return

        # Back-transform for metrics/plots if log used
        if use_log:
            y_pred_eval = np.expm1(y_pred_cv)
            y_true_eval = y.values
        else:
            y_pred_eval = y_pred_cv
            y_true_eval = y.values

        rmse = float(np.sqrt(mean_squared_error(y_true_eval, y_pred_eval)))
        r2 = float(r2_score(y_true_eval, y_pred_eval))

        m1, m2 = st.columns(2)
        m1.metric("RMSE (CV)", f"{rmse:,.0f}")
        m2.metric("R² (CV)", f"{r2:.3f}")

        # Predicted vs Actual scatter
        df_scatter = pd.DataFrame({"Actual": y_true_eval, "Predicted": y_pred_eval})
        fig_scatter = px.scatter(df_scatter, x="Actual", y="Predicted", title="Predicted vs Actual (Ridge, 5-fold CV)", opacity=0.6)
        # Add y=x reference line
        fig_scatter.add_shape(type="line", x0=df_scatter["Actual"].min(), y0=df_scatter["Actual"].min(),
                              x1=df_scatter["Actual"].max(), y1=df_scatter["Actual"].max(), line=dict(color="red", dash="dash"))
        st.plotly_chart(fig_scatter, use_container_width=True)

        # Residuals
        residuals = y_pred_eval - y_true_eval
        fig_resid = px.histogram(pd.DataFrame({"Residual": residuals}), x="Residual", nbins=40, title="Residuals (Predicted - Actual)")
        st.plotly_chart(fig_resid, use_container_width=True)

        # Feature importance via Gradient Boosting on transformed features
        try:
            preprocess.fit(X)
            X_tx = preprocess.transform(X)
            feature_names = []
            try:
                feature_names = preprocess.get_feature_names_out().tolist()
            except Exception:
                feature_names = [f"f{i}" for i in range(X_tx.shape[1])]

            gbr = GradientBoostingRegressor(random_state=42)
            y_fit = np.log1p(y) if use_log else y.values
            gbr.fit(X_tx, y_fit)
            importances = getattr(gbr, "feature_importances_", None)
            if importances is not None:
                imp_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False).head(20)
                fig_imp = px.bar(imp_df, x="importance", y="feature", orientation="h", title="Top features (Gradient Boosting importance)")
                st.plotly_chart(fig_imp, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not compute feature importances: {e}")

        # Links for deeper work
        st.markdown("Notebook path: `ml_immo/notebooks/house_prices_modeling.ipynb`")
        st.markdown("Data: place Kaggle `train.csv` and `test.csv` under `ml_immo/data/house_prices/`.")
        st.markdown("Metrics: RMSE, R² via 5‑fold cross‑validation.")

        with st.expander("How to interpret the charts"):
            st.markdown("- Predicted vs Actual: points close to the dashed line indicate good fit; systematic deviations suggest bias.\n- Residuals: centered, narrow distribution means low error; skew or heavy tails suggest missing interactions or outliers.\n- Feature importance: higher bars indicate stronger predictive power (in the transformed feature space). Use this to guide feature engineering.")

    render_ml_house_prices()
elif page == "ML: Spam Classification":
    def render_ml_spam() -> None:
        st.header("ML – Spam vs Ham (text classification)")
        st.caption("TF‑IDF + Logistic Regression / Naive Bayes; reports Precision/Recall/F1.")
        st.markdown("Notebook path: `spam_classification/notebooks/spam_classification.ipynb`")

        uploaded = st.file_uploader("Upload spam.csv (columns: text, label)", type=["csv"], key="spam_csv")
        sample_path = Path(__file__).resolve().parents[1] / "spam_classification" / "data" / "spam.csv"
        if uploaded is None:
            if sample_path.exists():
                st.caption(f"Using bundled sample: `{sample_path.as_posix()}`")
                st.download_button(
                    label="Download sample CSV",
                    data=sample_path.read_bytes(),
                    file_name="spam_sample.csv",
                    mime="text/csv",
                )
                uploaded = sample_path.open("rb")
            else:
                st.info("Upload a CSV with `text,label` to run the classification.")
                return

        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            return

        # Basic validation
        expected_cols = {"text", "label"}
        if not expected_cols.issubset(set(c.lower() for c in df.columns)):
            st.error("CSV must contain columns `text` and `label` (case-insensitive).")
            return
        # Normalize column names
        df.columns = [c.lower() for c in df.columns]
        df = df.dropna(subset=["text", "label"]).copy()
        df["label"] = df["label"].astype(str).str.strip().str.lower()

        # Controls
        colc1, colc2, colc3 = st.columns(3)
        with colc1:
            model_choices = st.multiselect("Models", ["Logistic Regression", "Naive Bayes"], default=["Logistic Regression", "Naive Bayes"]) 
        with colc2:
            test_size = st.slider("Test size", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
        with colc3:
            use_bigrams = st.checkbox("Use bigrams (1-2)", value=True)

        X = df["text"].astype(str).values
        y = df["label"].values
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
        except Exception:
            # Fallback without stratify if few samples
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        ngram_range = (1, 2) if use_bigrams else (1, 1)
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=ngram_range)

        results = []
        trained = {}
        for name in model_choices:
            if name == "Logistic Regression":
                clf = LogisticRegression(max_iter=500)
            elif name == "Naive Bayes":
                clf = MultinomialNB()
            else:
                continue
            pipe = Pipeline([
                ("tfidf", vectorizer),
                ("clf", clf),
            ])
            try:
                pipe.fit(X_train, y_train)
                y_pred = pipe.predict(X_test)
                acc = float(accuracy_score(y_test, y_pred))
                prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)
                results.append((name, acc, float(prec), float(rec), float(f1)))
                trained[name] = pipe
            except Exception as e:
                st.warning(f"{name} failed to train: {e}")

        if not results:
            st.error("No model could be trained. Check your dataset.")
            return

        res_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision (weighted)", "Recall (weighted)", "F1 (weighted)"]).sort_values("F1 (weighted)", ascending=False)
        st.dataframe(res_df, use_container_width=True)

        # Select model for diagnostics
        best_model = res_df.iloc[0, 0]
        model_to_show = st.selectbox("Diagnostics for model", res_df["Model"].tolist(), index=0)
        pipe = trained.get(model_to_show, trained.get(best_model))

        # Confusion matrix
        y_pred_show = pipe.predict(X_test)
        labels_sorted = sorted(pd.unique(np.concatenate([y_test, y_pred_show])))
        cm = confusion_matrix(y_test, y_pred_show, labels=labels_sorted)
        fig_cm = px.imshow(cm, text_auto=True, x=labels_sorted, y=labels_sorted, title=f"Confusion matrix – {model_to_show}")
        fig_cm.update_xaxes(title_text="Predicted")
        fig_cm.update_yaxes(title_text="Actual")
        st.plotly_chart(fig_cm, use_container_width=True)

        # Top features for Logistic Regression
        if model_to_show == "Logistic Regression":
            try:
                lr = pipe.named_steps["clf"]
                tfidf = pipe.named_steps["tfidf"]
                feature_names = tfidf.get_feature_names_out()
                # For binary, coef shape (1, n_features)
                coefs = lr.coef_[0] if lr.coef_.ndim == 2 else lr.coef_
                top_idx = np.argsort(coefs)[-20:][::-1]
                bot_idx = np.argsort(coefs)[:20]
                top_df = pd.DataFrame({"feature": feature_names[top_idx], "weight": coefs[top_idx]})
                bot_df = pd.DataFrame({"feature": feature_names[bot_idx], "weight": coefs[bot_idx]})
                colf1, colf2 = st.columns(2)
                with colf1:
                    st.subheader("Top tokens (spam leaning)")
                    st.plotly_chart(px.bar(top_df, x="weight", y="feature", orientation="h"), use_container_width=True)
                with colf2:
                    st.subheader("Top tokens (ham leaning)")
                    st.plotly_chart(px.bar(bot_df, x="weight", y="feature", orientation="h"), use_container_width=True)
            except Exception:
                st.info("Feature weights not available.")

        st.subheader("Try it")
        user_text = st.text_input("Type a message to classify", value="Win a free ticket now! Reply YES to claim.")
        if user_text:
            try:
                pred = pipe.predict([user_text])[0]
                if hasattr(pipe.named_steps["clf"], "predict_proba"):
                    proba = pipe.predict_proba([user_text])[0]
                    proba_map = {cls: float(p) for cls, p in zip(pipe.classes_, proba)}
                    st.write({"prediction": pred, "proba": proba_map})
                else:
                    st.write({"prediction": pred})
            except Exception as e:
                st.warning(f"Could not classify input: {e}")

    render_ml_spam()
elif page == "Data for Good: AQI Forecast":
    def render_dfg_aqi() -> None:
        st.header("Data for Good – AQI Forecast")
        st.caption("Next‑day city AQI using time‑series baselines; optional LSTM.")
        st.markdown("Notebook path: `data_for_good/air_quality_forecast/notebooks/aqi_forecast.ipynb`")

        data_path = Path("data_for_good/air_quality_forecast/data/aqi_city.csv")
        uploaded = st.file_uploader("Upload AQI CSV (date, aqi, temp, wind, humidity, ...)", type=["csv"], key="aqi_csv")

        if uploaded is None:
            if data_path.exists():
                st.caption(f"Using sample: `{data_path.as_posix()}`")
                df = pd.read_csv(data_path)
                st.download_button("Download sample CSV", data=data_path.read_bytes(), file_name="aqi_city.csv", mime="text/csv")
            else:
                st.info("No data provided. Generate with: `python scripts/generate_aqi_sample.py`.")
                return
        else:
            try:
                df = pd.read_csv(uploaded)
            except Exception as e:
                st.error(f"Could not read CSV: {e}")
                return

        needed_cols = {"date", "aqi"}
        if not needed_cols.issubset(set(c.lower() for c in df.columns)):
            st.error("CSV must include at least `date` and `aqi`. Optional: `temp, wind, humidity, ...`.")
            return
        df.columns = [c.lower() for c in df.columns]
        df = df.dropna(subset=["date", "aqi"]).copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")

        # Features
        df["aqi_lag1"] = df["aqi"].shift(1)
        df["aqi_lag7"] = df["aqi"].shift(7)
        df["dow"] = df["date"].dt.dayofweek
        for col in ["temp", "wind", "humidity"]:
            if col in df.columns:
                df[f"{col}_lag1"] = df[col].shift(1)

        df = df.dropna().reset_index(drop=True)
        if df.empty:
            st.warning("Not enough history after feature engineering.")
            return

        target = "aqi"
        feature_cols = [c for c in df.columns if c not in ["date", target]]

        # Train/validation split (last N as validation)
        val_ratio = st.slider("Validation ratio", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
        n = len(df)
        n_val = max(1, int(n * val_ratio))
        train_df = df.iloc[:-n_val]
        val_df = df.iloc[-n_val:]

        baseline_pred = val_df["aqi_lag1"].values
        rf = RandomForestRegressor(n_estimators=300, random_state=42)
        rf.fit(train_df[feature_cols], train_df[target])
        rf_pred = rf.predict(val_df[feature_cols])

        # Metrics
        b_rmse = float(np.sqrt(mean_squared_error(val_df[target], baseline_pred)))
        b_mae = float(mean_absolute_error(val_df[target], baseline_pred))
        m_rmse = float(np.sqrt(mean_squared_error(val_df[target], rf_pred)))
        m_mae = float(mean_absolute_error(val_df[target], rf_pred))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Baseline RMSE (lag-1)", f"{b_rmse:.1f}")
        c2.metric("Baseline MAE (lag-1)", f"{b_mae:.1f}")
        c3.metric("Model RMSE (RF)", f"{m_rmse:.1f}")
        c4.metric("Model MAE (RF)", f"{m_mae:.1f}")

        # Plot predictions
        plot_df = val_df[["date", target]].copy()
        plot_df["baseline"] = baseline_pred
        plot_df["rf_pred"] = rf_pred
        fig = px.line(plot_df, x="date", y=[target, "baseline", "rf_pred"], title="Next‑day AQI forecast – validation window")
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Notes"):
            st.markdown("- Baseline uses yesterday's AQI as today's forecast (persistence).\n- Random Forest uses lags and optional weather inputs.\n- For LSTM, see the notebook.")
    render_dfg_aqi()
elif page == "Data for Good: Local Climate 2050":
    def render_dfg_climate() -> None:
        st.header("Data for Good – Local climate projection (2050)")
        st.caption("Bias correction from local series + scenario exploration (RCP/SSP).")
        st.markdown("Notebook path: `data_for_good/local_climate_projection/notebooks/local_climate_projection.ipynb`")

        data_path = Path("data_for_good/local_climate_projection/data/local_temp.csv")
        uploaded = st.file_uploader("Upload local temperature CSV (date, tmean)", type=["csv"], key="loc_temp_csv")

        if uploaded is None:
            if data_path.exists():
                st.caption(f"Using sample: `{data_path.as_posix()}`")
                df = pd.read_csv(data_path)
                st.download_button("Download sample CSV", data=data_path.read_bytes(), file_name="local_temp.csv", mime="text/csv")
            else:
                st.info("No data provided. Generate with: `python scripts/generate_local_temp_sample.py`.")
                return
        else:
            try:
                df = pd.read_csv(uploaded)
            except Exception as e:
                st.error(f"Could not read CSV: {e}")
                return

        if not {"date", "tmean"}.issubset(set(c.lower() for c in df.columns)):
            st.error("CSV must include `date` and `tmean`.")
            return
        df.columns = [c.lower() for c in df.columns]
        df = df.dropna(subset=["date", "tmean"]).copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")

        # Seasonal climatology
        df["doy"] = df["date"].dt.dayofyear
        clim = df.groupby("doy")["tmean"].mean()

        # Scenario selection
        scenario = st.selectbox("Scenario (2050 delta)", [
            ("SSP1-2.6 (+1.0°C)", 1.0),
            ("SSP2-4.5 (+1.5°C)", 1.5),
            ("SSP3-7.0 (+2.5°C)", 2.5),
            ("SSP5-8.5 (+3.0°C)", 3.0),
        ], format_func=lambda x: x[0], index=1)
        scen_label, scen_delta = scenario

        # Project to 2050 with linear ramp from last observed date
        last_date = df["date"].max()
        end_proj = pd.Timestamp(year=2050, month=12, day=31)
        proj_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), end=end_proj, freq="D")
        if len(proj_dates) == 0:
            st.info("Projection window already past 2050.")
            return
        ramp = np.linspace(0.0, float(scen_delta), len(proj_dates))
        proj_doy = proj_dates.dayofyear
        proj_base = np.array([clim.get(d, clim.mean()) for d in proj_doy])
        proj_tmean = proj_base + ramp

        hist_df = df[["date", "tmean"]].copy()
        proj_df = pd.DataFrame({"date": proj_dates, "tmean": proj_tmean})
        plot_df = pd.concat([
            hist_df.assign(series="historical"),
            proj_df.assign(series=f"projection {scen_label}"),
        ])
        fig = px.line(plot_df, x="date", y="tmean", color="series", title="Local climate projection to 2050 (bias-corrected seasonal + scenario ramp)")
        st.plotly_chart(fig, use_container_width=True)

        csv_bytes = proj_df.to_csv(index=False).encode()
        st.download_button("Download projected series (CSV)", data=csv_bytes, file_name="local_temp_projection_2050.csv", mime="text/csv")
    render_dfg_climate()
elif page == "DL: Image Classification":
    def render_dl_image() -> None:
        st.header("Deep Learning – Image Classification")
        st.caption("CNNs/transfer learning (e.g., ResNet/EfficientNet) on a small image dataset.")

        data_root = Path("deep_learning/image_classification/data")
        data_root.mkdir(parents=True, exist_ok=True)

        # Lazy imports to avoid crashing if torch isn't installed
        try:
            import torch  # type: ignore
            import torch.nn as nn  # type: ignore
            from torch.utils.data import DataLoader, random_split  # type: ignore
            import torchvision  # type: ignore
            from torchvision import datasets, transforms  # type: ignore
            from torchvision.models import resnet18, efficientnet_b0  # type: ignore
            from torchvision.models import ResNet18_Weights, EfficientNet_B0_Weights  # type: ignore
            TORCH_OK = True
        except Exception as e:
            TORCH_OK = False
            st.warning("PyTorch/torchvision not available. Install dependencies to enable training: `pip install -r requirements.txt`.")
            st.stop()

        def list_classes(root: Path):
            if not root.exists():
                return []
            return [p.name for p in root.iterdir() if p.is_dir()]

        def generate_synthetic_dataset(root: Path, num_per_class: int = 40, size: int = 128) -> None:
            from PIL import Image, ImageDraw  # pillow is available
            import random
            classes = {
                "red": (220, 40, 40),
                "green": (40, 180, 90),
                "blue": (70, 120, 230),
            }
            for cls, color in classes.items():
                cls_dir = root / cls
                cls_dir.mkdir(parents=True, exist_ok=True)
                for i in range(num_per_class):
                    img = Image.new("RGB", (size, size), (255, 255, 255))
                    draw = ImageDraw.Draw(img)
                    # random rectangle
                    x0 = random.randint(0, size // 3)
                    y0 = random.randint(0, size // 3)
                    x1 = random.randint(size // 2, size)
                    y1 = random.randint(size // 2, size)
                    draw.rectangle([x0, y0, x1, y1], fill=color)
                    img.save(cls_dir / f"img_{i:03d}.png")

        with st.sidebar:
            st.subheader("Data – Image Classification")
            st.markdown("Root: `deep_learning/image_classification/data/`")
            if st.button("Generate tiny synthetic dataset (3 classes)"):
                generate_synthetic_dataset(data_root)
                st.success("Synthetic dataset generated (classes: red, green, blue).")

            subdirs = list_classes(data_root)
            st.caption(f"Detected classes: {', '.join(subdirs) if subdirs else 'none'}")

        if not list_classes(data_root):
            # Auto-generate a tiny synthetic dataset and cache it on disk for future runs
            generate_synthetic_dataset(data_root)
            st.caption("No dataset found – generated a tiny synthetic dataset (classes: red, green, blue) and cached it under `deep_learning/image_classification/data/`.")

        # Controls
        colm1, colm2, colm3, colm4 = st.columns(4)
        with colm1:
            model_name = st.selectbox("Model", ["ResNet18", "EfficientNet-B0"], index=0)
        with colm2:
            epochs = st.number_input("Epochs", min_value=1, max_value=10, value=2, step=1)
        with colm3:
            batch_size = st.selectbox("Batch size", [8, 16, 32], index=1)
        with colm4:
            lr = st.selectbox("Learning rate", [1e-3, 5e-4, 1e-4], index=0)
        freeze_backbone = st.checkbox("Freeze backbone (feature extractor)", value=True)
        val_split = st.slider("Validation split", min_value=0.1, max_value=0.5, value=0.2, step=0.05)

        # Data pipeline
        img_size = 224
        tf_train = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        tf_val = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        full_ds = datasets.ImageFolder(str(data_root), transform=tf_train)
        num_classes = len(full_ds.classes)
        n_total = len(full_ds)
        n_val = max(1, int(n_total * val_split))
        n_train = n_total - n_val
        train_ds, val_ds = random_split(full_ds, [n_train, n_val])
        # Set validation transform
        val_ds.dataset.transform = tf_val

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        # Model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model_name == "ResNet18":
            weights = ResNet18_Weights.DEFAULT
            model = resnet18(weights=weights)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
        else:
            weights = EfficientNet_B0_Weights.DEFAULT
            model = efficientnet_b0(weights=weights)
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, num_classes)

        if freeze_backbone:
            for name, p in model.named_parameters():
                if (model_name == "ResNet18" and not name.startswith("fc")) or (model_name != "ResNet18" and not name.startswith("classifier.1")):
                    p.requires_grad = False

        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=float(lr))

        train_btn = st.button("Train")
        if train_btn:
            prog = st.progress(0)
            hist = []
            for epoch in range(int(epochs)):
                model.train()
                running_loss = 0.0
                for xb, yb in train_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    optimizer.zero_grad()
                    out = model(xb)
                    loss = criterion(out, yb)
                    loss.backward()
                    optimizer.step()
                    running_loss += float(loss.item()) * xb.size(0)

                # Validation
                model.eval()
                all_preds = []
                all_true = []
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb = xb.to(device)
                        yb = yb.to(device)
                        logits = model(xb)
                        preds = torch.argmax(logits, dim=1)
                        all_preds.append(preds.cpu().numpy())
                        all_true.append(yb.cpu().numpy())

                all_preds_np = np.concatenate(all_preds) if all_preds else np.array([])
                all_true_np = np.concatenate(all_true) if all_true else np.array([])
                acc = float(accuracy_score(all_true_np, all_preds_np)) if all_true_np.size else 0.0
                prec, rec, f1, _ = precision_recall_fscore_support(all_true_np, all_preds_np, average=None, labels=list(range(num_classes)), zero_division=0)
                hist.append({"epoch": epoch + 1, "train_loss": running_loss / max(1, n_train), "val_acc": acc})
                prog.progress(int((epoch + 1) / int(epochs) * 100))

            # KPIs
            c1, c2 = st.columns(2)
            c1.metric("Val accuracy", f"{acc:.3f}")
            c2.metric("Epochs", str(epochs))

            # Confusion matrix
            cm = confusion_matrix(all_true_np, all_preds_np, labels=list(range(num_classes)))
            fig_cm = px.imshow(cm, text_auto=True, x=full_ds.classes, y=full_ds.classes, title="Confusion matrix")
            fig_cm.update_xaxes(title_text="Predicted")
            fig_cm.update_yaxes(title_text="Actual")
            st.plotly_chart(fig_cm, use_container_width=True)

            # Per-class F1
            per_class = pd.DataFrame({
                "class": full_ds.classes,
                "precision": prec,
                "recall": rec,
                "f1": f1,
            })
            st.dataframe(per_class, use_container_width=True)

            # Save
            out_dir = Path("deep_learning/image_classification/models"); out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{model_name.lower().replace(' ', '_')}_finetuned.pt"
            torch.save(model.state_dict(), out_path)
            st.success(f"Model saved: {out_path}")

        st.markdown("Notebook path: `deep_learning/image_classification/notebooks/image_classification.ipynb`")
        st.markdown("KPIs: accuracy, confusion matrix, per‑class F1.")
    render_dl_image()
elif page == "DL: Chatbot NLP":
    def render_dl_chatbot() -> None:
        st.header("Deep Learning – Chatbot NLP")
        st.caption("Intent classification + simple Q&A using LSTM/Transformers (e.g., DistilBERT).")
        st.markdown("Notebook path: `deep_learning/chatbot_nlp/notebooks/chatbot_nlp.ipynb`")
        st.markdown("Data: CSV/JSON with `text` and `intent` (and optional `answer`).")

        uploaded = st.file_uploader("Upload intents dataset (CSV with columns: text,intent[,answer])", type=["csv", "json"], key="chatbot_ds")
        sample_path = Path("deep_learning/chatbot_nlp/data/intents_sample.csv")

        @st.cache_data(show_spinner=False)
        def load_or_generate_intents(path: Path) -> pd.DataFrame:
            # If not present, synthesize a tiny dataset and persist it for future runs
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                df_syn = pd.DataFrame([
                    {"text": "hi", "intent": "greeting", "answer": "Hello! How can I help you today?"},
                    {"text": "hello", "intent": "greeting", "answer": "Hello! How can I help you today?"},
                    {"text": "what are your opening hours?", "intent": "opening_hours", "answer": "We are open Monday to Friday, 9am–6pm."},
                    {"text": "when do you open?", "intent": "opening_hours", "answer": "We are open Monday to Friday, 9am–6pm."},
                    {"text": "track my order 12345", "intent": "order_status", "answer": "You can track your order in your account under Orders."},
                    {"text": "where is my order?", "intent": "order_status", "answer": "You can track your order in your account under Orders."},
                    {"text": "how do I get a refund?", "intent": "refund_policy", "answer": "You can request a refund within 30 days from your account page."},
                    {"text": "i need support", "intent": "contact_support", "answer": "You can reach support at support@example.com or via the Help Center."},
                    {"text": "how can i contact you?", "intent": "contact_support", "answer": "You can reach support at support@example.com or via the Help Center."},
                    {"text": "bye", "intent": "goodbye", "answer": "Goodbye! Have a great day."},
                ])
                df_syn.to_csv(path, index=False)
                return df_syn
            return pd.read_csv(path)

        if uploaded is None:
            df = load_or_generate_intents(sample_path)
            st.caption(f"Using cached sample: `{sample_path.as_posix()}`")
            if sample_path.exists():
                st.download_button("Download sample CSV", data=sample_path.read_bytes(), file_name="intents_sample.csv", mime="text/csv")
        else:
            try:
                if uploaded.name.lower().endswith(".json"):
                    df = pd.read_json(uploaded)
                else:
                    df = pd.read_csv(uploaded)
            except Exception as e:
                st.error(f"Failed to read dataset: {e}")
                return

        needed = {"text", "intent"}
        if not needed.issubset(set(c.lower() for c in df.columns)):
            st.error("Dataset must include columns `text` and `intent` (case-insensitive). Optional: `answer`.")
            return
        df.columns = [c.lower() for c in df.columns]
        df = df.dropna(subset=["text", "intent"]).copy()
        df["intent"] = df["intent"].astype(str).str.strip().str.lower()
        intents = sorted(df["intent"].unique().tolist())
        st.markdown(f"Detected intents: `{', '.join(intents)}`")

        try:
            import torch  # type: ignore
            from transformers import AutoTokenizer, AutoModel  # type: ignore
        except Exception:
            st.warning("Transformers not available. Install dependencies to enable the live demo.")
            return

        @st.cache_resource(show_spinner=False)
        def load_encoder():
            tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            mdl = AutoModel.from_pretrained("distilbert-base-uncased")
            mdl.eval()
            return tok, mdl

        tokenizer, encoder = load_encoder()

        @st.cache_data(show_spinner=False)
        def build_intent_centroids(rows: pd.DataFrame):
            centroids = {}
            with torch.no_grad():
                for intent_name in intents:
                    texts = rows.loc[rows.intent == intent_name, "text"].astype(str).tolist()
                    if not texts:
                        continue
                    tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
                    outputs = encoder(**tokens)
                    cls = outputs.last_hidden_state[:, 0, :]
                    vec = torch.nn.functional.normalize(cls, dim=1).mean(dim=0)
                    vec = torch.nn.functional.normalize(vec, dim=0)
                    centroids[intent_name] = vec
            return centroids

        centroids = build_intent_centroids(df)
        if not centroids:
            st.info("No examples to compute intent centroids.")
            return

        user_text = st.text_input("Type a message", value="what are your opening hours?")
        if user_text:
            with torch.no_grad():
                toks = tokenizer([user_text], padding=True, truncation=True, return_tensors="pt")
                out = encoder(**toks)
                emb = out.last_hidden_state[:, 0, :]
                emb = torch.nn.functional.normalize(emb, dim=1)[0]
                scores = []
                for name, vec in centroids.items():
                    sim = float(torch.dot(emb, vec))
                    scores.append((name, sim))
                scores.sort(key=lambda x: x[1], reverse=True)

            names = [n for n, _ in scores]
            sims = np.array([s for _, s in scores], dtype=float)
            probs = np.exp(sims) / np.exp(sims).sum()
            res_df = pd.DataFrame({"intent": names, "score": sims, "prob": probs})
            st.dataframe(res_df.head(5), use_container_width=True)

            top_intent = names[0]
            answer = ""
            if "answer" in df.columns:
                ans_series = df.loc[df.intent == top_intent, "answer"].dropna()
                if not ans_series.empty:
                    answer = str(ans_series.iloc[0])
            st.success(f"Predicted intent: {top_intent}")
            if answer:
                st.info(f"Answer: {answer}")

        with st.expander("How to run the full notebook"):
            st.code(
                """
                # in project root
                .venv\\Scripts\\activate
                jupyter lab deep_learning/chatbot_nlp/notebooks/chatbot_nlp.ipynb
                """,
                language="bash",
            )
    render_dl_chatbot()
elif page == "Storytelling notes":
    def render_story_notes() -> None:
        st.header("Data storytelling notes")
        st.caption("Reusable ideas to enrich narrative and UX across dashboards.")
        st.markdown("- Annotations and callouts for key events")
        st.markdown("- Glossary with short definitions")
        st.markdown("- Source links and caveats near each chart")
        st.markdown("- Color‑blind friendly palettes and mobile layout")
        st.markdown("See: `data_storytelling/README.md`.")
    render_story_notes()
elif page == "Storytelling: Pandemics Map":
    def render_pandemics_map() -> None:
        st.header("Storytelling – Historical Pandemics Map")
        st.caption("Interactive Folium map showing spread by year; link to sources (WHO + historical archives).")
        st.markdown("Notebook path: `data_storytelling/pandemics_map/notebooks/pandemics_map.ipynb`")

        # Load static synthetic data by default
        data_path = Path("data_storytelling/pandemics_map/data/pandemics_sample.csv")
        if not data_path.exists():
            st.error("Sample dataset is missing. Please regenerate or provide a CSV with columns: year, country, disease, cases, deaths.")
            return
        df = pd.read_csv(data_path)
        df.columns = [c.lower() for c in df.columns]
        required = {"year", "country", "disease", "cases", "deaths"}
        if not required.issubset(set(df.columns)):
            st.error("Dataset must include columns: year, country, disease, cases, deaths.")
            return

        # Minimal country -> (lat, lon) for the sample
        country_coords = {
            "italy": (41.8719, 12.5674),
            "france": (46.2276, 2.2137),
            "england": (52.3555, -1.1743),
            "united kingdom": (55.3781, -3.4360),
            "united states": (39.8283, -98.5795),
            "india": (20.5937, 78.9629),
            "guinea": (9.9456, -9.6966),
            "liberia": (6.4281, -9.4295),
            "sierra leone": (8.4606, -11.7799),
        }

        # Controls
        years = sorted(df["year"].dropna().unique().tolist())
        sel_year = st.selectbox("Year", years, index=0)
        diseases = sorted(df["disease"].dropna().unique().tolist())
        sel_diseases = st.multiselect("Disease(s)", diseases, default=diseases)
        metric = st.radio("Metric", ["cases", "deaths"], index=0, horizontal=True)

        sub = df[(df["year"] == sel_year) & (df["disease"].isin(sel_diseases))].copy()
        if sub.empty:
            st.info("No records for the current filters.")
            return

        # Build Folium map
        try:
            import folium  # type: ignore
        except Exception:
            st.error("Folium is not available. Please install dependencies.")
            return

        # Color per disease (simple palette)
        palette = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628", "#f781bf", "#999999"]
        disease_to_color = {d: palette[i % len(palette)] for i, d in enumerate(diseases)}

        # Center map on median of known coords
        coords = [country_coords.get(str(c).lower()) for c in sub["country"]]
        coords = [c for c in coords if c]
        center = (20, 0) if not coords else (float(pd.Series([lat for lat, _ in coords]).median()), float(pd.Series([lon for _, lon in coords]).median()))
        m = folium.Map(location=center, zoom_start=2, tiles="cartodbpositron")

        for _, row in sub.iterrows():
            name = str(row["country"]) if pd.notna(row["country"]) else ""
            key = name.lower()
            loc = country_coords.get(key)
            if not loc:
                continue
            value = float(row[metric]) if pd.notna(row[metric]) else 0.0
            radius = max(5.0, min(40.0, (value ** 0.5)))  # compress dynamic range
            disease = str(row["disease"]) if pd.notna(row["disease"]) else ""
            color = disease_to_color.get(disease, "#3186cc")
            popup = folium.Popup(
                html=f"<b>{name}</b><br/>Disease: {disease}<br/>Year: {sel_year}<br/>{metric.title()}: {int(value):,}",
                max_width=250,
            )
            folium.CircleMarker(location=loc, radius=radius, color=color, fill=True, fill_opacity=0.6, popup=popup).add_to(m)

        # Render map
        from streamlit.components.v1 import html as st_html
        st_html(m._repr_html_(), height=550)

        with st.expander("Data (filtered)"):
            st.dataframe(sub[["year", "country", "disease", metric]].sort_values(metric, ascending=False), use_container_width=True)

        st.caption("Sources: WHO + historical archives (illustrative sample).")
    render_pandemics_map()

elif page == "Swiss Geo (STAC)":
    def render_stac() -> None:
        st.header("Swiss Geo (STAC)")
        st.caption("Browse Swiss federal geodata via STAC API and preview/download assets. Docs: data.geo.admin STAC landing page.")

        root = "https://data.geo.admin.ch/api/stac/v1"

        with st.expander("Step 1 – List collections"):
            provider_filter = st.text_input("Provider contains (optional)", value="")
            limit = st.number_input("Limit", min_value=1, max_value=100, value=50, step=1)
            if st.button("Fetch collections"):
                try:
                    resp = requests.get(f"{root}/collections", params={"limit": limit}, timeout=30)
                    resp.raise_for_status()
                    cols = resp.json().get("collections", [])
                    if provider_filter:
                        provider_filter_l = provider_filter.lower()
                        def has_provider(c):
                            provs = c.get("providers", []) or []
                            texts = [p.get("name", "") for p in provs]
                            return any(provider_filter_l in (t or "").lower() for t in texts)
                        cols = [c for c in cols if has_provider(c)]
                    st.session_state["stac_collections"] = cols
                    st.success(f"Loaded {len(cols)} collections")
                except Exception as e:
                    st.error(f"Failed to load collections: {e}")

            cols = st.session_state.get("stac_collections", [])
            if cols:
                dfc = pd.DataFrame([{"id": c.get("id"), "title": c.get("title"), "license": c.get("license")} for c in cols])
                st.dataframe(dfc, use_container_width=True)

        with st.expander("Step 2 – Pick a collection & search items"):
            cols = st.session_state.get("stac_collections", [])
            col_ids = [c.get("id") for c in cols] if cols else []
            collection_id = st.selectbox("Collection id", col_ids, index=0 if col_ids else None)
            c1, c2 = st.columns(2)
            with c1:
                bbox_str = st.text_input("BBox (lonmin,latmin,lonmax,latmax)", value="5.96,45.82,10.49,47.81")
            with c2:
                datetime_q = st.text_input("Datetime (RFC3339 interval)", value="../..")
            limit_items = st.number_input("Items limit", min_value=1, max_value=100, value=10)
            if st.button("Search items") and collection_id:
                try:
                    params = {"limit": int(limit_items)}
                    # Prefer POST /search for bbox/datetime if provided, else GET items
                    use_post = True
                    payload = {"collections": [collection_id], "limit": int(limit_items)}
                    if bbox_str:
                        try:
                            bbox_vals = [float(x.strip()) for x in bbox_str.split(",")]
                            if len(bbox_vals) == 4:
                                payload["bbox"] = bbox_vals
                        except Exception:
                            pass
                    if datetime_q and datetime_q != "../..":
                        payload["datetime"] = datetime_q
                    r = requests.post(f"{root}/search", json=payload, timeout=60) if use_post else requests.get(f"{root}/collections/{collection_id}/items", params=params, timeout=60)
                    r.raise_for_status()
                    items = r.json().get("features", [])
                    st.session_state["stac_items"] = items
                    st.success(f"Loaded {len(items)} items")
                except Exception as e:
                    st.error(f"Search failed: {e}")

            items = st.session_state.get("stac_items", [])
            if items:
                ids = [it.get("id") for it in items]
                item_idx = st.selectbox("Item", list(range(len(items))), format_func=lambda i: ids[i] if ids and i < len(ids) else str(i))
                sel = items[item_idx]
                st.json({k: sel[k] for k in ["id","bbox","properties"] if k in sel})

                st.subheader("Assets")
                assets = sel.get("assets", {}) or {}
                if not assets:
                    st.info("No assets on this item.")
                else:
                    for aid, a in assets.items():
                        href = a.get("href")
                        atype = a.get("type")
                        st.markdown(f"- `{aid}` — {atype or ''}")
                        if href:
                            c1, c2 = st.columns([1,1])
                            with c1:
                                st.write(href)
                            with c2:
                                if st.button(f"Download {aid}"):
                                    try:
                                        resp = requests.get(href, timeout=60)
                                        resp.raise_for_status()
                                        out_dir = Path("data_sources/geo_admin"); out_dir.mkdir(parents=True, exist_ok=True)
                                        suffix = Path(href).name
                                        out_path = out_dir / suffix
                                        out_path.write_bytes(resp.content)
                                        st.success(f"Saved: {out_path}")
                                    except Exception as e:
                                        st.error(f"Download failed: {e}")

                        # Quick preview for JSON/GeoJSON
                        if href and ((atype and "json" in atype.lower()) or href.lower().endswith((".json",".geojson"))):
                            try:
                                preview = requests.get(href, timeout=60).json()
                                # show first features if FeatureCollection
                                if isinstance(preview, dict) and preview.get("type") == "FeatureCollection":
                                    feats = preview.get("features", [])[:3]
                                    st.json({"type": "FeatureCollection", "features": feats})
                                else:
                                    st.json(preview if isinstance(preview, dict) else {"preview": str(preview)[:1000]})
                            except Exception:
                                pass

    render_stac()

 


# Sidebar footer with portfolio titles and contact
with st.sidebar:
    st.markdown("---")
    st.markdown("CO₂ & Energy evolution – Interactive dashboard")
    st.markdown("Global inequalities – Country comparator")
    st.markdown("GitHub: [michaelgermini](https://github.com/michaelgermini)")
    st.markdown("Contact: [michael@germini.info](mailto:michael@germini.info)")


