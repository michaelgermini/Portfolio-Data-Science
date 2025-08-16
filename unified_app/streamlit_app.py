import pandas as pd
import streamlit as st
import plotly.express as px
import requests
from pathlib import Path
 
# ML/Preprocessing imports for House Prices page
import numpy as np
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor


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


st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Section",
    [
        "Inequalities",
        "Climate & Energy",
        "ML: House Prices",
        "ML: Spam Classification",
        "Data for Good: AQI Forecast",
        "Data for Good: Local Climate 2050",
        "Storytelling notes",
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

        if uploaded_train is None:
            st.info("Upload the Kaggle training file to see metrics and charts.")
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
        st.markdown("Run locally:")
        st.code(
            """
            # in project root
            .venv\\Scripts\\activate  # if you use the venv
            jupyter lab spam_classification/notebooks/spam_classification.ipynb
            """,
            language="bash",
        )
        st.markdown("Data: put `spam.csv` under `spam_classification/data/` with columns `text`, `label`.")
    render_ml_spam()
elif page == "Data for Good: AQI Forecast":
    def render_dfg_aqi() -> None:
        st.header("Data for Good – AQI Forecast")
        st.caption("Next‑day city AQI using time‑series baselines; optional LSTM.")
        st.markdown("Notebook path: `data_for_good/air_quality_forecast/notebooks/aqi_forecast.ipynb`")
        st.markdown("Run locally:")
        st.code(
            """
            # in project root
            .venv\\Scripts\\activate  # if you use the venv
            jupyter lab data_for_good/air_quality_forecast/notebooks/aqi_forecast.ipynb
            """,
            language="bash",
        )
        st.markdown("Data: `data_for_good/air_quality_forecast/data/aqi_city.csv` with columns `date, aqi, temp, wind, ...`.")
    render_dfg_aqi()
elif page == "Data for Good: Local Climate 2050":
    def render_dfg_climate() -> None:
        st.header("Data for Good – Local climate projection (2050)")
        st.caption("Bias correction from local series + scenario exploration (RCP/SSP).")
        st.markdown("Notebook path: `data_for_good/local_climate_projection/notebooks/local_climate_projection.ipynb`")
        st.markdown("Run locally:")
        st.code(
            """
            # in project root
            .venv\\Scripts\\activate  # if you use the venv
            jupyter lab data_for_good/local_climate_projection/notebooks/local_climate_projection.ipynb
            """,
            language="bash",
        )
        st.markdown("Data: `data_for_good/local_climate_projection/data/local_temp.csv` with columns `date, tmean`.")
    render_dfg_climate()
else:
    def render_story_notes() -> None:
        st.header("Data storytelling notes")
        st.caption("Reusable ideas to enrich narrative and UX across dashboards.")
        st.markdown("- Annotations and callouts for key events")
        st.markdown("- Glossary with short definitions")
        st.markdown("- Source links and caveats near each chart")
        st.markdown("- Color‑blind friendly palettes and mobile layout")
        st.markdown("See: `data_storytelling/README.md`.")
    render_story_notes()

 


# Sidebar footer with portfolio titles and contact
with st.sidebar:
    st.markdown("---")
    st.markdown("**CO₂ & Energy evolution – Interactive dashboard**")
    st.markdown("**Global inequalities – Country comparator**")
    st.markdown("GitHub: [michaelgermini](https://github.com/michaelgermini)")
    st.markdown("Contact: [michael@germini.info](mailto:michael@germini.info)")


