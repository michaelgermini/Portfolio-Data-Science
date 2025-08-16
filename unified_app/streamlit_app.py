import pandas as pd
import streamlit as st
import plotly.express as px
import requests
from pathlib import Path


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
        indicator_choice = st.selectbox("Default indicator if no CSV is provided", ["GDP per capita (NY.GDP.PCAP.CD)", "None"], index=0, key="ineg_indic")

    @st.cache_data(show_spinner=False)
    def load_local_sample() -> pd.DataFrame:
        try:
            sample_path = Path(__file__).resolve().parents[1] / "exploration_inegalites" / "data" / "inegalites_sample.csv"
            if sample_path.exists():
                df_local = pd.read_csv(sample_path)
                return pivot_long(df_local)
        except Exception:
            pass
        return pd.DataFrame()

    df = pd.DataFrame()
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            df = pivot_long(df)
        except Exception as e:
            st.error(f"Erreur lors de la lecture du CSV: {e}")
    else:
        # 1) Essayer l'échantillon local
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
page = st.sidebar.radio("Section", ["Inequalities", "Climate & Energy"], index=0)

if page == "Inequalities":
    render_inegalites()
else:
    render_climat()


# Sidebar footer with portfolio titles and contact
with st.sidebar:
    st.markdown("---")
    st.markdown("**CO₂ & Energy evolution – Interactive dashboard**")
    st.markdown("**Global inequalities – Country comparator**")
    st.markdown("GitHub: [michaelgermini](https://github.com/michaelgermini)")
    st.markdown("Contact: [michael@germini.info](mailto:michael@germini.info)")


