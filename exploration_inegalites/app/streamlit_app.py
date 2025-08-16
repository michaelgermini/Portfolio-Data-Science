import io
from typing import List, Dict
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px
import requests


st.set_page_config(page_title="Global Inequalities – Country Comparator", layout="wide")


@st.cache_data(show_spinner=False)
def fetch_worldbank_indicator(indicator: str) -> pd.DataFrame:
    # World Bank bulk CSV endpoint (paged API is cumbersome). We'll try OWID mirror if exists.
    # Fallback: empty DataFrame
    try:
        # Try World Bank API (CSV download). Some indicators exposed via bulk download.
        url = (
            f"https://api.worldbank.org/v2/country/all/indicator/{indicator}?downloadformat=csv"
        )
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200 and resp.headers.get("Content-Type", "").startswith("application/zip"):
            # Streamlit cannot unzip directly from memory without extra libs, skip for simplicity.
            return pd.DataFrame()
    except Exception:
        pass

    # As a practical default, use OWID datasets when available
    try:
        if indicator == "NY.GDP.PCAP.CD":
            df = pd.read_csv(
                "https://raw.githubusercontent.com/owid/owid-datasets/master/datasets/GDP%20per%20capita%20(Penn%20World%20Table)/GDP%20per%20capita%20(Penn%20World%20Table).csv"
            )
            # Normalize columns
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
    # Try to infer wide format (country/year columns)
    if "Country Name" in df.columns and any(c.isdigit() for c in df.columns):
        value_cols = [c for c in df.columns if c.isdigit()]
        tidy = df.melt(id_vars=["Country Name"], value_vars=value_cols, var_name="year", value_name="value")
        tidy = tidy.rename(columns={"Country Name": "country"})
        tidy["indicator"] = "value"
        tidy["year"] = tidy["year"].astype(int)
        return tidy
    return df


def country_selector(label: str, options: List[str]) -> List[str]:
    return st.multiselect(label, options=options, default=["France", "Germany"] if "France" in options and "Germany" in options else options[:2])


st.title("Global inequalities analysis – Country comparator")
st.caption("Upload your CSV (World Bank/UN/IMF) or use a default indicator (GDP per capita). Compare two sets of countries side-by-side.")

with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload a CSV (expected columns: country, year, indicator, value)", type=["csv"]) 
    indicator = st.selectbox(
        "Default indicator if no CSV is provided",
        ["GDP per capita (NY.GDP.PCAP.CD)", "None"],
        index=0,
    )
    st.markdown("---")
    st.markdown("**Global inequalities – Country comparator**")
    st.markdown("GitHub: [michaelgermini](https://github.com/michaelgermini)")
    st.markdown("Contact: [michael@germini.info](mailto:michael@germini.info)")

@st.cache_data(show_spinner=False)
def load_local_sample() -> pd.DataFrame:
    try:
        sample_path = Path(__file__).resolve().parents[1] / "data" / "inegalites_sample.csv"
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
    if df.empty and indicator.startswith("GDP"):
        df = fetch_worldbank_indicator("NY.GDP.PCAP.CD")

if df.empty:
    st.info("No data available. Upload a CSV (country, year, indicator, value) or enable internet to use the default source.")
    st.stop()

available_countries = sorted(df["country"].dropna().unique().tolist())
available_indicators = sorted(df["indicator"].dropna().unique().tolist())

with st.sidebar:
    selected_indicator = st.selectbox("Indicator", available_indicators, index=0)
    countries_left = country_selector("Countries – left panel", available_countries)
    countries_right = country_selector("Countries – right panel", available_countries)
    year_range = st.slider(
        "Period",
        min_value=int(pd.to_numeric(df["year"], errors="coerce").min()),
        max_value=int(pd.to_numeric(df["year"], errors="coerce").max()),
        value=(int(pd.to_numeric(df["year"], errors="coerce").min()), int(pd.to_numeric(df["year"], errors="coerce").max())),
    )

def filter_df(base: pd.DataFrame, countries: List[str], indicator: str, year_min: int, year_max: int) -> pd.DataFrame:
    tmp = base.copy()
    tmp = tmp[(tmp["indicator"] == indicator)]
    tmp = tmp[tmp["country"].isin(countries)]
    tmp = tmp[(pd.to_numeric(tmp["year"], errors="coerce") >= year_min) & (pd.to_numeric(tmp["year"], errors="coerce") <= year_max)]
    tmp["year"] = pd.to_numeric(tmp["year"], errors="coerce").astype("Int64")
    return tmp.dropna(subset=["year"]).sort_values(["country", "year"]) 


left, right = st.columns(2)

with left:
    st.subheader("Left panel")
    dfl = filter_df(df, countries_left, selected_indicator, year_range[0], year_range[1])
    if dfl.empty:
        st.warning("Aucune donnée pour la sélection.")
    else:
        fig = px.line(dfl, x="year", y="value", color="country", markers=True, title=selected_indicator)
        st.plotly_chart(fig, use_container_width=True)
        last_year = dfl.groupby("country").apply(lambda x: x.sort_values("year").tail(1)).reset_index(drop=True)
        st.dataframe(last_year[["country", "year", "value"]].rename(columns={"value": "last_year_value"}), use_container_width=True)

with right:
    st.subheader("Right panel")
    dfr = filter_df(df, countries_right, selected_indicator, year_range[0], year_range[1])
    if dfr.empty:
        st.warning("Aucune donnée pour la sélection.")
    else:
        fig = px.line(dfr, x="year", y="value", color="country", markers=True, title=selected_indicator)
        st.plotly_chart(fig, use_container_width=True)
        last_year = dfr.groupby("country").apply(lambda x: x.sort_values("year").tail(1)).reset_index(drop=True)
        st.dataframe(last_year[["country", "year", "value"]].rename(columns={"value": "last_year_value"}), use_container_width=True)

st.markdown("---")
st.caption("Tip: Import multiple indicators by concatenating your sources (UN/World Bank/IMF) in tidy long format: country, year, indicator, value.")



