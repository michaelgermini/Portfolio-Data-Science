from typing import Optional
import pandas as pd
import streamlit as st
import plotly.express as px
import requests


st.set_page_config(page_title="Climate & Energy – 3D Timeline", layout="wide")


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


st.title("CO₂ & Energy evolution – Interactive dashboard")
st.caption("Explore the evolution of CO₂ emissions and energy consumption since 1960. Killer feature: a 3D timeline with decade scrub.")

with st.sidebar:
    st.header("Data")
    st.write("By default, OWID data are loaded if internet is available.")
    country = st.text_input("Country filter (code or partial name)", value="France")
    z_axis = st.selectbox("Z axis (height)", ["co2", "energy_per_capita", "primary_energy_consumption"], index=0)
    st.markdown("---")
    st.markdown("**CO₂ & Energy evolution – Interactive dashboard**")
    st.markdown("GitHub: [michaelgermini](https://github.com/michaelgermini)")
    st.markdown("Contact: [michael@germini.info](mailto:michael@germini.info)")

df = load_owid_co2()

if df.empty:
    st.info("Unable to load OWID data. Check your connection or import a compatible CSV (feature coming soon).")
    st.stop()

df = df[(df["year"] >= 1960)]

if country:
    mask = df["country"].str.contains(country, case=False, na=False) | df["iso_code"].str.contains(country, case=False, na=False)
    df = df[mask]

if df.empty:
    st.warning("Aucun enregistrement après filtrage.")
    st.stop()

df["decade"] = df["year"].apply(decade)

# Variables pour axes
df["energy_per_capita"] = df.get("energy_per_capita", pd.Series([None] * len(df)))
df["primary_energy_consumption"] = df.get("primary_energy_consumption", pd.Series([None] * len(df)))

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
    metric = st.selectbox("Variable", ["co2", z_axis], index=0)
    fig2d = px.line(df, x="year", y=metric, color="country", markers=True)
    st.plotly_chart(fig2d, use_container_width=True)

st.markdown("---")
st.caption("Source: Our World in Data (OWID).")



