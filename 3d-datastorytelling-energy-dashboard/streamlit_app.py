import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import streamlit.components.v1 as components


APP_DIR = Path(__file__).parent
FRONTEND_DIR = APP_DIR / "frontend"
FRONTEND_DIST_INDEX = FRONTEND_DIR / "dist" / "index.html"


def load_default_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    years = np.arange(2000, 2051)

    # Trend assumptions
    fossil_start, fossil_end = 70.0, 15.0
    ren_start, ren_end = 20.0, 65.0
    nuc_start, nuc_end = 10.0, 20.0

    fossil = np.linspace(fossil_start, fossil_end, len(years))
    renew = np.linspace(ren_start, ren_end, len(years))
    nuclear = np.linspace(nuc_start, nuc_end, len(years))

    # Normalize to 100%
    total_pct = fossil + renew + nuclear
    fossil = fossil / total_pct * 100.0
    renew = renew / total_pct * 100.0
    nuclear = nuclear / total_pct * 100.0

    # Total consumption trend (MWh)
    base_total = np.linspace(120000.0, 90000.0, len(years))
    seasonal = 0.05 * base_total * np.sin(np.linspace(0, 6 * np.pi, len(years)))
    total_mwh = base_total + seasonal

    # CO2 emissions proportional to fossils (simplified)
    co2 = total_mwh * (fossil / 100.0) * 0.25

    # Savings increase over time
    savings = np.linspace(0, 2500000.0, len(years))

    solar = np.maximum(0.0, renew * 0.35)  # proxy for solar share within renewables

    df = pd.DataFrame(
        {
            "year": years,
            "fossil": fossil,
            "renewables": renew,
            "nuclear": nuclear,
            "solar": solar,
            "total_mwh": total_mwh,
            "co2": co2,
            "savings": savings,
        }
    )
    return df


def coerce_uploaded_df(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    expected_cols = {"year", "fossil", "renewables", "nuclear", "solar"}
    if not expected_cols.issubset(set(df.columns)):
        st.warning(
            "CSV incomplet. Colonnes attendues: year, fossil, renewables, nuclear, solar (totaux/kpis optionnels).\nUtilisation du dataset par défaut."
        )
        return load_default_data()
    # Ensure numeric
    for col in ["fossil", "renewables", "nuclear", "solar", "total_mwh", "co2", "savings"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def compute_kpis(df: pd.DataFrame, selected_year: int) -> dict:
    row = df.loc[df["year"] == selected_year].iloc[0]
    total_mwh = row.get("total_mwh", np.nan)
    pct_ren = row["renewables"]
    co2 = row.get("co2", np.nan)
    savings = row.get("savings", np.nan)
    return {
        "total_mwh": float(total_mwh) if pd.notna(total_mwh) else None,
        "pct_renewables": float(pct_ren),
        "co2": float(co2) if pd.notna(co2) else None,
        "savings": float(savings) if pd.notna(savings) else None,
    }


def prepare_payload(df: pd.DataFrame, selected_year: int, focus: str, energies: list[str], chapter: str) -> dict:
    row = df.loc[df["year"] == selected_year].iloc[0]
    energy_map = {
        "fossil": {"label": "Fossiles", "color": "#ff5252"},
        "renewables": {"label": "Renouvelables", "color": "#2ecc71"},
        "nuclear": {"label": "Nucléaire", "color": "#3498db"},
        "solar": {"label": "Solaire", "color": "#f1c40f"},
    }
    series = []
    for key in energies:
        if key not in energy_map:
            continue
        series.append(
            {
                "key": key,
                "label": energy_map[key]["label"],
                "value_pct": float(row[key]),
                "color": energy_map[key]["color"],
            }
        )

    kpis = compute_kpis(df, selected_year)
    payload = {
        "year": int(selected_year),
        "series": series,
        "kpis": kpis,
        "focus": focus,
        "chapter": chapter,
        "meta": {
            "title": "EnergyScape 3D",
            "unit": "% part d'énergie",
            "updated_at": int(time.time()),
        },
    }
    return payload


def render_frontend(payload: dict, dev_mode: bool, height: int = 640):
    if dev_mode:
        html = f"""
        <div id=host style="height:{height}px; width:100%">
          <iframe id="energy3d" src="http://localhost:5173" style="width:100%; height:100%; border:0; border-radius:8px; background:#0b0f1a"></iframe>
        </div>
        <script>
          const payload = {json.dumps(payload)};
          const frame = document.getElementById('energy3d');
          function post() {{
            try {{ frame.contentWindow.postMessage({{ type: 'ENERGY_PAYLOAD', payload }}, '*'); }} catch(e) {{}}
          }}
          window.addEventListener('message', (ev) => {{
            if (ev && ev.data && ev.data.type === 'FRONTEND_READY') {{ post(); }}
          }});
          setTimeout(post, 800);
        </script>
        """
        components.html(html, height=height)
        return

    # Production: read built index and inject payload
    if FRONTEND_DIST_INDEX.exists():
        html_text = FRONTEND_DIST_INDEX.read_text(encoding="utf-8")
        inject = f"<script>window.__INITIAL_ENERGY_PAYLOAD__ = {json.dumps(payload)}; window.parent && window.parent.postMessage && window.parent.postMessage({{ type: 'FRONTEND_READY' }}, '*');</script>"
        if "</head>" in html_text:
            html_text = html_text.replace("</head>", inject + "</head>")
        else:
            html_text = html_text + inject
        components.html(html_text, height=height)
    else:
        st.info("Build frontend manquant. Activez le mode Dev server ou exécutez `npm run build` dans `frontend/`.")


def main():
    st.set_page_config(page_title="EnergyScape 3D", layout="wide")
    st.title("EnergyScape 3D – Data Storytelling")

    with st.sidebar:
        st.header("Données & Contrôles")
        dev_mode = st.toggle("Dev server (Vite)", value=True)
        uploaded = st.file_uploader("Charger un CSV", type=["csv"]) 
        if uploaded is not None:
            df = coerce_uploaded_df(uploaded)
        else:
            df = load_default_data()

        chapter = st.radio("Chapitre", ["Vue d’ensemble", "Chauffage", "Électricité", "Photovoltaïque"], index=0)

        energies_all = ["fossil", "renewables", "nuclear", "solar"]
        energies = st.multiselect(
            "Types d'énergie",
            energies_all,
            default=energies_all,
            format_func=lambda k: {"fossil": "Fossiles", "renewables": "Renouvelables", "nuclear": "Nucléaire", "solar": "Solaire"}[k],
        )

        if "autoplay" not in st.session_state:
            st.session_state.autoplay = False
        if "year" not in st.session_state:
            st.session_state.year = int(df["year"].min())

        min_year = int(df["year"].min())
        max_year = int(df["year"].max())

        st.session_state.year = st.slider("Année", min_value=min_year, max_value=max_year, value=st.session_state.year, step=1)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("▶️ Autoplay" if not st.session_state.autoplay else "⏸️ Pause"):
                st.session_state.autoplay = not st.session_state.autoplay
        with c2:
            if st.button("↩️ Reset"):
                st.session_state.year = min_year
                st.session_state.autoplay = False

    # Autoplay tick
    if st.session_state.autoplay:
        time.sleep(0.6)
        st.session_state.year = st.session_state.year + 1 if st.session_state.year < max_year else min_year
        st.rerun()

    # KPIs
    k = compute_kpis(df, st.session_state.year)
    col1, col2, col3, col4 = st.columns(4)
    focus = st.session_state.get("focus", "none")
    with col1:
        if st.button("Consommation totale (MWh)"):
            focus = "total"
        st.metric("Consommation totale", f"{k['total_mwh']:.0f}" if k['total_mwh'] else "-", "")
    with col2:
        if st.button("% Renouvelables"):
            focus = "renewables"
        st.metric("% Renouvelables", f"{k['pct_renewables']:.1f}%")
    with col3:
        if st.button("Émissions CO₂"):
            focus = "co2"
        st.metric("Émissions CO₂", f"{k['co2']:.0f}" if k['co2'] else "-", "")
    with col4:
        if st.button("Économies (€)"):
            focus = "savings"
        st.metric("Économies", f"{k['savings']:.0f}" if k['savings'] else "-", "")
    st.session_state.focus = focus

    # 2D chart (Altair) – optionnel
    with st.expander("Tendance % par énergie (2000 → 2050)", expanded=False):
        melted = df.melt(id_vars=["year"], value_vars=["fossil", "renewables", "nuclear", "solar"], var_name="energy", value_name="pct")
        chart = (
            alt.Chart(melted)
            .mark_line()
            .encode(
                x="year:O",
                y=alt.Y("pct:Q", title="%"),
                color=alt.Color(
                    "energy:N",
                    scale=alt.Scale(
                        domain=["fossil", "renewables", "nuclear", "solar"],
                        range=["#ff5252", "#2ecc71", "#3498db", "#f1c40f"],
                    ),
                ),
                tooltip=["year", "energy", alt.Tooltip("pct", format=".1f")],
            )
            .properties(height=240)
        )
        st.altair_chart(chart, use_container_width=True)

    # Prepare and send payload to 3D frontend
    chapter_allowed = {
        "Vue d’ensemble": ["fossil", "renewables", "nuclear", "solar"],
        "Chauffage": ["fossil", "renewables"],
        "Électricité": ["renewables", "nuclear"],
        "Photovoltaïque": ["solar"],
    }
    energies_to_send = [e for e in energies if e in chapter_allowed.get(chapter, energies)]
    if not energies_to_send:
        energies_to_send = chapter_allowed.get(chapter, energies)
    payload = prepare_payload(df, st.session_state.year, st.session_state.focus, energies_to_send, chapter)
    render_frontend(payload, dev_mode=dev_mode, height=640)


if __name__ == "__main__":
    main()


