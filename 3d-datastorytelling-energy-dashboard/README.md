## EnergyScape 3D – Data Storytelling de la Transition Énergétique

Une application Streamlit + React/Three.js qui raconte l'évolution de la consommation énergétique (2000 → 2050) via une scène 3D interactive.

### Stack
- Backend/UI data: Streamlit, Pandas, Numpy, Altair
- Frontend 3D: React, Three.js, Framer Motion, d3-scale (Vite)
- Intégration: `st.components.v1.html` + `postMessage` JSON

### Prérequis
- Python 3.10+
- Node.js 18+

### Installation
1) Dépendances Python
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2) Frontend React 3D
```
cd frontend
npm install
npm run dev
```

3) Lancer Streamlit (nouvelle console)
```
streamlit run streamlit_app.py
```

### Développement
- Par défaut, l'app Streamlit tente d'embarquer l'iframe sur `http://localhost:5173` (serveur Vite).
- Activez/désactivez le mode "Dev server" dans la barre latérale.

### Build de production (frontend)
```
cd frontend
npm run build
```
Puis, dans Streamlit, basculez le switch "Dev server" sur OFF. L'app lira `frontend/dist/index.html` et injectera les données côté client.

### Données
- Vous pouvez charger un CSV (colonnes conseillées: `year,fossil,renewables,nuclear,solar,total_mwh,co2, savings`) ou utiliser le dataset par défaut généré.
- Filtres: année, types d'énergie.
- KPIs: consommation totale, % renouvelables, CO2, économies.

### Notes
- Cette maquette est un squelette prêt à brancher et étendre (chapitres/KPIs, animations, styles, etc.).


