## Data Science Portfolio – Unified Streamlit App

A cohesive portfolio that demonstrates end‑to‑end data work: data cleaning, multi‑source integration, interactive visualization, machine learning, and data storytelling. The unified Streamlit app offers a clean navigation to explore inequalities across countries and climate‑energy trends, plus links to hands‑on ML and “Data for Good” notebooks.

### What’s inside (Unified App)
- Inequalities (Country comparator)
  - Side‑by‑side country selection (left/right panels)
  - Indicators supported by default: GDP per capita, plus any custom indicators via CSV
  - Tabs: Time series, Latest values (table), Ranking (top 25, year selector)
  - Presets: G7, BRICS; quick apply to left/right
  - Deep‑linking: selections can be encoded in the URL
  - Offline‑friendly: auto‑loads a sample dataset at startup
- Climate & Energy
  - 3D timeline (decade scrub) for CO₂ vs energy metrics
  - Time series per metric
  - Country filter by name or ISO code

Why it’s useful
- Compare countries quickly across key indicators for health, education, water access, and prosperity
- Explore climate and energy trajectories with an intuitive 3D animation
- Extend with your own data by dropping tidy CSVs (long format)

Data format hints
- Long/tidy format for inequalities: `country, year, indicator, value`
- The unified app auto‑loads `exploration_inegalites/data/inegalites_sample.csv` when no upload is provided
- Climate page loads OWID CO₂ when internet is available

### Getting started
- Create a Python 3.10+ environment (recommended)
- Install dependencies: `pip install -r requirements.txt`
- Run a Streamlit dashboard:
  - Inequalities: `streamlit run "exploration_inegalites/app/streamlit_app.py"`
  - Climate & Energy: `streamlit run "climat_energie/app/streamlit_app.py"`
  - Unified app: `streamlit run "unified_app/streamlit_app.py"`

On Windows (optional venv)
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run "unified_app/streamlit_app.py"
```

### Structure
- `exploration_inegalites/`: compare health, education, water access, GDP/capita (comparative dashboard)
- `climat_energie/`: CO₂ & energy consumption, 3D timeline (dashboard)
- `ml_immo/`: house price regression (Kaggle House Prices) – notebook
- `spam_classification/`: spam vs ham – notebook
- `data_storytelling/`: components for narrative (reused by dashboards)
- `data_for_good/`: air quality forecast & local climate – notebooks (skeletons)

### Data
Apps can automatically download public datasets (OWID/WorldBank) if internet is available. Otherwise, you can import your own CSV via the interface.

Primary sources
- Our World in Data (OWID): CO₂ data and related energy metrics
- World Bank / UN / IMF: socio‑economic indicators for inequalities

### Notes
- Some heavy libraries (deep learning) are not in the base `requirements.txt` to keep installation light. See notebook sections for optional installs.

### Unified app – Navigation and usage
- Sidebar
  - Section switcher: Inequalities, Climate & Energy
  - Upload CSV and choose default indicator (inequalities)
  - Country selectors for left/right, period slider
  - Presets (G7, BRICS) with “Apply to left/right”
- Tabs (Inequalities)
  - Time series: multi‑country lines (left/right panels)
  - Latest values: last available value per country in period
  - Ranking: top 25 countries by chosen indicator for a selected year

Architecture overview
- Streamlit + Plotly for the app UI and interactive charts
- Pandas for data wrangling (long format: `country, year, indicator, value`)
- Scikit‑learn in notebooks (house prices, spam classification)

Sample dataset
- `exploration_inegalites/data/inegalites_sample.csv`
  - Countries: France, Germany, United States, India, Nigeria, Brazil, China, South Africa
  - Indicators: `gdp_per_capita_usd`, `life_expectancy_years`, `school_enrollment_primary_percent`, `water_access_percent`
  - Years: 2000, 2005, 2010, 2015, 2020

### Deploy on Streamlit Community Cloud
- Repository: `michaelgermini/Portfolio-Data-Science`
- App file: `unified_app/streamlit_app.py`
- Python version: 3.11
- Requirements file: `requirements.txt`

Optional improvements (roadmap)
- Add map view for inequalities (choropleth)
- Per‑capita toggle for climate variables
- Caching for large CSV uploads
- Simple unit tests for data loading and transformations

### Contact
- GitHub: `michaelgermini`  
- Email: `michael@germini.info`



