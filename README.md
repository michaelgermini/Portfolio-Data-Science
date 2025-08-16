## Data Science Portfolio

Structured project to showcase a complete skill set: exploration, visualization, classic ML, deep learning, storytelling, and data for good.

### Getting started
- Create a Python 3.10+ environment (recommended)
- Install dependencies: `pip install -r requirements.txt`
- Run a Streamlit dashboard:
  - Inequalities: `streamlit run "exploration_inegalites/app/streamlit_app.py"`
  - Climate & Energy: `streamlit run "climat_energie/app/streamlit_app.py"`
  - Unified app: `streamlit run "unified_app/streamlit_app.py"`

### Structure
- `exploration_inegalites/`: compare health, education, water access, GDP/capita (comparative dashboard)
- `climat_energie/`: CO₂ & energy consumption, 3D timeline (dashboard)
- `ml_immo/`: house price regression (Kaggle House Prices) – notebook
- `spam_classification/`: spam vs ham – notebook
- `data_storytelling/`: components for narrative (reused by dashboards)
- `data_for_good/`: air quality forecast & local climate – notebooks (skeletons)

### Data
Apps can automatically download public datasets (OWID/WorldBank) if internet is available. Otherwise, you can import your own CSV via the interface.

### Notes
- Some heavy libraries (deep learning) are not in the base `requirements.txt` to keep installation light. See notebook sections for optional installs.



