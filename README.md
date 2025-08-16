## Data Science Portfolio – Unified Streamlit App

A single, polished Streamlit app that showcases end‑to‑end data skills: multi‑source ingestion, cleaning, interactive analytics, classic ML, deep learning, geospatial browsing, and storytelling.

### Highlights
- Inequalities: side‑by‑side country comparator (presets G7/BRICS), tabs (Time series, Latest, Ranking), deep‑linking, offline‑friendly sample
- Climate & Energy: 3D timeline for CO₂ vs energy metrics + standard time series
- ML Demos: Kaggle House Prices (regression), Spam vs Ham (text) with interpretable visuals
- Deep Learning: Image classification (ResNet18/EfficientNet‑B0) and Chatbot NLP (DistilBERT) with live in‑app demos
- Data for Good: AQI next‑day forecast and Local Climate projection to 2050
- Storytelling: Pandemics map (Folium) and an interactive playground for annotations
- Swiss Geo (STAC): browse collections/items and download assets
- Extra: Crypto Global Dashboard 3D (separate app, linked)

### Unified App – Pages
- Inequalities (Country comparator)
  - Upload tidy CSVs or use defaults; indicators in long format: `country, year, indicator, value`
  - Tabs: Time series, Latest values, Ranking; presets for quick selections
  - Deep‑linking: selections encoded in the URL query string
- Climate & Energy
  - 3D animated timeline by decade (Plotly) + 2D time series
- ML: House Prices (regression)
  - Upload Kaggle `train.csv` or auto‑use bundled sample
  - 5‑fold CV metrics (RMSE, R²), Predicted vs Actual, Residuals, Top features (GBR)
- ML: Spam Classification
  - TF‑IDF + Logistic Regression/Naive Bayes; Accuracy, Precision/Recall/F1; Try‑it text box
- DL: Image Classification
  - Transfer learning: ResNet18/EfficientNet‑B0; synthetic dataset generator; KPIs and confusion matrix; saves finetuned model
- DL: Chatbot NLP
  - Live in‑app demo with DistilBERT embeddings and intent centroids; returns top intents and an answer if provided
- Data for Good: AQI Forecast
  - Baseline (lag‑1) vs RandomForest on lagged features; RMSE/MAE; forecast chart
- Data for Good: Local Climate 2050
  - Seasonal climatology + scenario ramp (SSP); plot to 2050; CSV export
- Storytelling: Pandemics Map
  - Folium map; year/disease filters; circle size by cases/deaths; filtered table
- Storytelling notes
  - Renders `data_storytelling/README.md` in‑app (principles, glossary, patterns)
- Playground: Interactive
  - Quick narrative chart with highlight band and user annotations
- Crypto: Global 3D Dashboard
  - Linked project with 3D globe and live crypto metrics (run separately)

### Quickstart
- Python 3.11 recommended
- Install dependencies:
  - `pip install -r requirements.txt`
- Run the unified app:
  - `python -m streamlit run "unified_app/streamlit_app.py"`

On Windows (venv recommended)
```
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m streamlit run "unified_app/streamlit_app.py"
```

### Data & Samples
- Inequalities sample auto‑generated on first run if missing:
  - `exploration_inegalites/data/inegalites_sample.csv` (or `python scripts/generate_inequalities_sample.py`)
- House Prices sample (quick demo):
  - `ml_immo/data/house_prices/train.csv`
- Spam classification sample (auto‑loaded if no CSV uploaded):
  - `spam_classification/data/spam.csv`
- Image classification synthetic dataset (use sidebar button or CLI):
  - Path: `deep_learning/image_classification/data/`
  - CLI: `python deep_learning/image_classification/utils/generate_synthetic_dataset.py`
- AQI sample data:
  - `data_for_good/air_quality_forecast/data/aqi_city.csv` (or `python scripts/generate_aqi_sample.py`)
- Local climate sample data:
  - `data_for_good/local_climate_projection/data/local_temp.csv` (or `python scripts/generate_local_temp_sample.py`)
- Pandemics map sample:
  - `data_storytelling/pandemics_map/data/pandemics_sample.csv`

### Development Notes
- App UI: Streamlit + Plotly; data wrangling with pandas
- ML: scikit‑learn (regression/classification)
- DL: torch/torchvision (CPU by default); transfer learning for image classification
- NLP: transformers (DistilBERT embeddings for in‑app demo)
- STAC: simple `requests` client to query `data.geo.admin.ch` STAC API

### Security & Quality
- Linting: Ruff
- Dependency audit: pip‑audit (clean as of last run)
- SAST: Bandit (basic scan)
- Updated: `requests==2.32.4` (addresses GHSA‑9hjg‑9r4m‑mvj7)

### Deploy on Streamlit Community Cloud
- Repository: `michaelgermini/Portfolio-Data-Science`
- App file: `unified_app/streamlit_app.py`
- Python: 3.11
- Requirements: `requirements.txt`

### Contact
- GitHub: `michaelgermini`
- Email: `michael@germini.info`



