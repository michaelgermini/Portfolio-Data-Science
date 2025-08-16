## Data Science Portfolio – Unified Streamlit App

A single, polished Streamlit app that showcases end‑to‑end data skills: multi‑source ingestion, cleaning, interactive analytics, classic ML, deep learning, and geospatial browsing.

### Highlights
- Inequalities: side‑by‑side country comparator with presets (G7, BRICS), tabs (Time series, Latest, Ranking), deep‑linking via URL, and offline‑friendly sample auto‑load
- Climate & Energy: 3D timeline for CO₂ vs energy metrics, plus regular time series; country filter by name or ISO code
- ML Demos: Kaggle House Prices (regression) and Spam vs Ham (text classification) with interpretable visuals
- Deep Learning Demos: Image classification (ResNet18/EfficientNet‑B0, transfer learning) with a built‑in synthetic dataset generator
- Swiss Geo (STAC): browse federal geodata collections/items and download assets

### Unified App – Pages
- Inequalities (Country comparator)
  - Upload tidy CSVs or use defaults; indicators in long format: `country, year, indicator, value`
  - Tabs: Time series, Latest values, Ranking; presets for quick selections
  - Deep‑linking: selections encoded in the URL query string
- Climate & Energy
  - 3D animated timeline by decade with Plotly
  - Secondary 2D time series chart for the selected metric
- ML: House Prices (regression)
  - Upload Kaggle `train.csv` or auto‑use bundled sample
  - 5‑fold CV metrics (RMSE, R²), Predicted vs Actual, Residuals, Top features (GBR)
- ML: Spam Classification
  - TF‑IDF + Logistic Regression/Naive Bayes; reports Accuracy, Precision/Recall/F1
  - Try‑it input box to classify your own text
- DL: Image Classification
  - Transfer learning: ResNet18/EfficientNet‑B0 on your foldered dataset
  - Sidebar button to generate a tiny synthetic dataset (classes: red/green/blue)
  - KPIs: validation accuracy, confusion matrix, per‑class F1; saves a finetuned model
- DL: Chatbot NLP (notebook scaffold)
- Data for Good: AQI Forecast & Local Climate 2050 (notebook scaffolds)
- Storytelling: Pandemics Map (notebook scaffold)
- Swiss Geo (STAC)
  - List collections, search items (bbox/datetime), preview JSON/GeoJSON, download assets to `data_sources/geo_admin/`

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
  - `exploration_inegalites/data/inegalites_sample.csv`
- House Prices sample (for quick demo):
  - `ml_immo/data/house_prices/train.csv`
- Spam classification sample (auto‑loaded if no CSV uploaded):
  - `spam_classification/data/spam.csv`
- Image classification synthetic dataset (use sidebar button or CLI):
  - Path: `deep_learning/image_classification/data/`
  - CLI: `python deep_learning/image_classification/utils/generate_synthetic_dataset.py`

### Development Notes
- App UI: Streamlit + Plotly; data wrangling with pandas
- ML: scikit‑learn (regression/classification)
- DL: torch/torchvision (CPU by default); transfer learning for image classification
- STAC: simple `requests` client to query `data.geo.admin.ch` STAC API

### Security & Quality
- Linting: Ruff
- Dependency audit: pip‑audit (clean as of last run)
- SAST: Bandit (basic scan)
- Updated: `requests==2.32.4` (addresses GHSA‑9hjg‑9r4m‑mvj7)

### Deploy on Streamlit Community Cloud
- Repository: `michaelgermini/portfolio-data-science`
- App file: `unified_app/streamlit_app.py`
- Python: 3.11
- Requirements: `requirements.txt`

### Contact
- GitHub: `michaelgermini`
- Email: `michael@germini.info`



