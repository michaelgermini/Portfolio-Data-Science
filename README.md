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

---

## 🔍 COMPREHENSIVE AUDIT - Data Science Portfolio

### 📊 **EXECUTIVE SUMMARY**

This Data Science portfolio presents a **comprehensive and well-structured project** with a unified Streamlit application covering numerous data science domains. The audit reveals **good quality code** with some improvement points in terms of security and maintenance.

---

## 🏗️ **ARCHITECTURE & STRUCTURE**

### ✅ **Strengths**
- **Modular architecture** well-organized with clear separation of responsibilities
- **Unified Streamlit application** with page-based navigation
- **Consistent data structure** with standardized formats (tidy CSV)
- **Centralized dependency management** via `requirements.txt`
- **Comprehensive documentation** in README

### 📁 **Project Structure**
```
portfolio Data Science/
├── unified_app/           # Main application
├── data_sources/          # Data sources
├── scripts/              # Generation utilities
├── exploration_inegalites/
├── deep_learning/
├── spam_classification/
├── hospital_data_analysis/
└── [other specialized modules]
```

---

## 🔒 **SECURITY**

### ⚠️ **Detected Vulnerabilities**

#### **1. Dependencies with vulnerabilities (7 issues)**
```bash
jinja2   3.1.4   GHSA-q2x7-8rv6-6q7h 3.1.5
jinja2   3.1.4   GHSA-gmj6-6f8f-6699 3.1.5  
jinja2   3.1.4   GHSA-cpwx-vrp4-4pq7 3.1.6
werkzeug 2.3.7   PYSEC-2023-221      2.3.8,3.0.1
werkzeug 2.3.7   GHSA-2g68-c3qc-8985 3.0.3
werkzeug 2.3.7   GHSA-q34m-jh98-gwm2 3.0.6
```

#### **2. Security issues in code (11 issues)**
- **9 low severity issues**: Overly permissive exception handling
- **2 medium severity issues**: Insecure Hugging Face downloads

### 🛡️ **Security Recommendations**

1. **Update vulnerable dependencies**:
   ```bash
   pip install --upgrade jinja2 werkzeug
   ```

2. **Fix insecure Hugging Face downloads**:
   ```python
   # Instead of:
   tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
   
   # Use:
   tok = AutoTokenizer.from_pretrained("distilbert-base-uncased", revision="main")
   ```

3. **Improve exception handling**:
   ```python
   # Instead of:
   except Exception:
       pass
   
   # Use:
   except (ValueError, TypeError) as e:
       logger.warning(f"Expected error: {e}")
   ```

---

## 📝 **CODE QUALITY**

### ✅ **Positive Points**
- **Well-documented code** with explanatory comments
- **Appropriate use of libraries** (pandas, scikit-learn, PyTorch)
- **Error handling** with appropriate try/except blocks
- **Consistent data structure**

### ⚠️ **Linting Issues (6 problems)**
```bash
F401: Unused import 'Lasso'
F401: Unused import 'torchvision'  
F841: Variable 'e' assigned but never used
F841: Variable 'TORCH_OK' assigned but never used
E702: Multiple statements on one line (semicolon)
```

### 🔧 **Quality Recommendations**

1. **Clean unused imports**:
   ```python
   # Remove:
   from sklearn.linear_model import Ridge, Lasso  # Lasso unused
   import torchvision  # Unused
   ```

2. **Fix unused variables**:
   ```python
   # Instead of:
   except Exception as e:
       pass
   
   # Use:
   except Exception:
       pass
   ```

3. **Separate statements on multiple lines**:
   ```python
   # Instead of:
   out_dir = Path("deep_learning/image_classification/models"); out_dir.mkdir(parents=True, exist_ok=True)
   
   # Use:
   out_dir = Path("deep_learning/image_classification/models")
   out_dir.mkdir(parents=True, exist_ok=True)
   ```

---

## 📦 **DEPENDENCY MANAGEMENT**

### ✅ **Current State**
- **21 dependencies** well-defined in `requirements.txt`
- **Pinned versions** for reproducibility
- **Clear separation** between main and optional dependencies

### ⚠️ **Outdated Dependencies (13 packages)**
```bash
streamlit    1.37.1 → 1.48.1
plotly       5.22.0 → 6.3.0
pillow       10.4.0 → 11.3.0
protobuf     5.29.5 → 6.32.0
# ... and 9 others
```

### 🔄 **Recommendations**

1. **Update critical dependencies**:
   ```bash
   pip install --upgrade streamlit plotly pillow protobuf
   ```

2. **Test after updates** to ensure compatibility

3. **Consider adding version constraints** for critical dependencies

---

## 🧪 **FEATURES & COVERAGE**

### ✅ **Implemented Features**
- ✅ **Inequality analysis** with country comparison
- ✅ **ML: House price prediction** (Kaggle House Prices)
- ✅ **ML: Spam/ham classification** with TF-IDF
- ✅ **Deep Learning: Image classification** (ResNet18/EfficientNet)
- ✅ **NLP: Chatbot with DistilBERT**
- ✅ **Data for Good: AQI and local climate forecasting**
- ✅ **Storytelling: Interactive maps** (Folium)
- ✅ **Geospatial: Swiss STAC API**
- ✅ **3D Dashboards** (Crypto, Energy)

### 📊 **Technical Coverage**
- **Machine Learning**: Regression, Classification, NLP
- **Deep Learning**: Transfer Learning, Computer Vision
- **Data Engineering**: ETL, Feature Engineering
- **Visualization**: Plotly, Folium, 3D Dashboards
- **Deployment**: Streamlit Community Cloud

---

## 🚀 **DEPLOYMENT & OPERATIONS**

### ✅ **Deployment Configuration**
- **Streamlit Community Cloud** configured
- **GitHub Repository**: `michaelgermini/Portfolio-Data-Science`
- **Python 3.11 environment** specified
- **Requirements.txt** for dependencies

### 🔧 **Operations Recommendations**

1. **Add automated tests**:
   ```python
   # tests/test_app.py
   import pytest
   import streamlit as st
   
   def test_app_loads():
       # Basic tests to verify app loads
       pass
   ```

2. **Configure CI/CD**:
   ```yaml
   # .github/workflows/test.yml
   name: Test
   on: [push, pull_request]
   jobs:
     test:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v4
         - uses: actions/setup-python@v5
         - run: pip install -r requirements.txt
         - run: python -m pytest tests/
   ```

---

## 📈 **QUALITY METRICS**

| Metric | Score | Status |
|--------|-------|--------|
| **Security** | 7/10 | ⚠️ Vulnerabilities detected |
| **Code Quality** | 8/10 | ✅ Good, some improvements |
| **Documentation** | 9/10 | ✅ Excellent |
| **Architecture** | 9/10 | ✅ Very well structured |
| **Features** | 10/10 | ✅ Complete and varied |
| **Deployment** | 8/10 | ✅ Configured, tests missing |

**Overall Score: 8.5/10** 🎯

---

## 🎯 **PRIORITY ACTION PLAN**

### 🔴 **Urgent (Security)**
1. **Update jinja2 and werkzeug** to fix vulnerabilities
2. **Fix insecure Hugging Face downloads**
3. **Improve overly permissive exception handling**

### ⚡ **Important (Quality)**
1. **Clean unused imports** (Lasso, torchvision)
2. **Fix unused variables** (e, TORCH_OK)
3. **Separate multiple statements** on one line

### 📈 **Improvement (Maintenance)**
1. **Update outdated dependencies**
2. **Add automated tests**
3. **Configure CI/CD with GitHub Actions**

---

## 🏆 **CONCLUSION**

This Data Science portfolio is **exceptionally comprehensive** and demonstrates deep mastery of numerous data science domains. The architecture is solid, documentation excellent, and features very diverse.

**Major strengths:**
- Impressive technical coverage
- Well-thought modular architecture
- Detailed documentation
- Operational deployment

**Recommended improvements:**
- Fix security vulnerabilities
- Clean code (linting)
- Add automated tests

This portfolio constitutes an **excellent example** of a professional data science project and demonstrates advanced technical skills. 🚀

---

### Contact
- GitHub: `michaelgermini`
- Email: `michael@germini.info`



