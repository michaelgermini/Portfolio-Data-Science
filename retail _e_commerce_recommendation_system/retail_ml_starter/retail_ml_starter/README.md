# Retail / E-commerce ML Starter (Python)

This starter gives you two end‑to‑end building blocks:

1. **Recommandation produit** (filtrage collaboratif par factorisation de matrices, impl. NumPy)
   - **KPIs**: `precision@k`, `ndcg@k`
   - **Script démo**: `python main_reco_demo.py`

2. **Prévision de ventes** (baseline SARIMA + options Prophet/LSTM)
   - **KPIs**: `MAPE`, `RMSE`
   - **Script démo** (SARIMA): `python forecasting/sarima_demo.py`
   - **Script démo** (Prophet, si installé): `python forecasting/prophet_demo.py`
   - **Script démo** (LSTM, si installé): `python forecasting/lstm_demo.py`

> Données d'exemple **synthétiques** pour démarrer rapidement. Branchez vos données réelles ensuite.

## Installation

Python 3.10+ recommandé.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **Optionnel** : installez aussi Prophet / TensorFlow si vous voulez tester ces variantes.
> Prophet peut nécessiter `pystan` selon votre plateforme.

## Structure

```
retail_ml_starter/
├── README.md
├── requirements.txt
├── reco/
│   ├── __init__.py
│   ├── recommender.py
│   ├── metrics.py
│   └── data_utils.py
├── forecasting/
│   ├── __init__.py
│   ├── sarima_demo.py
│   ├── prophet_demo.py
│   └── lstm_demo.py
└── main_reco_demo.py
```

## Brancher vos données

### Recommandation
- Attendez un **matrice user-item implicite** binaire (achat/clic = 1, sinon 0).
- Voir `reco/data_utils.py` pour un exemple de split train/test (leave‑one‑out).
- Pour des signaux explicites (notes), adaptez la **perte** dans `reco/recommender.py`.

### Prévision
- Format simple `pandas.Series` ou DataFrame avec une colonne `sales` indexée par une date (`DatetimeIndex`).
- Voir `forecasting/sarima_demo.py` pour un exemple de série avec saisonnalité.

## Roadmap d'amélioration
- **Biais utilisateur/article**, **regularisation** par groupe, **weighting** pour feedback implicite (type WRMF).
- **Features** de contenu (hybride) via régression sur embeddings.
- **Échelle**: passer à des implémentations optimisées (Faiss/Annoy pour k-NN, `implicit` pour ALS, Spark pour large scale).
- **MLOps**: Hydra/MLflow, tests unitaires, CI, packaging, Docker, monitoring.

Bon build !
