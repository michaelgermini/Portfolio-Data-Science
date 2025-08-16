# Hospital Data Analysis – Starter Kit

Projet prêt à l’emploi pour analyser des données hospitalières et construire un dashboard interactif (Streamlit) couvrant :

- Prédiction des réadmissions (Logistic Regression, Random Forest, XGBoost optionnel)
- Analyse de survie (Kaplan–Meier, Cox)
- Prévision des flux (SARIMA via statsmodels)
- Optimisation simple des ressources (allocation de lits via programmation linéaire)

Le projet fonctionne avec vos CSV (upload dans le dashboard) ou avec des données synthétiques générées automatiquement.

## Installation

1) Créer un environnement Python 3.10+ (recommandé)

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell
```

2) Installer les dépendances

```bash
pip install -r requirements.txt
```

## Lancer le dashboard

```bash
streamlit run app/streamlit_app.py
```

## Données attendues (exemples)

Le dashboard accepte un CSV avec des colonnes proches de :

- `patient_id`, `age`, `sex`, `diagnosis`, `department`
- `comorbidity_index`, `prior_admissions`, `length_of_stay`
- `admit_date`, `discharge_date`, `readmitted_within_30d`
- `time_to_event` (jours), `event_observed` (0/1) pour l’analyse de survie

Si vos colonnes diffèrent, utilisez la génération synthétique (bouton par défaut) ou adaptez `hospital_analytics/data/preprocessing.py`.

## Structure

```
Hospital data analysis/
├─ app/
│  └─ streamlit_app.py
├─ hospital_analytics/
│  ├─ __init__.py
│  ├─ data/
│  │  ├─ preprocessing.py
│  │  └─ synthetic.py
│  ├─ forecasting/
│  │  └─ resource_forecasting.py
│  ├─ models/
│  │  └─ readmission.py
│  ├─ optimization/
│  │  └─ resource_optimization.py
│  └─ survival/
│     └─ survival_analysis.py
├─ README.md
└─ requirements.txt
```

## Notes

- XGBoost est optionnel : s’il n’est pas installé, le dashboard continue avec Logistic Regression et Random Forest.
- L’optimisation utilise `scipy.optimize.linprog` (pas besoin de solveur externe).
- Les modules sont écrits pour être lisibles et faciles à adapter à vos conventions métier.
