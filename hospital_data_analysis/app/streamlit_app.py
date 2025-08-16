import streamlit as st
import pandas as pd
import numpy as np

from hospital_analytics.data.synthetic import generate_synthetic_hospital_data
from hospital_analytics.data.preprocessing import (
	load_csv,
	prepare_readmission_dataset,
	create_survival_dataset,
	aggregate_daily_admissions,
)
from hospital_analytics.models.readmission import train_and_eval_models
from hospital_analytics.survival.survival_analysis import fit_kaplan_meier, fit_cox_model
from hospital_analytics.forecasting.resource_forecasting import fit_sarima, forecast
from hospital_analytics.optimization.resource_optimization import allocate_beds

st.set_page_config(page_title="Hospital Data Analysis", layout="wide")

st.sidebar.title("Data")
mode = st.sidebar.radio("Chargement des données", ["Données synthétiques", "Téléverser CSV"])  # noqa: E501

if mode == "Téléverser CSV":
	upload = st.sidebar.file_uploader("Choisir un CSV", type=["csv"])
	if upload is not None:
		data = pd.read_csv(upload)
		data = data.copy()
	else:
		st.sidebar.info("En attente d'un fichier… Utilisation des données synthétiques par défaut.")
		data = generate_synthetic_hospital_data()
else:
	data = generate_synthetic_hospital_data()

st.title("Analyse de données hospitalières")
st.caption("Prédiction des réadmissions, survie, prévisions et optimisation")

st.subheader("Aperçu des données")
st.dataframe(data.head(50), use_container_width=True)

# Tabs
ml_tab, survival_tab, forecast_tab, optim_tab = st.tabs([
	"Réadmissions (ML)",
	"Survie (KM/Cox)",
	"Prévisions (SARIMA)",
	"Optimisation (Lits)",
])

with ml_tab:
	st.markdown("**Classification des réadmissions (30 jours)**")
	try:
		X, y, num_cols, cat_cols = prepare_readmission_dataset(data)
		results = train_and_eval_models(X, y, num_cols, cat_cols)
		st.write(pd.DataFrame([{ "Model": r.name, "ROC AUC": r.roc_auc, "Accuracy": r.accuracy, "F1": r.f1 } for r in results]))
	except Exception as e:
		st.warning(f"ML non disponible: {e}")

with survival_tab:
	st.markdown("**Kaplan–Meier & Cox**")
	try:
		df_surv, duration_col, event_col = create_survival_dataset(data)
		kmf = fit_kaplan_meier(df_surv, duration_col, event_col)
		st.pyplot(kmf.plot_survival_function().figure)

		covars = [c for c in ["age", "comorbidity_index", "prior_admissions", "length_of_stay"] if c in df_surv.columns]
		if covars:
			cph = fit_cox_model(df_surv, duration_col, event_col, covariates=covars)
			st.write(cph.summary)
	except Exception as e:
		st.warning(f"Survie non disponible: {e}")

with forecast_tab:
	st.markdown("**Prévision des admissions quotidiennes (30 jours)**")
	try:
		daily = aggregate_daily_admissions(data)
		res = fit_sarima(daily)
		fc = forecast(res, steps=30)
		st.line_chart(fc.set_index("date")["forecast"])
		st.dataframe(fc.tail(10))
	except Exception as e:
		st.warning(f"Prévision non disponible: {e}")

with optim_tab:
	st.markdown("**Allocation de lits entre services**")
	try:
		# Build simple demand/capacity from data
		by_dept = data.groupby("department").size().rename("demand").to_frame().reset_index()
		by_dept["capacity"] = (by_dept["demand"] * 0.8).round().astype(int).clip(lower=5)
		res = allocate_beds(
			departments=by_dept["department"].tolist(),
			demand=by_dept["demand"].astype(int).tolist(),
			capacity=by_dept["capacity"].astype(int).tolist(),
		)
		st.dataframe(res)
	except Exception as e:
		st.warning(f"Optimisation non disponible: {e}")
