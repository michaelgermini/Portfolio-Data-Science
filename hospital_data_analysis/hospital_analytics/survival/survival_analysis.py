from __future__ import annotations

import pandas as pd
from lifelines import KaplanMeierFitter, CoxPHFitter


def fit_kaplan_meier(df: pd.DataFrame, duration_col: str, event_col: str) -> KaplanMeierFitter:
	kmf = KaplanMeierFitter()
	kmf.fit(durations=df[duration_col], event_observed=df[event_col])
	return kmf


def fit_cox_model(
	df: pd.DataFrame,
	duration_col: str,
	event_col: str,
	covariates: list[str] | None = None,
) -> CoxPHFitter:
	cox_df = df.copy()
	if covariates is not None:
		cox_df = cox_df[[duration_col, event_col] + covariates].dropna()
	cph = CoxPHFitter()
	cph.fit(cox_df, duration_col=duration_col, event_col=event_col)
	return cph
