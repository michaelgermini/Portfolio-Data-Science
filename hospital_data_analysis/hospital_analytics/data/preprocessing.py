from __future__ import annotations

from typing import Iterable, Tuple
import pandas as pd


def coerce_datetime(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
	updated = df.copy()
	for c in cols:
		if c in updated.columns:
			updated[c] = pd.to_datetime(updated[c], errors="coerce")
	return updated


def load_csv(filepath: str) -> pd.DataFrame:
	df = pd.read_csv(filepath)
	df = coerce_datetime(df, ["admit_date", "discharge_date"])
	return df


def prepare_readmission_dataset(
	df: pd.DataFrame,
	target_col: str = "readmitted_within_30d",
) -> Tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
	candidate_numeric = [
		"age",
		"comorbidity_index",
		"prior_admissions",
		"length_of_stay",
	]
	candidate_categorical = [
		"sex",
		"diagnosis",
		"department",
	]

	numeric_cols = [c for c in candidate_numeric if c in df.columns]
	categorical_cols = [c for c in candidate_categorical if c in df.columns]

	missing_target = target_col not in df.columns
	if missing_target:
		raise ValueError(
			f"Target column '{target_col}' is missing. Provide it or generate synthetic data."
		)

	feature_cols = numeric_cols + categorical_cols
	X = df[feature_cols].copy()
	y = df[target_col].astype(int).copy()

	return X, y, numeric_cols, categorical_cols


def create_survival_dataset(
	df: pd.DataFrame,
	duration_col: str = "time_to_event",
	event_col: str = "event_observed",
) -> Tuple[pd.DataFrame, str, str]:
	if duration_col not in df.columns or event_col not in df.columns:
		raise ValueError(
			f"Expected columns '{duration_col}' and '{event_col}' for survival analysis."
		)
	return df.copy(), duration_col, event_col


def aggregate_daily_admissions(df: pd.DataFrame) -> pd.DataFrame:
	if "admit_date" not in df.columns:
		raise ValueError("Column 'admit_date' is required for time series aggregation.")
	tmp = df.copy()
	tmp["admit_date"] = pd.to_datetime(tmp["admit_date"], errors="coerce")
	daily = (
		tmp.dropna(subset=["admit_date"])  # type: ignore[arg-type]
		.groupby(tmp["admit_date"].dt.date)
		.size()
		.rename("admissions")
		.to_frame()
		.reset_index(names="date")
	)
	daily["date"] = pd.to_datetime(daily["date"])  # normalize to Timestamp
	return daily.sort_values("date").reset_index(drop=True)
