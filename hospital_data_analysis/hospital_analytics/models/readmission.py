from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

try:
	from xgboost import XGBClassifier  # type: ignore
	has_xgb = True
except Exception:
	has_xgb = False


@dataclass
class ModelResult:
	name: str
	roc_auc: float
	accuracy: float
	f1: float
	pipeline: Pipeline


def build_preprocess_pipeline(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
	numeric_transformer = Pipeline(steps=[("scaler", StandardScaler(with_mean=False))])
	categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse=True)
	preprocessor = ColumnTransformer(
		transformers=[
			("num", numeric_transformer, numeric_cols),
			("cat", categorical_transformer, categorical_cols),
		]
	)
	return preprocessor


def train_and_eval_models(
	X: pd.DataFrame,
	y: pd.Series,
	numeric_cols: list[str],
	categorical_cols: list[str],
	random_state: int = 42,
) -> list[ModelResult]:
	results: list[ModelResult] = []

	preprocessor = build_preprocess_pipeline(numeric_cols, categorical_cols)

	models: list[tuple[str, object]] = [
		("LogReg", LogisticRegression(max_iter=200, n_jobs=None, random_state=random_state)),
		("RF", RandomForestClassifier(n_estimators=300, max_depth=None, random_state=random_state)),
	]
	if has_xgb:
		models.append(
			(
				"XGB",
				XGBClassifier(
					n_estimators=400,
					max_depth=5,
					learning_rate=0.05,
					subsample=0.9,
					colsample_bytree=0.9,
					eval_metric="logloss",
					random_state=random_state,
				)
			),
		)

	from sklearn.model_selection import train_test_split

	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.25, random_state=random_state, stratify=y
	)

	for name, estimator in models:
		pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])
		pipe.fit(X_train, y_train)
		pred_proba = pipe.predict_proba(X_test)[:, 1]
		pred_label = (pred_proba >= 0.5).astype(int)

		roc = roc_auc_score(y_test, pred_proba)
		acc = accuracy_score(y_test, pred_label)
		f1 = f1_score(y_test, pred_label)
		results.append(
			ModelResult(name=name, roc_auc=float(roc), accuracy=float(acc), f1=float(f1), pipeline=pipe)
		)

	return results
