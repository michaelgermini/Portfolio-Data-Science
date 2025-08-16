from __future__ import annotations

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


def fit_sarima(daily: pd.DataFrame, order=(1, 0, 1), seasonal_order=(0, 1, 1, 7)):
	series = daily.set_index("date")["admissions"].asfreq("D").fillna(method="ffill")
	model = SARIMAX(series, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
	res = model.fit(disp=False)
	return res


def forecast(res, steps: int = 30) -> pd.DataFrame:
	fc = res.get_forecast(steps=steps)
	df = fc.summary_frame()
	return df.reset_index().rename(columns={"index": "date", "mean": "forecast"})
