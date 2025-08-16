# Data for Good – AQI Forecast

Notebook: `notebooks/aqi_forecast.ipynb`

Sample data:
- `data/aqi_city.csv` with columns: `date, aqi, temp, wind, humidity`
- Generate synthetic data:
```bash
python scripts/generate_aqi_sample.py
```

Suggested baselines in the notebook:
- Persistence (AQI[t+1] = AQI[t])
- SARIMA / ARIMA
- Regressors (RandomForest/XGBoost) on lagged features
- Optional: LSTM (PyTorch)

KPIs:
- RMSE / MAE for next‑day AQI
- Directional accuracy (improvement/worsening)

