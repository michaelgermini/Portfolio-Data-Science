import numpy as np
import pandas as pd

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))) * 100.0

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def main():
    try:
        from prophet import Prophet
    except Exception as e:
        print("Prophet n'est pas installé. Faites: pip install prophet")
        return

    # Série simple quotidienne
    n = 500
    t = np.arange(n)
    y = 100 + 0.05 * t + 5 * np.sin(2 * np.pi * t / 7) + np.random.default_rng(0).normal(0, 2, n)
    df = pd.DataFrame({"ds": pd.date_range("2022-01-01", periods=n, freq="D"), "y": y})

    train = df.iloc[:-60]
    test = df.iloc[-60:]

    m = Prophet(seasonality_mode="additive", weekly_seasonality=True, yearly_seasonality=False, daily_seasonality=False)
    m.fit(train)
    future = m.make_future_dataframe(periods=len(test), freq="D")
    fcst = m.predict(future)

    y_pred = fcst.set_index("ds").loc[test["ds"], "yhat"].values
    print(f"RMSE: {rmse(test['y'].values, y_pred):.3f}")
    print(f"MAPE: {mape(test['y'].values, y_pred):.2f}%")

if __name__ == "__main__":
    main()
