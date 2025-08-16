import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def make_seasonal_series(n=500, season=7, seed=42):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    trend = 0.05 * t
    seas = 5 * np.sin(2 * np.pi * t / season)
    noise = rng.normal(0, 2.0, size=n)
    y = 50 + trend + seas + noise
    idx = pd.date_range("2022-01-01", periods=n, freq="D")
    return pd.Series(y, index=idx, name="sales")

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))) * 100.0

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def main():
    s = make_seasonal_series(n=400, season=7, seed=11)
    train = s.iloc[:-56]
    test = s.iloc[-56:]

    # SARIMA simple (p,d,q)(P,D,Q)s
    order = (2, 1, 2)
    seasonal_order = (1, 1, 1, 7)

    model = SARIMAX(train, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    fc = res.forecast(steps=len(test))

    print(f"RMSE: {rmse(test.values, fc.values):.3f}")
    print(f"MAPE: {mape(test.values, fc.values):.2f}%")

    # Plot (matplotlib, single plot, no explicit colors)
    plt.figure()
    plt.plot(train.index, train.values, label="train")
    plt.plot(test.index, test.values, label="test")
    plt.plot(test.index, fc.values, label="forecast")
    plt.title("SARIMA forecast vs actuals")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
