import numpy as np

def series_to_supervised(series, window=14):
    X, y = [], []
    for i in range(len(series) - window):
        X.append(series[i:i+window])
        y.append(series[i+window])
    X = np.array(X).reshape(-1, window, 1)
    y = np.array(y).reshape(-1, 1)
    return X, y

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))) * 100.0

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def main():
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models
    except Exception as e:
        print("TensorFlow n'est pas installé. Faites: pip install tensorflow")
        return

    # Série synthétique
    n = 600
    t = np.arange(n)
    y = 200 + 0.05 * t + 10 * np.sin(2 * np.pi * t / 7) + np.random.default_rng(1).normal(0, 3, n)
    window = 21

    X, Y = series_to_supervised(y, window=window)
    split = int(0.85 * len(X))
    X_train, X_test = X[:split], X[split:]
    Y_train, Y_test = Y[:split], Y[split:]

    model = models.Sequential([
        layers.Input(shape=(window, 1)),
        layers.LSTM(64, return_sequences=False),
        layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, Y_train, epochs=10, batch_size=32, verbose=1, validation_split=0.1)

    y_pred = model.predict(X_test).reshape(-1)
    print(f"RMSE: {rmse(Y_test.reshape(-1), y_pred):.3f}")
    print(f"MAPE: {mape(Y_test.reshape(-1), y_pred):.2f}%")

if __name__ == "__main__":
    main()
